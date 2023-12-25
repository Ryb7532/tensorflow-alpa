#include "tensorflow/compiler/xla/service/spmd/grad_acc_rewrite.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/pass_context.h"
#include "tensorflow/compiler/xla/service/spmd/spmd_partitioner_util.h"

namespace xla {
namespace spmd {

HloInstruction* GetAllReduce(HloInstruction* src) {
  auto opcode = src->opcode();
  if (opcode == HloOpcode::kAllReduce) {
    return src;
  } else if (opcode == HloOpcode::kConvert || opcode == HloOpcode::kReshape ||
             opcode == HloOpcode::kCopy || opcode == HloOpcode::kBitcast ||
             opcode == HloOpcode::kTranspose) {
    return GetAllReduce(src->mutable_operand(0));
  } else if (opcode == HloOpcode::kMultiply) {
    HloInstruction* lhs = GetAllReduce(src->mutable_operand(0));
    HloInstruction* rhs = GetAllReduce(src->mutable_operand(1));

    if (lhs != nullptr && rhs == nullptr) {
      return lhs;
    } else if (lhs == nullptr && rhs != nullptr) {
      return rhs;
    }
  }
  return nullptr;
}

HloInstruction* MaybeConvert(HloInstruction* src, const Shape& dst_shape) {
  if (ShapeUtil::SameElementType(src->shape(), dst_shape)) {
    return src;
  }
  const Shape new_shape =
      ShapeUtil::ChangeElementType(src->shape(), dst_shape.element_type());
  return src->parent()->AddInstruction(
      HloInstruction::CreateConvert(new_shape, src));
}

HloInstruction* MaybeReshapeConvert(HloInstruction* src,
                                    const Shape& dst_shape) {
  if (ShapeUtil::Compatible(src->shape(), dst_shape)) {
    return src;
  }
  return src->parent()->AddInstruction(
      HloInstruction::CreateReshape(dst_shape, MaybeConvert(src, dst_shape)));
}

HloInstruction::InstructionVector MaybeReshapeConvertTuple(
    const HloInstruction::InstructionVector& src, const Shape& dst_shape) {
  HloInstruction::InstructionVector ret;
  if (dst_shape.IsTuple()) {
    CHECK(dst_shape.tuple_shapes_size() == src.size());
    for (int i = 0; i < src.size(); ++i) {
      ret.push_back(MaybeReshapeConvert(src[i], dst_shape.tuple_shapes(i)));
    }
  } else {
    CHECK(dst_shape.IsArray() && src.size() == 1);
    ret.push_back(MaybeReshapeConvert(src[0], dst_shape));
  }
  return ret;
}

StatusOr<bool> GradAccRewrite::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!pass_context::GetBool("auto_sharding::rewrite_for_grad_acc", false)) {
    return false;
  }

  // std::cerr << "===== Enter GradAccRewrite =====" << std::endl;
  // std::cerr << module->ToString();
  // std::cerr << "=====================================" << std::endl;

  auto indices = pass_context::GetIntVector("auto_sharding::rewrite_indices");

  HloComputation* entry = module->entry_computation();
  HloInstruction* output_tuple = entry->root_instruction();

  std::vector<HloInstruction*> to_remove;

  for (size_t i : indices) {
    HloInstruction* add_ins = output_tuple->mutable_operand(i);
    if (add_ins->opcode() != HloOpcode::kAdd) {
      continue;
    }

    HloInstruction* allreduce_ins = GetAllReduce(add_ins->mutable_operand(1));

    if (allreduce_ins == nullptr || allreduce_ins->users().size() != 1) {
      continue;
    }

    CHECK_EQ(allreduce_ins->operand_count(), 1);

    HloInstruction* allreduce_user = allreduce_ins->users().front();

    for (size_t i = 0; i < allreduce_user->operand_count(); ++i) {
      if (allreduce_user->operand(i) == allreduce_ins) {
        allreduce_user->ReplaceOperandWith(
            i, MaybeReshapeConvert(allreduce_ins->mutable_operand(0),
                                   allreduce_user->operand(i)->shape()));
      }
    }

    // allreduce_ins->ReplaceOperandWith(0, add_ins);
    allreduce_ins->ReplaceOperandWith(
        0, MaybeReshapeConvert(add_ins, allreduce_ins->shape()));
    output_tuple->ReplaceOperandWith(
        i, MaybeReshapeConvert(allreduce_ins, add_ins->shape()));
    allreduce_ins->set_metadata_op_name(kSkippableAllReduce);

    if (!ShapeUtil::SameElementType(allreduce_ins->shape(), add_ins->shape())) {
      // Fix type mismatch
      auto old_allreduce = Cast<HloAllReduceInstruction>(allreduce_ins);
      const Shape& new_shape = add_ins->shape();
      auto new_allreduce =
          entry->AddInstruction(HloInstruction::CreateAllReduce(
              new_shape,
              MaybeReshapeConvertTuple(old_allreduce->operands(), new_shape),
              MakeBinaryAdd(new_shape.element_type(), entry->parent()),
              old_allreduce->replica_groups(),
              old_allreduce->constrain_layout(), old_allreduce->channel_id(),
              old_allreduce->use_global_device_ids()));
      new_allreduce->set_metadata(old_allreduce->metadata());
      old_allreduce->ReplaceAllUsesWith(
          MaybeReshapeConvert(new_allreduce, old_allreduce->shape()));
      to_remove.push_back(old_allreduce);
    }
  }

  for (auto ins : to_remove) {
    entry->RemoveInstruction(ins);
  }

  // std::cerr << "===== Exit GradAccRewrite =====" << std::endl;
  // std::cerr << module->ToString();
  // std::cerr << "=====================================" << std::endl;

  return true;
}

std::string GetGradSyncChannelIds(const HloModule* module) {
  std::string ret = ".";
  for (auto inst : module->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kAllReduce &&
        inst->metadata().op_name() == kSkippableAllReduce) {
      ret += std::to_string(inst->channel_id().value()) + ".";
    }
  }

  return ret;
}

/***** Added by Ryb7532 *****/
StatusOr<bool> GradAccCommDelay::Run(HloModule* backward_hlo, HloModule* applygrad_hlo) {
  if (!pass_context::GetBool("auto_sharding::rewrite_for_grad_acc", false)) {
    return false;
  }

  // for debug
  std::cerr << "===== Enter GradAccRewrite =====" << std::endl;
  std::cerr << module->ToString();
  std::cerr << "=====================================" << std::endl;

  auto output_indices = pass_context::GetIntVector("auto_sharding::rewrite_indices");
  auto input_indices = pass_context::GetIntVector("auto_sharding::rewrite_applygrad_indices");

  HloComputation* backward_entry = backward_hlo->entry_computation();
  HloInstruction* output_tuple = backward_entry->root_instruction();
  HloComputation* applygrad_entry = applygrad_hlo->entry_computation();

  std::vector<HloInstruction*> to_remove_in_backward, to_remove_in_applygrad;

  for (size_t j = 0; j<indices.size(); j++) {
    int64_t out_index = output_indices[j], in_index = input_indices[j];
    HloInstruction* add_ins = output_tuple->mutable_operand(out_index);
    if (add_ins->opcode() != HloOpcode::kAdd) {
      continue;
    }

    HloInstruction* allreduce_ins = GetAllReduce(add_ins->mutable_operand(1));

    if (allreduce_ins == nullptr || allreduce_ins->users().size() != 1) {
      continue;
    }

    CHECK_EQ(allreduce_ins->operand_count(), 1);

    HloInstruction* allreduce_user = allreduce_ins->users().front();

    // detach allreduce_ins
    for (size_t i = 0; i < allreduce_user->operand_count(); ++i) {
      if (allreduce_user->operand(i) == allreduce_ins) {
        allreduce_user->ReplaceOperandWith(
            i, MaybeReshapeConvert(allreduce_ins->mutable_operand(0),
                                   allreduce_user->operand(i)->shape()));
      }
    }

    if (in_index == -1) {
      // insert allreduce_ins between add_ins and output
      // allreduce_ins->ReplaceOperandWith(0, add_ins);
      allreduce_ins->ReplaceOperandWith(
          0, MaybeReshapeConvert(add_ins, allreduce_ins->shape()));
      output_tuple->ReplaceOperandWith(
          out_index, MaybeReshapeConvert(allreduce_ins, add_ins->shape()));
      allreduce_ins->set_metadata_op_name(kSkippableAllReduce);

      if (!ShapeUtil::SameElementType(allreduce_ins->shape(), add_ins->shape())) {
        // Fix type mismatch
        auto old_allreduce = Cast<HloAllReduceInstruction>(allreduce_ins);
        const Shape& new_shape = add_ins->shape();
        auto new_allreduce =
            backward_entry->AddInstruction(HloInstruction::CreateAllReduce(
                new_shape,
                MaybeReshapeConvertTuple(old_allreduce->operands(), new_shape),
                MakeBinaryAdd(new_shape.element_type(), backward_entry->parent()),
                old_allreduce->replica_groups(),
                old_allreduce->constrain_layout(), old_allreduce->channel_id(),
                old_allreduce->use_global_device_ids()));
        new_allreduce->set_metadata(old_allreduce->metadata());
        old_allreduce->ReplaceAllUsesWith(
            MaybeReshapeConvert(new_allreduce, old_allreduce->shape()));
        to_remove_in_backward.push_back(old_allreduce);
      }
    } else {
      // TODO: remove these lines (left for debug)
      // allreduce_ins->ReplaceOperandWith(0, add_ins);
      allreduce_ins->ReplaceOperandWith(
          0, MaybeReshapeConvert(add_ins, allreduce_ins->shape()));
      // output_tuple->ReplaceOperandWith(
      //     out_index, MaybeReshapeConvert(allreduce_ins, add_ins->shape()));
      allreduce_ins->set_metadata_op_name(kAllReduceToBeRemoved);

      HloInstruction* param_ins = applygrad_entry->parameter_instruction(in_index);

      TF_RETURN_IF_ERROR(ShapeUtil::SameElementType(add_ins->shape(), param_ins->shape()));
      auto old_allreduce = Cast<HloAllReduceInstruction>(allreduce_ins);
      const Shape& new_shape = old_allreduce->shape();
      HloInstruction::InstructionVector new_operands;
      new_operands.push_back(param_ins);
      auto new_allreduce =
          applygrad_entry->AddInstruction(HloInstruction::CreateAllReduce(
              new_shape,
              MaybeReshapeConvertTuple(new_operands, new_shape),
              MakeBinaryAdd(new_shape.element_type(), applygrad_entry->parent()),
              old_allreduce->replica_groups(),
              old_allreduce->constrain_layout(), old_allreduce->channel_id(),
              old_allreduce->use_global_device_ids()));
      new_allreduce->set_metadata(old_allreduce->metadata());
      param_ins->ReplaceAllUsesWith(
          MaybeReshapeConvert(new_allreduce, param_ins->shape()));

      // TODO: uncomment out (left for debug)
      // to_remove_in_backward.push_back(old_allreduce);
      allreduce_ins = new_allreduce;

      if (!ShapeUtil::SameElementType(allreduce_ins->shape(), param_ins->shape())) {
        // Fix type mismatch
        auto old_allreduce = Cast<HloAllReduceInstruction>(allreduce_ins);
        const Shape& new_shape = param_ins->shape();
        auto new_allreduce =
            applygrad_entry->AddInstruction(HloInstruction::CreateAllReduce(
                new_shape,
                MaybeReshapeConvertTuple(old_allreduce->operands(), new_shape),
                MakeBinaryAdd(new_shape.element_type(), applygrad_entry->parent()),
                old_allreduce->replica_groups(),
                old_allreduce->constrain_layout(), old_allreduce->channel_id(),
                old_allreduce->use_global_device_ids()));
        new_allreduce->set_metadata(old_allreduce->metadata());
        old_allreduce->ReplaceAllUsesWith(
            MaybeReshapeConvert(new_allreduce, old_allreduce->shape()));
        to_remove_in_applygrad.push_back(old_allreduce);
      }
    }
  }

  for (auto ins : to_remove_in_backward) {
    backward_entry->RemoveInstruction(ins);
  }

  for (auto ins : to_remove_in_applygrad) {
    applygrad_entry->RemoveInstruction(ins);
  }

  // for debug
  std::cerr << "===== Exit GradAccRewrite =====" << std::endl;
  std::cerr << module->ToString();
  std::cerr << "=====================================" << std::endl;

  return true;
}

}  // namespace spmd
}  // namespace xla
