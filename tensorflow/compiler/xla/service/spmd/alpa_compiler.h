#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_ALPA_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_ALPA_COMPILER_H_

#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"

namespace xla {
namespace spmd {

// Run the auto sharding pass to add sharding anotations
// for each HLO instruction.
Status RunAutoShardingPass(HloModule* hlo_module,
                           const CompileOptions& options);

// Run the SPMD partitioner pass.
Status RunSpmdPartitionerPass(HloModule* hlo_module,
                              const CompileOptions& options);

// Set the shardings for output tensors.
Status SetHloModuleOutputShardings(HloModule* hlo_module,
                                   const std::vector<OpSharding>& op_shardings);

// Set the shardings for input tensors.
Status SetHloModuleInputShardings(HloModule* hlo_module,
                                  const std::vector<OpSharding>& op_shardings);

// Run the SPMD partitioner pass with optimization of delaying grad_acc comm.
Status RunCommDelaySpmdPartitionerPass(vector<HloModule*> hlo_modules,
                                       const CompileOptions& options);

};  // namespace spmd
};  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_ALPA_COMPILER_H_
