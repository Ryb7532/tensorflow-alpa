/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// SWIG typemaps and declarations for building, compiling, and
// executing XLA computations, wrapping most of what is declared in
// local_computation_builder.h.
//
// The typemaps below implement/assert the following correspondences
// (with elaborations below):
//
//    C++                                  Python
// -------------------------------------+---------------------------------------
//  Span<int64>                        <-  sequence of int
//  vector<int>                        ->  sequence of int
//  Span<LocalOp>                      <-  sequence of LocalOp
//  Literal                            <-> (nested tuple of) numpy ndarray
//  std::vector<Literal>               <-  sequence of (nested tuple of) ndarray
//  Shape                               -> pair holding (dtype, dimensions)
//                                     <-  object duck-typed as xla_client.Shape
//  ProgramShape                       ->  pair of ([arg_shapes], ret_shape)
//  std::vector<Shape>                 <-  sequence of xla_client.Shape objects
//  PrimitiveType                      <-  int
//  Span<pair<int64, in64>>            <-  sequence of int pairs
//  PaddingConfig proto                <-  corresponding Python proto
//  ConvolutionDimensionNumbers proto  <-  corresponding Python proto
//  DotDimensionNumbers proto          <-  corresponding Python proto
//  GatherDimensionNumbers proto       <-  corresponding Python proto
//  ScatterDimensionNumbers proto      <-  corresponding Python proto
//  Span<ReplicaGroup proto>           <-  sequence of ReplicaGroup Python proto
//
// Arrows indicate whether a conversion only ever occurs in one
// direction, or whether it is maintained bidirectionally.
//
// The Python objects corresponding to C++ Literals have the type:
//
//   T = ndarray | (T, ...)
//
// where a terminal numpy ndarray translates to a Literal with a
// non-tuple Shape, an XLA primitive element type corresponding to the
// ndarray's dtype. Meanwhile, a non-terminal "tuple of T" translates
// to a tuple-shaped Literal whose tuple components are translated
// recursively. For example, if x is a numpy ndarray in Python, with
// shape (2, 3) and dtype of dtype('float32'), then x translates to a
// Literal with rank 2, dimension 2 and 3, and XLA primitive type
// F32. Meanwhile,
//
//   (x, (x, x), (x,)),
//
// translates to a tuple-shaped XLA Literal, whose component subshapes
// are a 2x3 F32-shaped literal followed by two tuple-shaped literals.
//
// Shapes output by C++ become Python objects with the type:
//
//   T            = (dtype, S)
//   S            = DIMENSIONS | TUPLE_SHAPES
//   DIMENSIONS   = (int, ...)
//   TUPLE_SHAPES = (T, ...)
//
// In the pair described by the T rule, the terminal dtype determines
// whether S expands as DIMENSIONS or TUPLE_SHAPES. Namely if it is
// dtype('O'), numpy's object dtype, the structure represents a tuple
// shape and the expansion of the non-terminal S is
// TUPLE_SHAPES. Otherwise, dtype describes a primitive element type
// and S expands into DIMENSIONS giving dimension sizes. For example:
//
//   (dtype('float32'), (3, 5, 7))
//
// describes a 3x5x7 array of F32s, and
//
//   (dtype('O'), ((dtype('float32'), (2, 3)),
//                 (dtype('float64'), (4, 5))))
//
// describes a tuple shape with two subshapes: the first a 2x3 F32,
// and the other a 4x5 F64.
//
// The Python int corresponding to a PrimitiveType enum must be valid
// per xla_data.proto (e.g. xla_data.PRED, xla_data.F32).
//
// The SWIG object wrappers generated by this file are not intended
// for end use, but rather for internal use in the Python XLA client,
// xla_client.py.
//
// One central reason for the Python-side indirection is that the
// Python-side objects produced by the typemaps in this file are
// further packaged up by xla_client before being passed on. For
// instance, the Python pair produced for a C++ Shape is further
// wrapped in a Python class (xla_client.Shape) so as not to expose
// the raw pair externally.
//
// Other SWIG object wrappers (e.g. of Computation) are further
// wrapped by xla_client in order to set up a custom destructor that
// triggers memory deallocation on the C++ side.

%module(threads="1") local_computation_builder

// Keep the GIL except where explicitly specified.
%nothread;

%include "tensorflow/python/platform/base.i"

%{
// Must be included first
#include "tensorflow/python/lib/core/numpy.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/python/numpy_bridge.h"
#include "tensorflow/compiler/xla/python/local_computation_builder.h"

using namespace xla;
using namespace xla::swig;

namespace xla {

namespace swig {

bool GetIntAttr(PyObject* o, const char* field, int64* result) {
  PyObject* fo = PyObject_GetAttrString(o, field);
  if (!fo) {
    return false;
  }
  const int64 value = numpy::PyIntOrPyLongToLong(fo);
  if (value == -1 && PyErr_Occurred()) {
    Py_DECREF(fo);
    return false;
  }
  Py_DECREF(fo);
  *result = value;
  return true;
}

// Returns "ok"; true if there is no error, false if there was an error.
bool HandleStringAttribute(PyObject* o,
                           const char* attr_name,
                           std::function<void(string s)> f) {
  if (!PyObject_HasAttrString(o, attr_name)) {
    return true;  // It's ok for the object to not have the attribute.
  }
  PyObject* attr = PyObject_GetAttrString(o, attr_name);
  if (attr == nullptr) {
    return false;  // An error occurred getting the attribute.
  }
  if (attr == Py_None) {
    Py_DECREF(attr);
    return true;  // The attribute is None, which we consider ok.
  }
  if (!PyString_Check(attr)) {
    string message = absl::StrFormat("%s must be a string or none; got %s",
        attr_name, numpy::PyObjectCppRepr(attr));
    PyErr_SetString(PyExc_TypeError, message.c_str());
    Py_DECREF(attr);
    return false;  // Type error, not ok.
  }
  f(PyString_AsString(attr));
  Py_DECREF(attr);
  return true;  // Handled string attribute, ok!
}

bool HandleRepeatedInt64Attribute(
    PyObject* o, const char* attr_name,
    tensorflow::protobuf::RepeatedField<tensorflow::protobuf_int64>* field) {
  PyObject* seq = PyObject_GetAttrString(o, attr_name);
  if (!seq) {
    return false;
  }

  int length = PySequence_Size(seq);
  if (length == -1) {
    Py_DECREF(seq);
    return false;
  }

  for (int i = 0; i < length; ++i) {
    PyObject* item = PySequence_GetItem(seq, i);
    if (!item) {
      Py_DECREF(seq);
      return false;
    }
    const int64 dimension = numpy::PyIntOrPyLongToLong(item);
    if (dimension == -1 && PyErr_Occurred()) {
      Py_DECREF(item);
      Py_DECREF(seq);
      return false;
    }
    *field->Add() = dimension;
    Py_DECREF(item);
  }
  Py_DECREF(seq);
  return true;
}

}  // namespace swig
}  // namespace xla
%}

// Required to use PyArray_* functions.
%init %{
tensorflow::ImportNumpy();
%}

// Basic types


%typemap(out) std::vector<int> {
  PyObject* out = PyList_New($1.size());
  for (int i = 0; i < $1.size(); ++i) {
    PyList_SET_ITEM(out, i, PyInt_FromLong($1[i]));
  }
  $result = out;
}

%typemap(out) StatusOr<bool> {
  if ($1.ok()) {
    $result = PyBool_FromLong($1.ConsumeValueOrDie());
  } else {
    PyErr_SetString(PyExc_RuntimeError, $1.status().ToString().c_str());
    SWIG_fail;
  }
}

%typemap(out) Status {
  if (!$1.ok()) {
    PyErr_SetString(
        PyExc_RuntimeError, $1.ToString().c_str());
    SWIG_fail;
  }
  Py_INCREF(Py_None);
  $result = Py_None;
}

%typemap(in) absl::Span<const int64>
    (std::vector<int64> temps) {
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError, "Argument is not a sequence");
    SWIG_fail;
  }
  const int size = PySequence_Size($input);
  temps.resize(size);
  for (int i = 0; i < size; ++i) {
    PyObject* o = PySequence_GetItem($input, i);
    PyObject* py_int = numpy::PyNumberToPyInt(o);
    if (!py_int) {
      PyErr_SetString(
          PyExc_TypeError,
          "Argument sequence element cannot be converted to int");
      Py_DECREF(o);
      SWIG_fail;
    }
    temps[i] = numpy::PyIntOrPyLongToLong(py_int);
    if (temps[i] == -1 && PyErr_Occurred()) {
      Py_DECREF(py_int);
      Py_DECREF(o);
      SWIG_fail;
    }
    Py_DECREF(py_int);
    Py_DECREF(o);
  }
  $1 = temps;
}

// Computation builder types

%typemap(in) absl::Span<const xla::swig::LocalOp>(
      std::vector<LocalOp> temps) {
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError, "Argument is not a sequence");
    SWIG_fail;
  }
  const int size = PySequence_Size($input);
  for (int i = 0; i < size; ++i) {
    PyObject* o = PySequence_GetItem($input, i);
    LocalOp* op;
    if ((SWIG_ConvertPtr(o, (void**)&op, $descriptor(xla::swig::LocalOp*),
                         SWIG_POINTER_EXCEPTION)) == -1) {
      SWIG_fail;
    }
    temps.push_back(*op);
    Py_DECREF(o);
  }
  $1 = temps;
}

// Computation and buffer/allocation types

%typemap(out) StatusOr<xla::swig::LocalClient> {
  if ($1.ok()) {
    xla::swig::LocalClient value = $1.ValueOrDie();
    {
      auto $1 = value;
      $typemap(out, xla::swig::LocalClient)
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, $1.status().ToString().c_str());
    SWIG_fail;
  }
}

%typemap(out) StatusOr<xla::swig::LocalExecutable*> {
  if ($1.ok()) {
    auto* value = $1.ValueOrDie();
    {
      auto* $1 = value;
      $typemap(out, xla::swig::LocalExecutable*)
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, $1.status().ToString().c_str());
    SWIG_fail;
  }
}

%typemap(out) StatusOr<xla::swig::XrtExecutable*> {
  if ($1.ok()) {
    auto* value = $1.ValueOrDie();
    {
      auto* $1 = value;
      $typemap(out, xla::swig::XrtExecutable*)
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, $1.status().ToString().c_str());
    SWIG_fail;
  }
}

%typemap(out) StatusOr<xla::swig::LocalShapedBuffer*> {
  if ($1.ok()) {
    auto* value = $1.ValueOrDie();
    {
      auto* $1 = value;
      $typemap(out, xla::swig::LocalShapedBuffer*)
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, $1.status().ToString().c_str());
    SWIG_fail;
  }
}

%typemap(out) StatusOr<xla::swig::LocalShapedBufferTuple*> {
  if ($1.ok()) {
    auto* value = $1.ValueOrDie();
    {
      auto* $1 = value;
      $typemap(out, xla::swig::LocalShapedBufferTuple*)
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, $1.status().ToString().c_str());
    SWIG_fail;
  }
}

%typemap(out) StatusOr<xla::swig::XrtAllocation*> {
  if ($1.ok()) {
    auto* value = $1.ValueOrDie();
    {
      auto* $1 = value;
      $typemap(out, xla::swig::XrtAllocation*)
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, $1.status().ToString().c_str());
    SWIG_fail;
  }
}

%typemap(out) StatusOr<xla::swig::XrtAllocationTuple*> {
  if ($1.ok()) {
    auto* value = $1.ValueOrDie();
    {
      auto* $1 = value;
      $typemap(out, xla::swig::XrtAllocationTuple*)
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, $1.status().ToString().c_str());
    SWIG_fail;
  }
}

%typemap(out) StatusOr<xla::swig::Computation*> {
  if ($1.ok()) {
    auto* value = $1.ValueOrDie();
    {
      auto* $1 = value;
      $typemap(out, xla::swig::Computation*)
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, $1.status().ToString().c_str());
    SWIG_fail;
  }
}

%typemap(in) absl::Span<xla::swig::LocalShapedBuffer* const>
    (std::vector<LocalShapedBuffer*> temps) {
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError, "Argument is not a sequence");
    SWIG_fail;
  }
  const int size = PySequence_Size($input);
  temps.reserve(size);
  for (int i = 0; i < size; ++i) {
    PyObject* o = PySequence_GetItem($input, i);
    LocalShapedBuffer* lsbp;
    if ((SWIG_ConvertPtr(o, (void**) &lsbp, $descriptor(xla::swig::LocalShapedBuffer*),
                         SWIG_POINTER_EXCEPTION)) == -1) {
      SWIG_fail;
    }
    temps.push_back(lsbp);
    Py_DECREF(o);
  }
  $1 = temps;
}

%typemap(in) absl::Span<const std::vector<xla::swig::LocalShapedBuffer*> >
    (std::vector<std::vector<LocalShapedBuffer*> > temps) {
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError, "Argument is not a sequence");
    SWIG_fail;
  }
  const int size = PySequence_Size($input);
  temps.reserve(size);
  for (int i = 0; i < size; ++i) {
    PyObject* o = PySequence_GetItem($input, i);
    std::vector<LocalShapedBuffer*> vec;
    const int vec_size = PySequence_Size(o);
    vec.reserve(vec_size);
    for (int j = 0; j < vec_size; ++j) {
      PyObject* vec_elt = PySequence_GetItem(o, j);
      LocalShapedBuffer* lsbp;
      if ((SWIG_ConvertPtr(vec_elt, (void**) &lsbp, $descriptor(xla::swig::LocalShapedBuffer*),
                           SWIG_POINTER_EXCEPTION)) == -1) {
        Py_DECREF(vec_elt);
        Py_DECREF(o);
        SWIG_fail;
      }
      vec.push_back(lsbp);
      Py_DECREF(vec_elt);
    }
    temps.push_back(vec);
    Py_DECREF(o);
  }
  $1 = temps;
}

%typemap(in) absl::Span<xla::swig::XrtAllocation* const>
    (std::vector<XrtAllocation*> temps) {
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError, "Argument is not a sequence");
    SWIG_fail;
  }
  const int size = PySequence_Size($input);
  temps.reserve(size);
  for (int i = 0; i < size; ++i) {
    PyObject* o = PySequence_GetItem($input, i);
    XrtAllocation* xrta;
    if ((SWIG_ConvertPtr(o, (void**) &xrta, $descriptor(xla::swig::XrtAllocation*),
                         SWIG_POINTER_EXCEPTION)) == -1) {
      SWIG_fail;
    }
    temps.push_back(xrta);
    Py_DECREF(o);
  }
  $1 = temps;
}

// Literal

%typemap(in) const Literal& (StatusOr<Literal> literal_status) {
  literal_status = numpy::XlaLiteralFromPyObject($input);
  if (!literal_status.ok()) {
    PyErr_SetString(PyExc_RuntimeError, literal_status.status().ToString().c_str());
    SWIG_fail;
  }
  $1 = &literal_status.ValueOrDie();
}

%typemap(out) Literal (StatusOr<numpy::Safe_PyObjectPtr> obj_status) {
  obj_status = numpy::PyObjectFromXlaLiteral(*$1);
  if (!obj_status.ok()) {
    PyErr_SetString(PyExc_RuntimeError, obj_status.status().ToString().c_str());
    SWIG_fail;
  }
  $result = obj_status.ValueOrDie().release();
}

%typemap(out) StatusOr<Literal> (StatusOr<numpy::Safe_PyObjectPtr> obj_status) {
  if (!$1.ok()) {
    PyErr_SetString(PyExc_RuntimeError, $1.status().ToString().c_str());
    SWIG_fail;
  }
  obj_status = numpy::PyObjectFromXlaLiteral($1.ValueOrDie());
  if (!obj_status.ok()) {
    PyErr_SetString(PyExc_RuntimeError, obj_status.status().ToString().c_str());
    SWIG_fail;
  }
  $result = obj_status.ValueOrDie().release();
}

%typemap(in) const std::vector<Literal>& (std::vector<Literal> temps) {
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError, "Argument is not a sequence");
    SWIG_fail;
  }
  const int size = PySequence_Size($input);
  for (int i = 0; i < size; ++i) {
    PyObject* o = PySequence_GetItem($input, i);
    StatusOr<Literal> literal_status = numpy::XlaLiteralFromPyObject(o);
    if (!literal_status.ok()) {
      PyErr_SetString(PyExc_RuntimeError, literal_status.status().ToString().c_str());
      Py_DECREF(o);
      SWIG_fail;
    }
    temps.push_back(literal_status.ConsumeValueOrDie());
    Py_DECREF(o);
  }
  $1 = &temps;
}

// OpMetadata

%typemap(in) const OpMetadata& (OpMetadata temp) {
  StatusOr<OpMetadata> statusor = numpy::OpMetadataFromPyObject($input);
  if (!statusor.ok()) {
    PyErr_SetString(PyExc_RuntimeError, statusor.status().ToString().c_str());
    SWIG_fail;
  }
  temp = std::move(statusor).ValueOrDie();
  $1 = &temp;
}

// Shape

%typemap(out) const Shape& {
  $result = numpy::PyShapeInfoFromXlaShape(*$1).release();
}

%typemap(out) StatusOr<Shape> {
  if ($1.ok()) {
    $result = numpy::PyShapeInfoFromXlaShape($1.ConsumeValueOrDie()).release();
  } else {
    PyErr_SetString(PyExc_RuntimeError, $1.status().ToString().c_str());
    SWIG_fail;
  }
}


%typemap(out) StatusOr<ProgramShape> {
  if ($1.ok()) {
    $result = numpy::PyProgramShapeInfoFromXlaProgramShape(
        $1.ConsumeValueOrDie()).release();
  } else {
    PyErr_SetString(PyExc_RuntimeError, $1.status().ToString().c_str());
    SWIG_fail;
  }
}


%typemap(in) const Shape& (Shape temp) {
  StatusOr<Shape> statusor = numpy::XlaShapeFromPyShape($input);
  if (!statusor.ok()) {
    PyErr_SetString(PyExc_RuntimeError, statusor.status().ToString().c_str());
    SWIG_fail;
  }
  temp = std::move(statusor).ValueOrDie();
  $1 = &temp;
}

%typemap(in) const absl::optional<Shape>& (
    absl::optional<Shape> temp) {
  if ($input == Py_None) {
    temp = absl::nullopt;
    $1 = &temp;
  } else {
    StatusOr<Shape> statusor = numpy::XlaShapeFromPyShape($input);
    if (!statusor.ok()) {
      PyErr_SetString(PyExc_RuntimeError, statusor.status().ToString().c_str());
      SWIG_fail;
    }
    temp = std::move(statusor).ValueOrDie();
    $1 = &temp;
  }
}

%typemap(out) std::unique_ptr<Shape> {
  $result = numpy::PyShapeInfoFromXlaShape(*$1).release();
}

%typemap(in) const std::vector<Shape>& (std::vector<Shape> temps) {
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError, "Argument is not a sequence");
    SWIG_fail;
  }
  const int size = PySequence_Size($input);
  for (int i = 0; i < size; ++i) {
    PyObject* o = PySequence_GetItem($input, i);
    StatusOr<Shape> statusor = numpy::XlaShapeFromPyShape(o);
    Py_DECREF(o);
    if (!statusor.ok()) {
      PyErr_SetString(PyExc_RuntimeError, statusor.status().ToString().c_str());
      SWIG_fail;
    }
    temps.push_back(statusor.ConsumeValueOrDie());
  }
  $1 = &temps;
}

%typemap(in) const std::vector<absl::optional<Shape> >& (
    std::vector<absl::optional<Shape> > temps) {
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError, "Argument is not a sequence");
    SWIG_fail;
  }
  const int size = PySequence_Size($input);
  for (int i = 0; i < size; ++i) {
    PyObject* o = PySequence_GetItem($input, i);
    if (o == Py_None) {
      temps.push_back(absl::nullopt);
    } else {
      StatusOr<Shape> statusor = numpy::XlaShapeFromPyShape(o);
      Py_DECREF(o);
      if (!statusor.ok()) {
        PyErr_SetString(PyExc_RuntimeError, statusor.status().ToString().c_str());
        SWIG_fail;
      }
      temps.push_back(statusor.ConsumeValueOrDie());
    }
  }
  $1 = &temps;
}

// PrimitiveType

%typemap(in) PrimitiveType {
  PyObject* py_int = numpy::PyNumberToPyInt($input);
  if (!py_int) {
    PyErr_SetString(PyExc_TypeError, "Argument cannot be converted to int");
    SWIG_fail;
  }
  const long value = numpy::PyIntOrPyLongToLong(py_int);
  if (value == -1 && PyErr_Occurred()) {
    Py_DECREF(py_int);
    SWIG_fail;
  }
  if (!PrimitiveType_IsValid(value)) {
    PyErr_SetString(
        PyExc_TypeError, "Argument not valid for PrimitiveType enum");
    Py_DECREF(py_int);
    SWIG_fail;
  }
  $1 = static_cast<PrimitiveType>(value);
}

// Span<pair<int64, in64>>

%typemap(in) absl::Span<const std::pair<int64, int64> >
    (std::vector<std::pair<int64, int64> > temps) {
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError, "Argument is not a sequence");
    SWIG_fail;
  }
  const int size = PySequence_Size($input);
  temps.reserve(size);
  for (int i = 0; i < size; ++i) {
    PyObject* o = PySequence_GetItem($input, i);
    if (!o) {
      SWIG_fail;
    }
    PyObject* first = PyTuple_GetItem(o, 0);
    if (!first) {
      Py_DECREF(o);
      SWIG_fail;
    }
    PyObject* first_pyint = numpy::PyNumberToPyInt(first);
    if (!first_pyint) {
      PyErr_SetString(
          PyExc_TypeError,
          "First pair item cannot be converted to int");
      Py_DECREF(o);
      SWIG_fail;
    }
    PyObject* second = PyTuple_GetItem(o, 1);
    if (!second) {
      Py_DECREF(o);
      Py_DECREF(first_pyint);
      SWIG_fail;
    }
    PyObject* second_pyint = numpy::PyNumberToPyInt(second);
    if (!second_pyint) {
      PyErr_SetString(
          PyExc_TypeError,
          "Second pair item cannot be converted to int");
      Py_DECREF(o);
      Py_DECREF(first_pyint);
      SWIG_fail;
    }
    const int64 first_value = numpy::PyIntOrPyLongToLong(first_pyint);
    if (first_value == -1 && PyErr_Occurred()) {
      Py_DECREF(o);
      Py_DECREF(first_pyint);
      Py_DECREF(second_pyint);
      SWIG_fail;
    }
    const int64 second_value = numpy::PyIntOrPyLongToLong(second_pyint);
    if (second_value == -1 && PyErr_Occurred()) {
      Py_DECREF(o);
      Py_DECREF(first_pyint);
      Py_DECREF(second_pyint);
      SWIG_fail;
    }
    temps.push_back(std::make_pair(first_value, second_value));
    Py_DECREF(o);
  }
  $1 = temps;
}

// DotDimensionNumbers

%typemap(in) const DotDimensionNumbers&
    (DotDimensionNumbers dimension_numbers) {
  if (!HandleRepeatedInt64Attribute(
        $input, "lhs_contracting_dimensions",
        dimension_numbers.mutable_lhs_contracting_dimensions())) {
    SWIG_fail;
  }
  if (!HandleRepeatedInt64Attribute(
        $input, "rhs_contracting_dimensions",
        dimension_numbers.mutable_rhs_contracting_dimensions())) {
    SWIG_fail;
  }
  if (!HandleRepeatedInt64Attribute(
        $input, "lhs_batch_dimensions",
        dimension_numbers.mutable_lhs_batch_dimensions())) {
    SWIG_fail;
  }
  if (!HandleRepeatedInt64Attribute(
        $input, "rhs_batch_dimensions",
        dimension_numbers.mutable_rhs_batch_dimensions())) {
    SWIG_fail;
  }

  $1 = &dimension_numbers;
}

// PaddingConfig

%typemap(in) const PaddingConfig&
    (PaddingConfig padding_config) {
  PyObject* dimensions = PyObject_GetAttrString($input, "dimensions");
  if (!dimensions) {
    SWIG_fail;
  }

  int length = PySequence_Size(dimensions);
  if (length == -1) {
    Py_DECREF(dimensions);
    SWIG_fail;
  }

  for (int i = 0; i < length; ++i) {
    PyObject* item = PySequence_GetItem(dimensions, i);
    if (!item) {
      Py_DECREF(dimensions);
      SWIG_fail;
    }
    int64 edge_padding_low, edge_padding_high, interior_padding;
    if (!GetIntAttr(item, "edge_padding_low", &edge_padding_low)
        || !GetIntAttr(item, "edge_padding_high", &edge_padding_high)
        || !GetIntAttr(item, "interior_padding", &interior_padding)) {
      Py_DECREF(item);
      Py_DECREF(dimensions);
      SWIG_fail;
    }
    Py_DECREF(item);

    PaddingConfig::PaddingConfigDimension* dimension =
        padding_config.add_dimensions();
    dimension->set_edge_padding_low(edge_padding_low);
    dimension->set_edge_padding_high(edge_padding_high);
    dimension->set_interior_padding(interior_padding);
  }
  Py_DECREF(dimensions);

  $1 = &padding_config;
}

// ConvolutionDimensionNumbers

%typemap(in) const ConvolutionDimensionNumbers&
    (ConvolutionDimensionNumbers dimension_numbers) {
  int64 value;

  if (!GetIntAttr($input, "input_batch_dimension", &value)) {
    SWIG_fail;
  }
  dimension_numbers.set_input_batch_dimension(value);

  if (!GetIntAttr($input, "input_feature_dimension", &value)) {
    SWIG_fail;
  }
  dimension_numbers.set_input_feature_dimension(value);

  if (!GetIntAttr($input, "output_batch_dimension", &value)) {
    SWIG_fail;
  }
  dimension_numbers.set_output_batch_dimension(value);

  if (!GetIntAttr($input, "output_feature_dimension", &value)) {
    SWIG_fail;
  }
  dimension_numbers.set_output_feature_dimension(value);

  if (!GetIntAttr($input, "kernel_output_feature_dimension", &value)) {
    SWIG_fail;
  }
  dimension_numbers.set_kernel_output_feature_dimension(value);

  if (!GetIntAttr($input, "kernel_input_feature_dimension", &value)) {
    SWIG_fail;
  }
  dimension_numbers.set_kernel_input_feature_dimension(value);

  if (!HandleRepeatedInt64Attribute(
        $input, "input_spatial_dimensions",
        dimension_numbers.mutable_input_spatial_dimensions())) {
    SWIG_fail;
  }
  if (!HandleRepeatedInt64Attribute(
        $input, "kernel_spatial_dimensions",
        dimension_numbers.mutable_kernel_spatial_dimensions())) {
    SWIG_fail;
  }
  if (!HandleRepeatedInt64Attribute(
        $input, "output_spatial_dimensions",
        dimension_numbers.mutable_output_spatial_dimensions())) {
    SWIG_fail;
  }

  $1 = &dimension_numbers;
}

// GatherDimensionNumbers

%typemap(in) const GatherDimensionNumbers&
    (GatherDimensionNumbers dimension_numbers) {
  if (!HandleRepeatedInt64Attribute(
        $input, "offset_dims",
        dimension_numbers.mutable_offset_dims())) {
    SWIG_fail;
  }
  if (!HandleRepeatedInt64Attribute(
        $input, "collapsed_slice_dims",
        dimension_numbers.mutable_collapsed_slice_dims())) {
    SWIG_fail;
  }
  if (!HandleRepeatedInt64Attribute(
        $input, "start_index_map",
        dimension_numbers.mutable_start_index_map())) {
    SWIG_fail;
  }

  int64 value;
  if (!GetIntAttr($input, "index_vector_dim", &value)) {
    SWIG_fail;
  }
  dimension_numbers.set_index_vector_dim(value);

  $1 = &dimension_numbers;
}

// ScatterDimensionNumbers

%typemap(in) const ScatterDimensionNumbers&
    (ScatterDimensionNumbers dimension_numbers) {
  if (!HandleRepeatedInt64Attribute(
        $input, "update_window_dims",
        dimension_numbers.mutable_update_window_dims())) {
    SWIG_fail;
  }
  if (!HandleRepeatedInt64Attribute(
        $input, "inserted_window_dims",
        dimension_numbers.mutable_inserted_window_dims())) {
    SWIG_fail;
  }
  if (!HandleRepeatedInt64Attribute(
        $input, "scatter_dims_to_operand_dims",
        dimension_numbers.mutable_scatter_dims_to_operand_dims())) {
    SWIG_fail;
  }

  int64 value;
  if (!GetIntAttr($input, "index_vector_dim", &value)) {
    SWIG_fail;
  }
  dimension_numbers.set_index_vector_dim(value);

  $1 = &dimension_numbers;
}

// Span<const ReplicaGroup>

%typemap(in) absl::Span<const ReplicaGroup >
    (std::vector<ReplicaGroup > temps) {
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError, "Argument is not a sequence");
    SWIG_fail;
  }
  const int size = PySequence_Size($input);
  temps.reserve(size);
  for (int i = 0; i < size; ++i) {
    PyObject* o = PySequence_GetItem($input, i);
    ReplicaGroup rgrp;
    if (!HandleRepeatedInt64Attribute(
            o, "replica_ids",
            rgrp.mutable_replica_ids())) {
        SWIG_fail;
    }
    temps.push_back(rgrp);
    Py_DECREF(o);
  }
  $1 = temps;
}


// ExecutableBuildOptions

%typemap(in) const ExecutableBuildOptions*
    (ExecutableBuildOptions build_options) {
  if ($input == Py_None) {
    $1 = NULL;
  } else {
    if (!HandleStringAttribute($input, "generate_hlo_graph", [&](string s) {
      build_options.mutable_debug_options()->set_xla_generate_hlo_graph(std::move(s));
    })) {
      return nullptr;
    }
    if (!HandleStringAttribute($input, "dump_optimized_hlo_proto_to", [&](string s) {
      build_options.mutable_debug_options()->set_xla_dump_optimized_hlo_proto_to(std::move(s));
    })) {
      return nullptr;
    }
    if (!HandleStringAttribute($input, "dump_unoptimized_hlo_proto_to", [&](string s) {
      build_options.mutable_debug_options()->set_xla_dump_unoptimized_hlo_proto_to(std::move(s));
    })) {
      return nullptr;
    }
    if (!HandleStringAttribute($input, "dump_per_pass_hlo_proto_to", [&](string s) {
      build_options.mutable_debug_options()->set_xla_dump_per_pass_hlo_proto_to(std::move(s));
    })) {
      return nullptr;
    }

    PyObject* o = PyObject_GetAttrString($input, "hlo_profile");
    if (o == NULL) {
      SWIG_fail;
    }
    if (o != Py_None) {
      if (!PyBool_Check(o)) {
        PyErr_SetString(PyExc_TypeError, "ExecutableBuildOptions.hlo_profile must be a bool or None.");
        SWIG_fail;
      }
      build_options.mutable_debug_options()->set_xla_hlo_profile(o == Py_True);
    }
    Py_DECREF(o);

    o = PyObject_GetAttrString($input, "result_shape");
    if (o == nullptr) {
      return nullptr;
    }
    if (o != Py_None) {
      StatusOr<Shape> statusor = numpy::XlaShapeFromPyShape(o);
      if (!statusor.ok()) {
        PyErr_SetString(PyExc_TypeError, absl::StrCat("ExecutableBuildOptions.result_shape could not be created from Python shape value: ", statusor.status().ToString()).c_str());
        Py_DECREF(o);
        SWIG_fail;
      }
      build_options.set_result_layout(statusor.ValueOrDie());
    }
    Py_DECREF(o);

    int64 num_replicas;
    if (!GetIntAttr($input, "num_replicas", &num_replicas)) {
      SWIG_fail;
    }
    build_options.set_num_replicas(num_replicas);

    $1 = &build_options;
  }
}

%ignoreall
%unignore xla;
%unignore xla::swig;
%unignore xla::swig::RegisterCpuCustomCallTarget;
%unignore xla::swig::LocalClient;
%unignore xla::swig::LocalClient::Get;
%unignore xla::swig::LocalClient::DeviceCount;
%unignore xla::swig::LocalClient::TransferToInfeed;
%unignore xla::swig::LocalClient::TransferFromOutfeed;
%unignore xla::swig::LocalShapedBuffer;
%unignore xla::swig::LocalShapedBuffer::FromLiteral;
%unignore xla::swig::LocalShapedBuffer::ToLiteral;
%unignore xla::swig::LocalShapedBuffer::shape;
%unignore xla::swig::LocalShapedBuffer::DestructureTuple;
%unignore xla::swig::LocalShapedBufferTuple;
%unignore xla::swig::LocalShapedBufferTuple::Release;
%unignore xla::swig::LocalShapedBufferTuple::size;
%unignore xla::swig::XrtAllocation;
%unignore xla::swig::XrtAllocation::FromLiteral;
%unignore xla::swig::XrtAllocation::ToLiteral;
%unignore xla::swig::XrtAllocation::shape;
%unignore xla::swig::XrtAllocationTuple;
%unignore xla::swig::XrtAllocationTuple::Release;
%unignore xla::swig::XrtAllocationTuple::size;
%unignore xla::swig::LocalExecutable;
%unignore xla::swig::LocalExecutable::DeviceOrdinals;
%unignore xla::swig::LocalExecutable::Execute;
%unignore xla::swig::LocalExecutable::ExecutePerReplica;
%unignore xla::swig::XrtExecutable;
%unignore xla::swig::XrtExecutable::DeviceOrdinals;
%unignore xla::swig::XrtExecutable::Execute;
%unignore xla::swig::Computation;
%unignore xla::swig::Computation::Compile;
%unignore xla::swig::Computation::CompileForXrt;
%unignore xla::swig::Computation::GetProgramShape;
%unignore xla::swig::Computation::GetReturnValueShape;
%unignore xla::swig::Computation::GetSerializedProto;
%unignore xla::swig::LocalOp;
%unignore xla::swig::ComputationBuilder;
%unignore xla::swig::ComputationBuilder::ComputationBuilder;
%unignore xla::swig::ComputationBuilder::Build;
%unignore xla::swig::ComputationBuilder::BuildWithRoot;
%unignore xla::swig::ComputationBuilder::SetOpMetadata;
%unignore xla::swig::ComputationBuilder::ClearOpMetadata;
%unignore xla::swig::ComputationBuilder::Parameter;
%unignore xla::swig::ComputationBuilder::GetShape;
%unignore xla::swig::ComputationBuilder::GetReturnValueShape;
%unignore xla::swig::ComputationBuilder::Infeed;
%unignore xla::swig::ComputationBuilder::Outfeed;
%unignore xla::swig::ComputationBuilder::ConstantLiteral;
%unignore xla::swig::ComputationBuilder::ConstantR0;
%unignore xla::swig::ComputationBuilder::Iota;
%unignore xla::swig::ComputationBuilder::BroadcastedIota;
%unignore xla::swig::ComputationBuilder::Broadcast;
%unignore xla::swig::ComputationBuilder::BroadcastInDim;
%unignore xla::swig::ComputationBuilder::Pad;
%unignore xla::swig::ComputationBuilder::Reshape;
%unignore xla::swig::ComputationBuilder::Collapse;
%unignore xla::swig::ComputationBuilder::AllToAll;
%unignore xla::swig::ComputationBuilder::CrossReplicaSum;
%unignore xla::swig::ComputationBuilder::Slice;
%unignore xla::swig::ComputationBuilder::SliceInDim;
%unignore xla::swig::ComputationBuilder::DynamicSlice;
%unignore xla::swig::ComputationBuilder::DynamicUpdateSlice;
%unignore xla::swig::ComputationBuilder::ConcatInDim;
%unignore xla::swig::ComputationBuilder::SelectAndScatterWithGeneralPadding;
%unignore xla::swig::ComputationBuilder::Select;
%unignore xla::swig::ComputationBuilder::Tuple;
%unignore xla::swig::ComputationBuilder::GetTupleElement;
%unignore xla::swig::ComputationBuilder::ConvertElementType;
%unignore xla::swig::ComputationBuilder::BitcastConvertType;
%unignore xla::swig::ComputationBuilder::Call;
%unignore xla::swig::ComputationBuilder::Transpose;
%unignore xla::swig::ComputationBuilder::Rev;
%unignore xla::swig::ComputationBuilder::Clamp;
%unignore xla::swig::ComputationBuilder::Map;
%unignore xla::swig::ComputationBuilder::Reduce;
%unignore xla::swig::ComputationBuilder::ReduceWindowWithGeneralPadding;
%unignore xla::swig::ComputationBuilder::RngNormal;
%unignore xla::swig::ComputationBuilder::RngUniform;
%unignore xla::swig::ComputationBuilder::RngBernoulli;
%unignore xla::swig::ComputationBuilder::While;
%unignore xla::swig::ComputationBuilder::Conditional;
%unignore xla::swig::ComputationBuilder::IsConstant;
%unignore xla::swig::ComputationBuilder::Eq;
%unignore xla::swig::ComputationBuilder::Ne;
%unignore xla::swig::ComputationBuilder::Ge;
%unignore xla::swig::ComputationBuilder::Gt;
%unignore xla::swig::ComputationBuilder::Lt;
%unignore xla::swig::ComputationBuilder::Le;
%unignore xla::swig::ComputationBuilder::Dot;
%unignore xla::swig::ComputationBuilder::DotGeneral;
%unignore xla::swig::ComputationBuilder::ConvGeneralDilated;
%unignore xla::swig::ComputationBuilder::Add;
%unignore xla::swig::ComputationBuilder::Sub;
%unignore xla::swig::ComputationBuilder::Mul;
%unignore xla::swig::ComputationBuilder::Div;
%unignore xla::swig::ComputationBuilder::Rem;
%unignore xla::swig::ComputationBuilder::Max;
%unignore xla::swig::ComputationBuilder::Min;
%unignore xla::swig::ComputationBuilder::And;
%unignore xla::swig::ComputationBuilder::Or;
%unignore xla::swig::ComputationBuilder::Xor;
%unignore xla::swig::ComputationBuilder::ShiftLeft;
%unignore xla::swig::ComputationBuilder::ShiftRightArithmetic;
%unignore xla::swig::ComputationBuilder::ShiftRightLogical;
%unignore xla::swig::ComputationBuilder::Not;
%unignore xla::swig::ComputationBuilder::Abs;
%unignore xla::swig::ComputationBuilder::Exp;
%unignore xla::swig::ComputationBuilder::Expm1;
%unignore xla::swig::ComputationBuilder::Floor;
%unignore xla::swig::ComputationBuilder::Ceil;
%unignore xla::swig::ComputationBuilder::Round;
%unignore xla::swig::ComputationBuilder::Log;
%unignore xla::swig::ComputationBuilder::Log1p;
%unignore xla::swig::ComputationBuilder::Sign;
%unignore xla::swig::ComputationBuilder::Cos;
%unignore xla::swig::ComputationBuilder::Sin;
%unignore xla::swig::ComputationBuilder::Tanh;
%unignore xla::swig::ComputationBuilder::Atan2;
%unignore xla::swig::ComputationBuilder::IsFinite;
%unignore xla::swig::ComputationBuilder::Pow;
%unignore xla::swig::ComputationBuilder::Neg;
%unignore xla::swig::ComputationBuilder::Sort;
%unignore xla::swig::ComputationBuilder::SortKeyVal;
%unignore xla::swig::ComputationBuilder::Sqrt;
%unignore xla::swig::ComputationBuilder::Rsqrt;
%unignore xla::swig::ComputationBuilder::Square;
%unignore xla::swig::ComputationBuilder::Reciprocal;
%unignore xla::swig::ComputationBuilder::Erfc;
%unignore xla::swig::ComputationBuilder::Erf;
%unignore xla::swig::ComputationBuilder::ErfInv;
%unignore xla::swig::ComputationBuilder::Lgamma;
%unignore xla::swig::ComputationBuilder::Digamma;
%unignore xla::swig::ComputationBuilder::Acos;
%unignore xla::swig::ComputationBuilder::Asin;
%unignore xla::swig::ComputationBuilder::Atan;
%unignore xla::swig::ComputationBuilder::Tan;
%unignore xla::swig::ComputationBuilder::Acosh;
%unignore xla::swig::ComputationBuilder::Asinh;
%unignore xla::swig::ComputationBuilder::Atanh;
%unignore xla::swig::ComputationBuilder::Cosh;
%unignore xla::swig::ComputationBuilder::Sinh;
%unignore xla::swig::ComputationBuilder::Real;
%unignore xla::swig::ComputationBuilder::Imag;
%unignore xla::swig::ComputationBuilder::Conj;
%unignore xla::swig::ComputationBuilder::Complex;
%unignore xla::swig::ComputationBuilder::Cholesky;
%unignore xla::swig::ComputationBuilder::QR;
%unignore xla::swig::ComputationBuilder::TriangularSolve;
%unignore xla::swig::ComputationBuilder::CustomCall;
%unignore xla::swig::ComputationBuilder::Gather;
%unignore xla::swig::ComputationBuilder::Scatter;
%unignore xla::swig::DeleteComputation;
%unignore xla::swig::DestructureXrtAllocationTuple;
%unignore xla::swig::DeleteLocalShapedBuffer;
%unignore xla::swig::DeleteXrtAllocation;
%unignore xla::swig::DeleteLocalExecutable;
%unignore xla::swig::DeleteXrtExecutable;

%thread;
%include "tensorflow/compiler/xla/python/local_computation_builder.h"
%nothread;

%unignoreall
