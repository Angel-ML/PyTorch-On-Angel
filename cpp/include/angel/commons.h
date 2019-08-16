/*
 * Tencent is pleased to support the open source community by making Angel available.
 *
 * Copyright (C) 2017-2018 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */
//
// Created by leleyu on 2019-04-08.
//

#ifndef GRAPH_INTERFACE_COMMONS_H
#define GRAPH_INTERFACE_COMMONS_H


// define torch tensors

#define TORCH_OPTION(type, is_grad) \
  (torch::TensorOptions().dtype(type).requires_grad(is_grad))

#define TORCH_OPTION_INT64 \
  TORCH_OPTION(torch::kInt64, false)

#define TORCH_OPTION_INT32 \
  TORCH_OPTION(torch::kInt32, false)

#define TORCH_OPTION_FLOAT \
  TORCH_OPTION(torch::kFloat, false)

#define TORCH_OPTION_FLOAT_GRAD \
  TORCH_OPTION(torch::kFloat, true)

#define DEFINE_ZEROS_DIM2_INT64(_name, dim1, dim2) \
  auto _name = torch::zeros({dim1, dim2}, TORCH_OPTION_INT64);

#define DEFINE_ZEROS_DIM2_FLOAT(_name, dim1, dim2) \
  auto _name = torch::zeros({dim1, dim2}, TORCH_OPTION_FLOAT);

#define DEFINE_ZEROS_DIM1_INT64(_name, dim1) \
  auto _name = torch::zeros({dim1}, TORCH_OPTION_INT64);

#define DEFINE_ZEROS_DIM1_INT32(_name, dim1) \
  auto _name = torch::zeros({dim1}, TORCH_OPTION_INT32);

#define DEFINE_ACCESSOR_DIM1_INT64(_tensor) \
  auto _tensor##_acr = _tensor.accessor<int64_t, 1>();

#define DEFINE_ACCESSOR_DIM1_INT32(_tensor) \
  auto _tensor##_acr = _tensor.accessor<int32_t, 1>();

#define DEFINE_ACCESSOR_DIM2_INT64(_tensor) \
  auto _tensor##_acr = _tensor.accessor<int64_t, 2>();

#define DEFINE_ACCESSOR_DIM2_FLOAT(_tensor) \
  auto _tensor##_acr = _tensor.accessor<float, 2>();

// arrays -> tensors

#define __DEFINE_ARRAY_TENSOR_DIM1(array, dim, type, is_grad) \
  auto array##_tensor = torch::from_blob(array, {dim}, TORCH_OPTION(type, is_grad));

#define __DEFINE_ARRAY_TENSOR_DIM2(array, dim1, dim2, type, is_grad) \
  auto array##_tensor = torch::from_blob(array, {dim1, dim2}, TORCH_OPTION(type, is_grad));

#define DEFINE_ARRAY_TENSOR_DIM2_FLOAT_GRAD(array, dim1, dim2)\
  __DEFINE_ARRAY_TENSOR_DIM2(array, dim1, dim2, torch::kFloat, true)



// jarrays -> tensors
#define __DEFINE_JARRAY_TENSOR_DIM1(_jarray, dim, type, is_grad) \
  auto _jarray##_tensor = torch::from_blob(_jarray##_cptr, {dim}, TORCH_OPTION(type, is_grad));

#define __DEFINE_JARRAY_TENSOR_DIM2(_jarray, dim1, dim2, type, is_grad) \
  auto _jarray##_tensor = torch::from_blob(_jarray##_cptr, {dim1, dim2}, TORCH_OPTION(type, is_grad));

#define DEFINE_JARRAY_TENSOR_DIM_INT64(_jarray) \
  __DEFINE_JARRAY_TENSOR_DIM1(_jarray, env->GetArrayLength(_jarray), torch::kInt64, false)

#define DEFINE_JARRAY_TENSOR_DIM_FLOAT(_jarray) \
  __DEFINE_JARRAY_TENSOR_DIM1(_jarray, env->GetArrayLength(_jarray), torch::kFloat, false)

#define DEFINE_JARRAY_TENSOR_DIM_FLOAT_GRAD(_jarray) \
  __DEFINE_JARRAY_TENSOR_DIM1(_jarray, env->GetArrayLength(_jarray), torch::kFloat, true)

#define DEFINE_JARRAY_TENSOR_DIM2_FLOAT(_jarray, dim1, dim2) \
  __DEFINE_JARRAY_TENSOR_DIM2(_jarray, dim1, dim2, torch::kFloat, false)

#define DEFINE_JARRAY_TENSOR_DIM2_FLOAT_GRAD(_jarray, dim1, dim2) \
  __DEFINE_JARRAY_TENSOR_DIM2(_jarray, dim1, dim2, torch::kFloat, true)


#define DEFINE_SCALA_TENSOR_DIM_INT64(scala) \
  auto scala##_tensor = torch::from_blob(&scala, {1}, TORCH_OPTION_INT64);

#define DEFINE_SCALA_TENSOR_DIM_INT32(scala) \
  auto scala##_tensor = torch::from_blob(&scala, {1}, TORCH_OPTION_INT32);

//#define DEFINE_TORCH_TENSOR_ARRAY(_jarray, type) \
//  auto _jarray##_option = torch::TensorOptions().dtype(type).requires_grad(false); \
//  auto _jarray##_tensor = torch::from_blob(_jarray##_cptr, {env->GetArrayLength(_jarray)}, _jarray##_option);
//
//#define DEFINE_TORCH_TENSOR_ARRAY_GRAD(_jarray, type) \
//  auto _jarray##_option = torch::TensorOptions().dtype(type).requires_grad(true);\
//  auto _jarray##_tensor = torch::from_blob(_jarray##_cptr, {env->GetArrayLength(_jarray)}, _jarray##_option);
//
//#define DEFINE_TORCH_TENSOR_SCALA(_jprimitive, type) \
//  auto _jprimitive##_option = torch::TensorOptions().dtype(type).requires_grad(false);\
//  auto _jprimitive##_tensor = torch::from_blob(&_jprimitive, {1}, _jprimitive##_option);

#define DEFINE_JFLOATARRAY(_carray_ptr, len) \
  jfloatArray _carray_ptr##_jarray = env->NewFloatArray(len); \
  env->SetFloatArrayRegion(_carray_ptr##_jarray, 0, len, reinterpret_cast<float*>(_carray_ptr));

#define DEFINE_JINTARRAY(_carray_ptr, len) \
  jintArray _carray_ptr##_jarray = env->NewIntArray(len); \
  env->SetIntArrayRegion(_carray_ptr##_jarray, 0, len, reinterpret_cast<int*>(_carray_ptr));


// convert primitives between java object and c++ object

// strings
#define DEFINE_STRING(_jstring) \
  const char* jstring##_cstr_ptr = env->GetStringUTFChars(_jstring, NULL); \
  std::string _jstring##_cstr(jstring##_cstr_ptr);

#define RELEASE_STRING(_jstring) \
  env->ReleaseStringUTFChars(_jstring, jstring##_cstr_ptr);

// jarrays
#define DEFINE_PRIMITIVE_ARRAY(_jarray) \
  void* _jarray##_cptr = env->GetPrimitiveArrayCritical(_jarray, &is_copy);

#define RELEASE_PRIMITIVE_ARRAY(_jarray) \
  env->ReleasePrimitiveArrayCritical(_jarray, _jarray##_cptr, 0);

#define DEFINE_MODEL_PTR(MODEL_TYPE, jptr) \
  auto* ptr = reinterpret_cast<MODEL_TYPE*>(jptr);

// extract pointers from java arrays
#define DEFINE_PRIMITIVE_ARRAYS3(jarray1, jarray2, jarray3) \
  DEFINE_PRIMITIVE_ARRAY(jarray1)\
  DEFINE_PRIMITIVE_ARRAY(jarray2)\
  DEFINE_PRIMITIVE_ARRAY(jarray3)

#define DEFINE_PRIMITIVE_ARRAYS4(jarray1, jarray2, jarray3, jarray4) \
  DEFINE_PRIMITIVE_ARRAY(jarray1);\
  DEFINE_PRIMITIVE_ARRAY(jarray2);\
  DEFINE_PRIMITIVE_ARRAY(jarray3);\
  DEFINE_PRIMITIVE_ARRAY(jarray4);

#define DEFINE_PRIMITIVE_ARRAYS5(jarray1, jarray2, jarray3, jarray4, jarray5) \
  DEFINE_PRIMITIVE_ARRAYS4(jarray1, jarray2, jarray3, jarray4); \
  DEFINE_PRIMITIVE_ARRAY(jarray5);

#define DEFINE_PRIMITIVE_ARRAYS6(jarray1, jarray2, jarray3, jarray4, jarray5, jarray6) \
  DEFINE_PRIMITIVE_ARRAYS5(jarray1, jarray2, jarray3, jarray4, jarray5); \
  DEFINE_PRIMITIVE_ARRAY(jarray6);

#define RELEASE_PRIMITIVE_ARRAYS3(jarray1, jarray2, jarray3) \
  RELEASE_PRIMITIVE_ARRAY(jarray1)\
  RELEASE_PRIMITIVE_ARRAY(jarray2)\
  RELEASE_PRIMITIVE_ARRAY(jarray3)

#define RELEASE_PRIMITIVE_ARRAYS4(jarray1, jarray2, jarray3, jarray4) \
  RELEASE_PRIMITIVE_ARRAY(jarray1);\
  RELEASE_PRIMITIVE_ARRAY(jarray2);\
  RELEASE_PRIMITIVE_ARRAY(jarray3);\
  RELEASE_PRIMITIVE_ARRAY(jarray4);

#define RELEASE_PRIMITIVE_ARRAYS5(jarray1, jarray2, jarray3, jarray4, jarray5) \
  RELEASE_PRIMITIVE_ARRAYS4(jarray1, jarray2, jarray3, jarray4) \
  RELEASE_PRIMITIVE_ARRAY(jarray5);

#define RELEASE_PRIMITIVE_ARRAYS6(jarray1, jarray2, jarray3, jarray4, jarray5, jarray6) \
  RELEASE_PRIMITIVE_ARRAYS5(jarray1, jarray2, jarray3, jarray4, jarray5);\
  RELEASE_PRIMITIVE_ARRAY(jarray6);

#define DEFINE_PRIMITIVE_OBJECT_ARRAY(jobjectarray) \
  std::vector<void*> jobjectarray##_cptrs;\
  jobjectarray##_cptrs.resize(env->GetArrayLength(jobjectarray));\
  for (int i = 0; i < env->GetArrayLength(jobjectarray); i++)  \
    jobjectarray##_cptrs[i] = env->GetPrimitiveArrayCritical((jarray)env->GetObjectArrayElement(jobjectarray, i), &is_copy);

#define RELEASE_PRIMITIVE_OBJECT_ARRAY(jobjectarray) \
  for (int i = 0; i < env->GetArrayLength(jobjectarray); i++) \
    env->ReleasePrimitiveArrayCritical((jarray)env->GetObjectArrayElement(jobjectarray, i), jobjectarray##_cptrs[i], 0);


// copy data from tensor.grad to jarray
#define __SET_FLOAT_GRAD(jarray, start, length) \
  env->SetFloatArrayRegion(jarray, start, length, jarray##_tensor.grad().data<float>());

#define SET_FLOAT_GRAD(jarray) \
  __SET_FLOAT_GRAD(jarray, 0, env->GetArrayLength(jarray));



#endif //GRAPH_INTERFACE_COMMONS_H
