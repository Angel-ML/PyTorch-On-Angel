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
// Created by leleyu on 2019-05-05.
//

#include <angel/pytorch/angel_torch.h>
#include <angel/pytorch/model.h>
#include <angel/pytorch/utils.h>

#include <angel/commons.h>
#include <angel/timer.h>
#include <angel/map.h>

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    name
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_tencent_angel_pytorch_Torch_name
  (JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  return env->NewStringUTF(ptr->get_name().data());
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    initPtr
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_com_tencent_angel_pytorch_Torch_initPtr
  (JNIEnv *env, jclass jcls, jstring jpath) {
  DEFINE_STRING(jpath);
  auto *ptr = new angel::TorchModel(jpath_cstr);
  RELEASE_STRING(jpath);
  return reinterpret_cast<int64_t>(ptr);
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    destroyPtr
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_tencent_angel_pytorch_Torch_destroyPtr
  (JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  delete (ptr);
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getType
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_tencent_angel_pytorch_Torch_getType
  (JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  return env->NewStringUTF(ptr->get_type_string().data());
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getMatsSize
 * Signature: (J)[I
 */
JNIEXPORT jintArray JNICALL Java_com_tencent_angel_pytorch_Torch_getMatsSize

  (JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  auto output = ptr->get_mats_size();
  auto output_ptr = output.data();
  DEFINE_JINTARRAY(output_ptr, static_cast<jsize>(output.size()));
  return output_ptr_jarray;
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getInputDim
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_tencent_angel_pytorch_Torch_getInputDim
  (JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  return static_cast<jint>(ptr->get_input_dim());
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getNumFields
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_tencent_angel_pytorch_Torch_getNumFields
  (JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  return static_cast<jint>(ptr->get_num_fields());
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getEmbeddingDim
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_tencent_angel_pytorch_Torch_getEmbeddingDim
  (JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  return static_cast<jint>(ptr->get_embedding_dim());
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getParametersTotalSize
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_tencent_angel_pytorch_Torch_getParametersTotalSize
  (JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  return ptr->get_parameters_total_size();
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    setParameters
 * Signature: (J[F)V
 */
JNIEXPORT void JNICALL Java_com_tencent_angel_pytorch_Torch_setParameters
  (JNIEnv *env, jclass jcls, jlong jptr, jfloatArray jvals) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  jboolean is_copy;
  DEFINE_PRIMITIVE_ARRAY(jvals)
  int len = env->GetArrayLength(jvals);
  ptr->set_parameters(jvals_cptr, len);
  RELEASE_PRIMITIVE_ARRAY(jvals);
}

void add_inputs(JNIEnv *env,
                std::vector<torch::jit::IValue> *inputs,
                std::vector<std::pair<std::string, void *>> *ptrs,
                jobject jparams,
                angel::TorchModelType type) {
  using namespace angel;
  int len = env->GetArrayLength((jarray) angel::jni_map_get(env, jparams, "index"));
  switch (type) {
    case angel::TorchModelType::BIAS_WEIGHT:
      add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "bias");
      add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "weights");
      break;
    case angel::TorchModelType::BIAS_WEIGHT_EMBEDDING:
      add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "bias");
      add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "weights");
      add_input(env, inputs, ptrs, jparams, {len, angel::jni_map_get_int(env, jparams, "embedding_dim")},
                TORCH_OPTION_FLOAT_GRAD, "embedding");
      break;
    case angel::TorchModelType::BIAS_WEIGHT_EMBEDDING_MATS:
      add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "bias");
      add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "weights");
      add_input(env, inputs, ptrs, jparams, {len, angel::jni_map_get_int(env, jparams, "embedding_dim")},
                TORCH_OPTION_FLOAT_GRAD, "embedding");
      add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "mats", "mats_sizes");
      break;
    case angel::TorchModelType::BIAS_WEIGHT_EMBEDDING_MATS_FIELD:
      add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "bias");
      add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "weights");
      add_input(env, inputs, ptrs, jparams, {len, angel::jni_map_get_int(env, jparams, "embedding_dim")},
                TORCH_OPTION_FLOAT_GRAD, "embedding");
      add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "mats", "mats_sizes");
      add_input(env, inputs, ptrs, jparams, TORCH_OPTION_INT64, "fields");
      break;
    default:
      break;
  }
}

void add_parameters(JNIEnv *env,
                    std::vector<torch::jit::IValue> *inputs,
                    std::vector<std::pair<std::string, void *>> *ptrs,
                    jobject jparams,
                    angel::TorchModelType type) {
  using namespace angel;
  switch (type) {
    case angel::TorchModelType::BIAS_WEIGHT:
      add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "bias");
      add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "weights");
      break;
    case angel::TorchModelType::BIAS_WEIGHT_EMBEDDING:
      add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "bias");
      add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "weights");
      add_input(env, inputs, ptrs, jparams,
                {angel::jni_map_get_int(env, jparams, "dim"), angel::jni_map_get_int(env, jparams, "embedding_dim")},
                TORCH_OPTION_FLOAT_GRAD, "embedding");
      break;
    case angel::TorchModelType::BIAS_WEIGHT_EMBEDDING_MATS:
    case angel::TorchModelType::BIAS_WEIGHT_EMBEDDING_MATS_FIELD:
      add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "bias");
      add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "weights");
      add_input(env, inputs, ptrs, jparams,
                {angel::jni_map_get_int(env, jparams, "dim"), angel::jni_map_get_int(env, jparams, "embedding_dim")},
                TORCH_OPTION_FLOAT_GRAD, "embedding");
      add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "mats", "mats_sizes");
      break;
    default:
      break;
  }
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    forward
 * Signature: (JLjava/util/Map;)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_tencent_angel_pytorch_Torch_forward
  (JNIEnv *env, jclass jcls, jlong jptr, jobject jparams, jboolean serving) {
  using namespace angel;
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  // build inputs
  std::vector<torch::jit::IValue> inputs;
  std::vector<std::pair<std::string, void *>> ptrs;

  int batch_size = angel::jni_map_get_int(env, jparams, "batch_size");
  // data inputs
  inputs.emplace_back(batch_size);
  ptrs.emplace_back(std::make_pair("batch_size", nullptr)); // why adding a nullptr ?
  add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_INT64, "index");
  add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_INT64, "feats");
  add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_FLOAT, "values");

  if (serving) {
    if (ptr->get_type() == angel::TorchModelType::BIAS_WEIGHT_EMBEDDING_MATS_FIELD) {
      add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_INT64, "fields");
    }
    auto output = ptr->serving_forward(inputs);
    auto output_ptr = output.data_ptr();
    DEFINE_JFLOATARRAY(output_ptr, batch_size);

    // release java arrays
    release_array(env, ptrs, jparams);
    return output_ptr_jarray;
  } else {
    add_inputs(env, &inputs, &ptrs, jparams, ptr->get_type());
    auto output = ptr->forward(inputs).toTensor();
    auto output_ptr = output.data_ptr();
    DEFINE_JFLOATARRAY(output_ptr, batch_size);

    // release java arrays
    release_array(env, ptrs, jparams);
    return output_ptr_jarray;
  }
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    backward
 * Signature: (JLjava/util/Map;)F
 */
JNIEXPORT jfloat JNICALL Java_com_tencent_angel_pytorch_Torch_backward
  (JNIEnv *env, jclass jcls, jlong jptr, jobject jparams) {
  using namespace angel;
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  // build inputs
  std::vector<torch::jit::IValue> inputs;
  std::vector<std::pair<std::string, void *>> ptrs;
  int batch_size = angel::jni_map_get_int(env, jparams, "batch_size");
  // data inputs
  inputs.emplace_back(batch_size);
  ptrs.emplace_back(std::make_pair("batch_size", nullptr));
  add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_INT64, "index");
  add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_INT64, "feats");
  add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_FLOAT, "values");
  // parameters inputs
  add_inputs(env, &inputs, &ptrs, jparams, ptr->get_type());
  // targets
  jarray targets = (jarray) jni_map_get(env, jparams, "targets");
  jboolean is_copy;
  DEFINE_PRIMITIVE_ARRAY(targets);
  DEFINE_JARRAY_TENSOR_DIM_FLOAT(targets);

  auto loss = ptr->backward(inputs, targets_tensor);
  set_grads(env, inputs, ptrs, jparams);
  release_array(env, ptrs, jparams);
  RELEASE_PRIMITIVE_ARRAY(targets);
  return loss;
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    save
 * Signature: (JLjava/lang/String;Ljava/util/Map;)V
 */
JNIEXPORT void JNICALL Java_com_tencent_angel_pytorch_Torch_save
  (JNIEnv *env, jclass jcls, jlong jptr, jobject jparams) {
  using namespace angel;
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  // build parameters
  std::vector<torch::jit::IValue> parameters;
  std::vector<std::pair<std::string, void *>> ptrs;
  // parameters inputs
  add_parameters(env, &parameters, &ptrs, jparams, ptr->get_type());
  ptr->save_module(parameters, ptr->get_type());
  release_array(env, ptrs, jparams);
}

void gcn_add_parameters(JNIEnv *env,
  std::vector<torch::jit::IValue> *inputs,
  std::vector<std::pair<std::string, void *>> *ptrs,
  jobject jparams) {
  using namespace angel;

  // calculate x dimension and add x
  int feature_dim = jni_map_get_int(env, jparams, "feature_dim");
  const std::vector<std::string> x_keys = {"x", "pos_x", "neg_x"};
  for (auto x_key: x_keys)
    if (jni_map_contain(env, jparams, x_key)) {
      jarray x = (jarray) jni_map_get(env, jparams, x_key);
      int x_num = env->GetArrayLength(x) / feature_dim;
      add_input(env, inputs, ptrs, jparams, {x_num, feature_dim}, TORCH_OPTION_FLOAT, x_key);
    }

  // add edge_index
  const std::vector<std::string> edge_keys = {"edge_index", "first_edge_index", "second_edge_index"};
  for (auto edge_key: edge_keys)
    if (jni_map_contain(env, jparams, edge_key)) {
      jarray edge_index = (jarray) jni_map_get(env, jparams, edge_key);
      int edge_num = env->GetArrayLength(edge_index) / 2;
      add_input(env, inputs, ptrs, jparams, {2, edge_num}, TORCH_OPTION_INT64, edge_key);
    }
}

void gcn_set_weights(JNIEnv *env,
  angel::TorchModel *model_ptr,
  std::vector<std::pair<std::string, void *>> *ptrs,
  jobject jparams) {
  using namespace angel;
  jboolean is_copy;
  jarray weights = (jarray) jni_map_get(env, jparams, "weights");
  void* weights_cptr = env->GetPrimitiveArrayCritical(weights, &is_copy);
  ptrs->push_back(std::make_pair("weights", weights_cptr));
  model_ptr->set_parameters(weights_cptr, env->GetArrayLength(weights));
}

void gcn_set_grads(JNIEnv *env,
  angel::TorchModel *model_ptr,
  const std::vector<std::pair<std::string, void*>> &ptrs,
  jobject jparams) {
  using namespace angel;
  jarray weights = (jarray) jni_map_get(env, jparams, "weights");
  for (auto item : ptrs)
    if (item.first == "weights")
      model_ptr->set_grads(item.second, env->GetArrayLength(weights));
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    gcnBackward
 * Signature: (JLjava/util/Map;)F
 */
JNIEXPORT jfloat JNICALL Java_com_tencent_angel_pytorch_Torch_gcnBackward
  (JNIEnv *env, jclass jcls, jlong jptr, jobject jparams) {
  using namespace angel;
  DEFINE_MODEL_PTR(angel::TorchModel, jptr)
  // build inputs
  std::vector<torch::jit::IValue> inputs;
  std::vector<std::pair<std::string, void*>> ptrs;

  gcn_add_parameters(env, &inputs, &ptrs, jparams);
  gcn_set_weights(env, ptr, &ptrs, jparams);

  // get targets if defined
  at::Tensor targets; // undefined targets
  if (jni_map_contain(env, jparams, "targets")) {
    jboolean is_copy;
    jarray jtargets = (jarray) jni_map_get(env, jparams, "targets");
    void* jtargets_cptr = env->GetPrimitiveArrayCritical(jtargets, &is_copy);
    targets = torch::from_blob(jtargets_cptr, {env->GetArrayLength(jtargets)}, TORCH_OPTION_INT64);
    ptrs.push_back(std::make_pair("targets", jtargets_cptr));
  }

  ptr->zero_grad();
  auto loss = ptr->backward(inputs, targets);
  gcn_set_grads(env, ptr, ptrs, jparams);

  release_array(env, ptrs, jparams);
  return loss;
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    gcnExecMethod
 * Signature: (JLjava/lang/String;Ljava/util/Map;)Ljava/lang/Object;
 */
JNIEXPORT jobject JNICALL Java_com_tencent_angel_pytorch_Torch_gcnExecMethod
  (JNIEnv *env, jclass jcls, jlong jptr, jstring jmethod, jobject jparams) {
  using namespace angel;
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  // build inputs
  std::vector<torch::jit::IValue> inputs;
  std::vector<std::pair<std::string, void *>> ptrs;

  gcn_add_parameters(env, &inputs, &ptrs, jparams);
  gcn_set_weights(env, ptr, &ptrs, jparams);

  const char* method = env->GetStringUTFChars(jmethod, 0);
  auto output = ptr->exec_method(method, inputs).toTensor();
  env->ReleaseStringUTFChars(jmethod, method);

  jarray joutput;
  if (output.dtype().id() == caffe2::TypeIdentifier::Get<float>()) {
    joutput = env->NewFloatArray(output.numel());
    env->SetFloatArrayRegion((jfloatArray)joutput, 0, output.numel(), output.data<float>());
  } else if (output.dtype().id() == caffe2::TypeIdentifier::Get<int64_t>()) {
    joutput = env->NewLongArray(output.numel());
    env->SetLongArrayRegion((jlongArray)joutput, 0, output.numel(), output.data<int64_t>());
  } else
    throw std::logic_error("the dtype for predict should be float or int64");

  release_array(env, ptrs, jparams);
  return joutput;
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getParameters
 * Signature: (J)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_tencent_angel_pytorch_Torch_getParameters
  (JNIEnv *env, jclass jcls, jlong jptr) {
  using namespace angel;
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  std::vector<at::Tensor> tensors = ptr->get_parameters();
  int size = 0;
  for (auto const& t: tensors)
    size += t.size(0);

  jfloatArray array = env->NewFloatArray(size);

  int start = 0;
  for (auto const& t: tensors) {
    env->SetFloatArrayRegion(array, start, t.size(0), t.data<float>());
    start += t.size(0);
  }

  return array;
}

