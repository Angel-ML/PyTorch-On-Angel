/*
 * Tencent is pleased to support the open source community by making Angel
 * available.
 *
 * Copyright (C) 2017-2018 THL A29 Limited, a Tencent company. All rights
 * reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
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
#include <angel/map.h>
#include <angel/timer.h>

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    name
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_tencent_angel_pytorch_Torch_name(
    JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  return env->NewStringUTF(ptr->get_name().data());
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    initPtr
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_com_tencent_angel_pytorch_Torch_initPtr(
    JNIEnv *env, jclass jcls, jstring jpath) {
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
JNIEXPORT void JNICALL Java_com_tencent_angel_pytorch_Torch_destroyPtr(
    JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  delete (ptr);
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getType
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_tencent_angel_pytorch_Torch_getType(
    JNIEnv *env, jclass jcls, jlong jptr) {
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
JNIEXPORT jint JNICALL Java_com_tencent_angel_pytorch_Torch_getInputDim(
    JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  return static_cast<jint>(ptr->get_input_dim());
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getUserInputDim
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_tencent_angel_pytorch_Torch_getUserInputDim(
    JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  return static_cast<jint>(ptr->get_user_input_dim());
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getItemInputDim
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_tencent_angel_pytorch_Torch_getItemInputDim(
    JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  return static_cast<jint>(ptr->get_item_input_dim());
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getNumFields
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_tencent_angel_pytorch_Torch_getNumFields(
    JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  return static_cast<jint>(ptr->get_num_fields());
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getUserNumFields
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_tencent_angel_pytorch_Torch_getUserNumFields(
    JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  return static_cast<jint>(ptr->get_user_num_fields());
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getItemNumFields
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_tencent_angel_pytorch_Torch_getItemNumFields(
    JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  return static_cast<jint>(ptr->get_item_num_fields());
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getEmbeddingDim
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_tencent_angel_pytorch_Torch_getEmbeddingDim(
    JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  return static_cast<jint>(ptr->get_embedding_dim());
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getUserEmbeddingDim
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_tencent_angel_pytorch_Torch_getUserEmbeddingDim(
    JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  return static_cast<jint>(ptr->get_user_embedding_dim());
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getItemEmbeddingDim
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_tencent_angel_pytorch_Torch_getItemEmbeddingDim(
    JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  return static_cast<jint>(ptr->get_item_embedding_dim());
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getNodeTypes
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_tencent_angel_pytorch_Torch_getNodeTypes
  (JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  return env->NewStringUTF(ptr->get_node_types().data());
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getEdgeTypes
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_tencent_angel_pytorch_Torch_getEdgeTypes
  (JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  return env->NewStringUTF(ptr->get_edge_types().data());
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getSchema
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_tencent_angel_pytorch_Torch_getSchema
  (JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  return env->NewStringUTF(ptr->get_schame().data());
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getEmbeddingsSize
 * Signature: (J)[I
 */
JNIEXPORT jintArray JNICALL
Java_com_tencent_angel_pytorch_Torch_getEmbeddingsSize(JNIEnv *env, jclass jcls,
                                                       jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  auto output = ptr->get_embeddings_size();
  auto output_ptr = output.data();
  DEFINE_JINTARRAY(output_ptr, static_cast<jsize>(output.size()));
  return output_ptr_jarray;
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getInputSizes
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_com_tencent_angel_pytorch_Torch_getInputSizes(
    JNIEnv *env, jclass jcls, jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  auto output = ptr->get_inputs_size();
  auto output_ptr = output.data();
  DEFINE_JLONGARRAY(output_ptr, static_cast<jsize>(output.size()));
  return output_ptr_jarray;
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getDenseColNums
 * Signature: (J)[I
 */
JNIEXPORT jintArray JNICALL
Java_com_tencent_angel_pytorch_Torch_getDenseColNums(JNIEnv *env, jclass jcls,
                                                     jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  auto output = ptr->get_dense_col_nums();
  auto output_ptr = output.data();
  DEFINE_JINTARRAY(output_ptr, static_cast<jsize>(output.size()));
  return output_ptr_jarray;
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getSparseColNums
 * Signature: (J)[I
 */
JNIEXPORT jintArray JNICALL
Java_com_tencent_angel_pytorch_Torch_getSparseColNums(JNIEnv *env, jclass jcls,
                                                      jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  auto output = ptr->get_sparse_col_nums();
  auto output_ptr = output.data();
  DEFINE_JINTARRAY(output_ptr, static_cast<jsize>(output.size()));
  return output_ptr_jarray;
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getParametersTotalSize
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL
Java_com_tencent_angel_pytorch_Torch_getParametersTotalSize(JNIEnv *env,
                                                            jclass jcls,
                                                            jlong jptr) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  return ptr->get_parameters_total_size();
}

void add_inputs(JNIEnv *env, std::vector<torch::jit::IValue> *inputs,
                std::vector<std::pair<std::string, void *>> *ptrs,
                jobject jparams, angel::TorchModelType type) {
  using namespace angel;
  int len =
      env->GetArrayLength((jarray)angel::jni_map_get(env, jparams, "index"));
  size_t start_pos = inputs->size();
  switch (type) {
  case angel::TorchModelType::EMBEDDINGS_MATS_FIELD:
    add_input(env, inputs, ptrs, jparams, TORCH_OPTION_INT64, "fields");
    add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "mats",
              "mats_sizes");
    add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD,
              "embeddings_array", "embeddings_sizes", "inputs_sizes");
    break;
  case angel::TorchModelType::EMBEDDINGS_MATS:
    add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "mats",
              "mats_sizes");
    add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD,
              "embeddings_array", "embeddings_sizes", len);
    break;
  case angel::TorchModelType::BIAS_WEIGHT_EMBEDDING_MATS_FIELD:
    add_input(env, inputs, ptrs, jparams, TORCH_OPTION_INT64, "fields");
  case angel::TorchModelType::BIAS_WEIGHT_EMBEDDING_MATS:
    add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "mats",
              "mats_sizes");
  case angel::TorchModelType::BIAS_WEIGHT_EMBEDDING:
    add_input(env, inputs, ptrs, jparams,
              {len, angel::jni_map_get_int(env, jparams, "embedding_dim")},
              TORCH_OPTION_FLOAT_GRAD, "embedding");
  case angel::TorchModelType::BIAS_WEIGHT:
    add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "weights");
    add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "bias");
  default:
    break;
  }
  size_t end_pos = inputs->size();

  // reverse inputs from [start_pos, end_pos)
  // reverse ptrs at the same time between we need to set gradients later
  size_t n = end_pos - start_pos;
  for (size_t i = 0; i < n / 2; i++) {
    auto temp1 = (*inputs)[i + start_pos];
    (*inputs)[i + start_pos] = (*inputs)[n - i - 1 + start_pos];
    (*inputs)[n - i - 1 + start_pos] = temp1;
    auto temp2 = (*ptrs)[i + start_pos];
    (*ptrs)[i + start_pos] = (*ptrs)[n - i - 1 + start_pos];
    (*ptrs)[n - i - 1 + start_pos] = temp2;
  }
}

void add_word2vec_inputs(JNIEnv *env, std::vector<torch::jit::IValue> *inputs,
                         std::vector<std::pair<std::string, void *>> *ptrs,
                         jobject jparams) {
  using namespace angel;
  add_input(env, inputs, ptrs, jparams,
            {jni_map_get_int(env, jparams, "batch_size"),
             jni_map_get_int(env, jparams, "embedding_dim")},
            TORCH_OPTION_FLOAT_GRAD, "srcEmbeddings");
  add_input(env, inputs, ptrs, jparams,
            {jni_map_get_int(env, jparams, "batch_size"),
             jni_map_get_int(env, jparams, "embedding_dim")},
            TORCH_OPTION_FLOAT_GRAD, "dstEmbeddings");
  add_input(env, inputs, ptrs, jparams,
            {jni_map_get_int(env, jparams, "neg_size"),
             jni_map_get_int(env, jparams, "embedding_dim")},
            TORCH_OPTION_FLOAT_GRAD, "negativeEmbeddings");
}

std::vector<std::pair<std::string, int>>
add_parameters(JNIEnv *env, std::vector<torch::jit::IValue> *inputs,
               std::vector<std::pair<std::string, void *>> *ptrs,
               jobject jparams, angel::TorchModelType type) {
  using namespace angel;

  std::vector<std::pair<std::string, int>> input_index;
  switch (type) {
  case angel::TorchModelType ::EMBEDDINGS_MATS:
    input_index.push_back(std::make_pair("mats", inputs->size()));
    add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "mats",
              "mats_sizes");
    input_index.push_back(std::make_pair("embeddings", inputs->size()));
    add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD,
              "embeddings_array", "embeddings_sizes", "inputs_sizes");
    break;
  case angel::TorchModelType::BIAS_WEIGHT_EMBEDDING_MATS:
  case angel::TorchModelType::BIAS_WEIGHT_EMBEDDING_MATS_FIELD:
    input_index.push_back(std::make_pair("mats", inputs->size()));
    add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "mats",
              "mats_sizes");
  case angel::TorchModelType::BIAS_WEIGHT_EMBEDDING:
    input_index.push_back(std::make_pair("embedding", inputs->size()));
    add_input(env, inputs, ptrs, jparams,
              {angel::jni_map_get_int(env, jparams, "dim"),
               angel::jni_map_get_int(env, jparams, "embedding_dim")},
              TORCH_OPTION_FLOAT_GRAD, "embedding");
  case angel::TorchModelType::BIAS_WEIGHT:
    input_index.push_back(std::make_pair("weights", inputs->size()));
    add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "weights");
    input_index.push_back(std::make_pair("bias", inputs->size()));
    add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "bias");
  default:
    break;
  }

  return input_index;
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    forward
 * Signature: (JLjava/util/Map;)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_tencent_angel_pytorch_Torch_forward(
    JNIEnv *env, jclass jcls, jlong jptr, jobject jparams, jboolean serving) {
  using namespace angel;
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  // build inputs
  std::vector<torch::jit::IValue> inputs;
  std::vector<std::pair<std::string, void *>> ptrs;

  int multi_forward_out = 1;
  if (angel::jni_map_contain(env, jparams, "multi_forward_out")) {
    multi_forward_out =
        angel::jni_map_get_int(env, jparams, "multi_forward_out");
  }

  int batch_size = angel::jni_map_get_int(env, jparams, "batch_size");
  int training = 0;
  // data inputs
  inputs.emplace_back(batch_size);
  inputs.emplace_back(training); // 0 represents predicting
  ptrs.emplace_back(
      std::make_pair("batch_size", nullptr)); // why adding a nullptr ?
  ptrs.emplace_back(std::make_pair("training", nullptr));
  add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_INT64, "index");
  add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_INT64, "feats");
  add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_FLOAT, "values");

  // targets, forward has no targets, add values instead to adjust the
  // "forward_" function in pt
  add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_FLOAT, "values");

  if (serving) {
    if (ptr->get_type() ==
        angel::TorchModelType::BIAS_WEIGHT_EMBEDDING_MATS_FIELD) {
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
    DEFINE_JFLOATARRAY(output_ptr, batch_size * multi_forward_out);

    // release java arrays
    release_array(env, ptrs, jparams);
    return output_ptr_jarray;
  }
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    forward
 * Signature: (JLjava/util/Map;)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_tencent_angel_pytorch_Torch_importance(
    JNIEnv *env, jclass jcls, jlong jptr, jobject jparams) {
  using namespace angel;
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  // build inputs
  std::vector<torch::jit::IValue> inputs;
  std::vector<std::pair<std::string, void *>> ptrs;

  int batch_size = angel::jni_map_get_int(env, jparams, "batch_size");
  // data inputs
  inputs.emplace_back(batch_size);
  ptrs.emplace_back(
      std::make_pair("batch_size", nullptr)); // why adding a nullptr ?
  add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_INT64, "index");
  add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_INT64, "feats");
  add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_FLOAT, "values");

  add_inputs(env, &inputs, &ptrs, jparams, ptr->get_type());
  auto output = ptr->importance(inputs).toTensor();
  auto output_ptr = output.data_ptr();
  DEFINE_JFLOATARRAY(output_ptr, output.numel());

  // release java arrays
  release_array(env, ptrs, jparams);
  return output_ptr_jarray;
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    backward
 * Signature: (JLjava/util/Map;)F
 */
JNIEXPORT jfloat JNICALL Java_com_tencent_angel_pytorch_Torch_backward(
    JNIEnv *env, jclass jcls, jlong jptr, jobject jparams) {
  using namespace angel;
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  // build inputs
  std::vector<torch::jit::IValue> inputs;
  std::vector<std::pair<std::string, void *>> ptrs;
  int batch_size = angel::jni_map_get_int(env, jparams, "batch_size");
  int training = 1;
  // data inputs
  inputs.emplace_back(batch_size);
  inputs.emplace_back(training); // 1 represents training
  ptrs.emplace_back(std::make_pair("batch_size", nullptr));
  ptrs.emplace_back(std::make_pair("training", nullptr));
  add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_INT64, "index");
  add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_INT64, "feats");
  add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_FLOAT, "values");
  // targets
  add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_FLOAT, "targets");
  // parameters inputs
  add_inputs(env, &inputs, &ptrs, jparams,
             ptr->get_type()); // add fields by type

  auto loss = ptr->recommend_backward(inputs);
  set_grads(env, inputs, ptrs, jparams);
  release_array(env, ptrs, jparams);

  return loss;
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    dssmForward
 * Signature: (JLjava/util/Map;)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_tencent_angel_pytorch_Torch_dssmForward(
    JNIEnv *env, jclass jcls, jlong jptr, jobject jparams) {
  using namespace angel;
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  // build inputs
  std::vector<torch::jit::IValue> inputs;
  std::vector<std::pair<std::string, void *>> ptrs;

  int batch_size = angel::jni_map_get_int(env, jparams, "batch_size");
  // data inputs
  inputs.emplace_back(batch_size);
  ptrs.emplace_back(
      std::make_pair("batch_size", nullptr)); // why adding a nullptr ?

  add_input_dssm(env, &inputs, &ptrs, jparams, TORCH_OPTION_INT64, "index",
                 "inputs_sizes");
  add_input_dssm(env, &inputs, &ptrs, jparams, TORCH_OPTION_FLOAT, "values",
                 "inputs_sizes");
  add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_FLOAT_GRAD,
            "embeddings_array", "embeddings_sizes", "inputs_sizes");
  add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "mats",
            "mats_sizes");
  // parameters inputs

  auto output = ptr->forward(inputs).toTensor();
  auto output_ptr = output.data_ptr();
  DEFINE_JFLOATARRAY(output_ptr, batch_size);

  // release java arrays
  release_array(env, ptrs, jparams);
  return output_ptr_jarray;
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    dssmBackward
 * Signature: (JLjava/util/Map;)F
 */
JNIEXPORT jfloat JNICALL Java_com_tencent_angel_pytorch_Torch_dssmBackward(
    JNIEnv *env, jclass jcls, jlong jptr, jobject jparams) {
  using namespace angel;
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  // build inputs
  std::vector<torch::jit::IValue> inputs;
  std::vector<std::pair<std::string, void *>> ptrs;
  int batch_size = angel::jni_map_get_int(env, jparams, "batch_size");
  // data inputs
  inputs.emplace_back(batch_size);
  ptrs.emplace_back(std::make_pair("batch_size", nullptr));
  add_input_dssm(env, &inputs, &ptrs, jparams, TORCH_OPTION_INT64, "index",
                 "inputs_sizes");
  add_input_dssm(env, &inputs, &ptrs, jparams, TORCH_OPTION_FLOAT, "values",
                 "inputs_sizes");
  add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_FLOAT_GRAD,
            "embeddings_array", "embeddings_sizes", "inputs_sizes");
  add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "mats",
            "mats_sizes");

  // targets
  jarray targets = (jarray)jni_map_get(env, jparams, "targets");
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
 * Method:    word2vecBackward
 * Signature: (JLjava/util/Map;)F
 */
JNIEXPORT jfloat JNICALL Java_com_tencent_angel_pytorch_Torch_word2vecBackward(
    JNIEnv *env, jclass jcls, jlong jptr, jobject jparams) {
  using namespace angel;
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  // build inputs
  std::vector<torch::jit::IValue> inputs;
  std::vector<std::pair<std::string, void *>> ptrs;
  int batch_size = angel::jni_map_get_int(env, jparams, "batch_size");
  // data inputs
  inputs.emplace_back(batch_size);
  ptrs.emplace_back(std::make_pair("batch_size", nullptr));
  // parameters inputs
  add_word2vec_inputs(env, &inputs, &ptrs, jparams);
  at::Tensor undefined;
  auto loss = ptr->backward(inputs, undefined);
  set_grads(env, inputs, ptrs, jparams);
  release_array(env, ptrs, jparams);
  return loss;
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    save
 * Signature: (JLjava/lang/String;Ljava/util/Map;)V
 */
JNIEXPORT void JNICALL Java_com_tencent_angel_pytorch_Torch_save(
    JNIEnv *env, jclass jcls, jlong jptr, jobject jparams) {
  using namespace angel;
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  // build parameters
  std::vector<torch::jit::IValue> parameters;
  std::vector<std::pair<std::string, void *>> ptrs;
  // parameters inputs
  auto input_index =
      add_parameters(env, &parameters, &ptrs, jparams, ptr->get_type());
  for (auto &pair : input_index) {
    ptr->set_parameter(pair.first, parameters[pair.second]);
  }
  //  ptr->save_module(parameters, ptr->get_type());
  jstring jpath = (jstring)jni_map_get(env, jparams, "path");
  const char *path = env->GetStringUTFChars(jpath, 0);
  ptr->save(path);
  release_array(env, ptrs, jparams);
}

void gcn_add_parameters(JNIEnv *env, std::vector<torch::jit::IValue> *inputs,
                        std::vector<std::pair<std::string, void *>> *ptrs,
                        jobject jparams, jboolean sparse) {
  using namespace angel;

  // calculate x dimension and add x
  int feature_dim = jni_map_get_int(env, jparams, "feature_dim");
  const std::vector<std::string> x_keys = {"x", "pos_x", "neg_x"};
  for (auto x_key : x_keys)
    if (jni_map_contain(env, jparams, x_key)) {
      jarray x = (jarray)jni_map_get(env, jparams, x_key);
      int x_num = env->GetArrayLength(x) / feature_dim;
      if (sparse) {
        int embedding_dim = jni_map_get_int(env, jparams, "embedding_dim");
        int x_num = env->GetArrayLength(x) / embedding_dim;
        add_input(env, inputs, ptrs, jparams, {x_num, embedding_dim},
                  TORCH_OPTION_FLOAT_GRAD, x_key);
      } else {
        add_input(env, inputs, ptrs, jparams, {x_num, feature_dim},
                  TORCH_OPTION_FLOAT, x_key);
      }
    }

  // add edge_index
  const std::vector<std::string> edge_keys = {"labeled_edge_index",
                                              "edge_index", "first_edge_index",
                                              "second_edge_index"};
  for (auto edge_key : edge_keys)
    if (jni_map_contain(env, jparams, edge_key)) {
      jarray edge_index = (jarray)jni_map_get(env, jparams, edge_key);
      int edge_num = env->GetArrayLength(edge_index) / 2;
      add_input(env, inputs, ptrs, jparams, {2, edge_num}, TORCH_OPTION_INT64,
                edge_key);
    }

  // add edge_index
  const std::vector<std::string> weight_keys = {
      "edge_weight", "first_edge_weight", "second_edge_weight"};
  for (auto weight_key : weight_keys)
    if (jni_map_contain(env, jparams, weight_key)) {
      jarray edge_index = (jarray)jni_map_get(env, jparams, weight_key);
      int edge_num = env->GetArrayLength(edge_index);
      add_input(env, inputs, ptrs, jparams, {1, edge_num}, TORCH_OPTION_FLOAT,
                weight_key);
    }

  // add edge_index
  if (jni_map_contain(env, jparams, "edge_dim")) {
    int edge_dim = jni_map_get_int(env, jparams, "edge_dim");
    const std::vector<std::string> edge_feature_keys = {
        "edge_feature", "first_edge_feature", "second_edge_feature"};
    for (auto edge_feature_key : edge_feature_keys)
      if (jni_map_contain(env, jparams, edge_feature_key)) {
        jarray edge_index = (jarray)jni_map_get(env, jparams, edge_feature_key);
        int edge_num = env->GetArrayLength(edge_index) / edge_dim;
        add_input(env, inputs, ptrs, jparams, {edge_num, edge_dim},
                  TORCH_OPTION_FLOAT, edge_feature_key);
      }
  }

  const std::vector<std::string> src_nodes = {"srcs"};
  for (auto src_node: src_nodes)
      if (jni_map_contain(env, jparams, src_node)) {
          jarray nodes = (jarray) jni_map_get(env, jparams, src_node);
          int src_nodes_num = env->GetArrayLength(nodes);
          add_input(env, inputs, ptrs, jparams, {src_nodes_num}, TORCH_OPTION_INT64, src_node);
      }

  const std::string src_type_str = "src_type";
  if (jni_map_contain(env, jparams, src_type_str)) {
    int src_type = jni_map_get_int(env, jparams, src_type_str);
    inputs->emplace_back(src_type);
    ptrs->push_back(std::make_pair(src_type_str, nullptr));
  }

  if (jni_map_contain(env, jparams, "context_dim")) {
    int context_dim = jni_map_get_int(env, jparams, "context_dim");
    const std::vector<std::string> context_keys = {"context_x"};
    for (auto context_x: context_keys)
        if (jni_map_contain(env, jparams, context_x)) {
            jarray x = (jarray) jni_map_get(env, jparams, context_x);
            int x_num = env->GetArrayLength(x) / context_dim;
            add_input(env, inputs, ptrs, jparams, {x_num, context_dim}, TORCH_OPTION_FLOAT_GRAD, context_x);
        }
    }

  if (jni_map_contain(env, jparams, "negative_num")) {
    int negative_num = jni_map_get_int(env, jparams, "negative_num");
    const std::vector<std::string> negative_keys = {"negatives"};
    for (auto x_key: negative_keys)
      if (jni_map_contain(env, jparams, x_key)) {
        jarray x = (jarray) jni_map_get(env, jparams, x_key);
        int num = env->GetArrayLength(x) / negative_num;
        add_input(env, inputs, ptrs, jparams, {num, negative_num}, TORCH_OPTION_INT64, x_key);
      }
    }

  if (jni_map_contain(env, jparams, "edge_type_num") && jni_map_contain(env, jparams, "neighbor_num")) {
      int edge_type_num = jni_map_get_int(env, jparams, "edge_type_num");
      int neighbor_num = jni_map_get_int(env, jparams, "neighbor_num");
      const std::vector<std::string> neighbor_keys = {"neighbors"};
      for (auto x_key: neighbor_keys) {
          if (jni_map_contain(env, jparams, x_key)) {
            jarray x = (jarray) jni_map_get(env, jparams, x_key);
            int num = env->GetArrayLength(x) / neighbor_num / edge_type_num;
            add_input(env, inputs, ptrs, jparams, {edge_type_num, num, neighbor_num}, TORCH_OPTION_INT64, x_key);
          }
      }
      if (jni_map_contain(env, jparams, "neighbors_type")) {
          jarray x = (jarray) jni_map_get(env, jparams, "neighbors_type");
          int num = env->GetArrayLength(x) / edge_type_num;
          add_input(env, inputs, ptrs, jparams, {edge_type_num, num}, TORCH_OPTION_INT64, "neighbors_type");
      }
      if (jni_map_contain(env, jparams, "neighbors_flag")) {
          jarray x = (jarray) jni_map_get(env, jparams, "neighbors_flag");
          add_input(env, inputs, ptrs, jparams, {edge_type_num}, TORCH_OPTION_INT64, "neighbors_flag");
      }
  }

  // add edge_type
  const std::vector<std::string> types_keys = {"first_edge_type",
                                               "second_edge_type"};
  for (auto type_key : types_keys) {
    if (jni_map_contain(env, jparams, type_key)) {
      jarray types = (jarray)jni_map_get(env, jparams, type_key);
      int type_num = env->GetArrayLength(types);
      add_input(env, inputs, ptrs, jparams, {type_num}, TORCH_OPTION_INT64,
                type_key);
    }
  }

  // add embedding
  if (jni_map_contain(env, jparams, "hidden_dim")) {
    int hidden_dim = jni_map_get_int(env, jparams, "hidden_dim");
    const std::vector<std::string> embedding_keys = {"embedding"};
    for (auto &embedding_key : embedding_keys) {
      if (jni_map_contain(env, jparams, embedding_key)) {
        jarray embed = (jarray)jni_map_get(env, jparams, embedding_key);
        int embed_num = env->GetArrayLength(embed) / hidden_dim;
        add_input(env, inputs, ptrs, jparams, {embed_num, hidden_dim},
                  TORCH_OPTION_FLOAT, embedding_key);
      }
    }
  }

  // add bipartite u and i
  if (jni_map_contain(env, jparams, "user_feature_dim")) {
    int user_feature_dim = jni_map_get_int(env, jparams, "user_feature_dim");
    const std::vector<std::string> u_keys = {"u", "pos_u", "neg_u"};
    for (auto u_key : u_keys) {
      if (jni_map_contain(env, jparams, u_key)) {
        jarray u = (jarray)jni_map_get(env, jparams, u_key);
        int u_num = env->GetArrayLength(u) / user_feature_dim;
        if (sparse) {
          int embedding_dim = jni_map_get_int(env, jparams, "u_embedding_dim");
          int u_num = env->GetArrayLength(u) / embedding_dim;
          add_input(env, inputs, ptrs, jparams, {u_num, embedding_dim},
                    TORCH_OPTION_FLOAT_GRAD, u_key);
        } else {
          add_input(env, inputs, ptrs, jparams, {u_num, user_feature_dim},
                    TORCH_OPTION_FLOAT, u_key);
        }
      }
    }
  }
  if (jni_map_contain(env, jparams, "item_feature_dim")) {
    int item_feature_dim = jni_map_get_int(env, jparams, "item_feature_dim");
    const std::vector<std::string> i_keys = {"i", "pos_i", "neg_i"};
    for (auto i_key : i_keys) {
      if (jni_map_contain(env, jparams, i_key)) {
        jarray i = (jarray)jni_map_get(env, jparams, i_key);
        int i_num = env->GetArrayLength(i) / item_feature_dim;
        if (sparse) {
          int embedding_dim = jni_map_get_int(env, jparams, "i_embedding_dim");
          int i_num = env->GetArrayLength(i) / embedding_dim;
          add_input(env, inputs, ptrs, jparams, {i_num, embedding_dim},
                    TORCH_OPTION_FLOAT_GRAD, i_key);
        } else {
          add_input(env, inputs, ptrs, jparams, {i_num, item_feature_dim},
                    TORCH_OPTION_FLOAT, i_key);
        }
      }
    }
  }

  // add bipartite edge_index
  const std::vector<std::string> bi_edge_keys = {
      "u_edge_index", "first_u_edge_index", "second_u_edge_index",
      "i_edge_index", "first_i_edge_index", "second_i_edge_index"};
  for (auto bi_edge_key : bi_edge_keys) {
    if (jni_map_contain(env, jparams, bi_edge_key)) {
      jarray bi_edge_index = (jarray)jni_map_get(env, jparams, bi_edge_key);
      int bi_edge_num = env->GetArrayLength(bi_edge_index) / 2;
      add_input(env, inputs, ptrs, jparams, {2, bi_edge_num},
                TORCH_OPTION_INT64, bi_edge_key);
    }
  }

  const std::vector<std::string> bi_item_type_keys = {
      "edge_type",          "first_u_edge_type",   "first_i_edge_type",
      "second_u_edge_type", "first_u_edge_i_type", "second_u_edge_i_type"};
  for (auto bi_item_type_key : bi_item_type_keys) {
    if (jni_map_contain(env, jparams, bi_item_type_key)) {
      jarray bi_item_type_index =
          (jarray)jni_map_get(env, jparams, bi_item_type_key);
      int bi_edge_num = env->GetArrayLength(bi_item_type_index);
      add_input(env, inputs, ptrs, jparams, {1, bi_edge_num},
                TORCH_OPTION_INT32, bi_item_type_key);
    }
  }

  const std::vector<std::string> batch_keys = {
      "batch_ids",     "u_batch_ids",     "i_batch_ids",
      "neg_batch_ids", "neg_u_batch_ids", "neg_i_batch_ids"};
  for (auto batch_key : batch_keys) {
    if (jni_map_contain(env, jparams, batch_key)) {
      jarray batch_index = (jarray)jni_map_get(env, jparams, batch_key);
      int batch_index_num = env->GetArrayLength(batch_index);
      add_input(env, inputs, ptrs, jparams, {1, batch_index_num},
                TORCH_OPTION_INT32, batch_key);
    }
  }

  const std::vector<std::string> field_keys = {
      "field_ids",     "u_field_ids",     "i_field_ids",
      "neg_field_ids", "neg_u_field_ids", "neg_i_field_ids"};
  for (auto field_key : field_keys) {
    if (jni_map_contain(env, jparams, field_key)) {
      jarray field_index = (jarray)jni_map_get(env, jparams, field_key);
      int field_index_num = env->GetArrayLength(field_index);
      add_input(env, inputs, ptrs, jparams, {1, field_index_num},
                TORCH_OPTION_INT32, field_key);
    }
  }

  int batchIds_size = jni_map_get_int(env, jparams,"batchIds_size");
  const std::vector<std::string> id_keys = {"batchIds", "fieldIds"};
    for (auto id_key: id_keys){
    if (jni_map_contain(env, jparams, id_key)) {
      add_input(env, inputs, ptrs, jparams, id_key, 1, batchIds_size);
    }
  }

  const std::string agg_key = "agg_node";
  if (jni_map_contain(env, jparams, agg_key)) {
    int agg_node = jni_map_get_int(env, jparams, agg_key);
    inputs->emplace_back(agg_node);
  }

  if (jni_map_contain(env, jparams, "feature_dense_dims")) {
    const std::vector<std::string> f_keys = {"feats_dense"};
    for (auto f_key: f_keys) {
      if (jni_map_contain(env, jparams, f_key)) {
        add_input(env, inputs, ptrs, jparams, false, f_key, "feature_dense_dims", "embedding_dims");
      }
    }
  }
}

void gcn_set_weights(JNIEnv *env, angel::TorchModel *model_ptr,
                     std::vector<std::pair<std::string, void *>> *ptrs,
                     jobject jparams) {
  using namespace angel;
  jboolean is_copy;
  jarray weights = (jarray)jni_map_get(env, jparams, "weights");
  void *weights_cptr = env->GetPrimitiveArrayCritical(weights, &is_copy);
  ptrs->push_back(std::make_pair("weights", weights_cptr));
  model_ptr->set_gcn_parameters(weights_cptr, env->GetArrayLength(weights));
}

void gcn_set_grads(JNIEnv *env, angel::TorchModel *model_ptr,
                   const std::vector<std::pair<std::string, void *>> &ptrs,
                   jobject jparams) {
  using namespace angel;
  jarray weights = (jarray)jni_map_get(env, jparams, "weights");
  for (auto item : ptrs)
    if (item.first == "weights")
      model_ptr->set_gcn_gradients(item.second, env->GetArrayLength(weights));
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    gcnBackward
 * Signature: (JLjava/util/Map;)F
 */
JNIEXPORT jfloat JNICALL Java_com_tencent_angel_pytorch_Torch_gcnBackward(
    JNIEnv *env, jclass jcls, jlong jptr, jobject jparams, jboolean sparse) {
  using namespace angel;
  DEFINE_MODEL_PTR(angel::TorchModel, jptr)
  // build inputs
  std::vector<torch::jit::IValue> inputs;
  std::vector<std::pair<std::string, void *>> ptrs;

  gcn_add_parameters(env, &inputs, &ptrs, jparams, sparse);
  gcn_set_weights(env, ptr, &ptrs, jparams);

  // get targets if defined
  at::Tensor targets; // undefined targets
  if (jni_map_contain(env, jparams, "targets")) {
    jboolean is_copy;
    jarray jtargets = (jarray)jni_map_get(env, jparams, "targets");
    void *jtargets_cptr = env->GetPrimitiveArrayCritical(jtargets, &is_copy);
    targets = torch::from_blob(jtargets_cptr, {env->GetArrayLength(jtargets)},
                               TORCH_OPTION_FLOAT);
    ptrs.push_back(std::make_pair("targets", jtargets_cptr));
  }

  ptr->zero_grad();
  auto loss = ptr->backward(inputs, targets);
  gcn_set_grads(env, ptr, ptrs, jparams);
  if (sparse) {
    set_grads(env, inputs, ptrs, jparams);
  }

  release_array(env, ptrs, jparams);
  return loss;
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    gatneBackward
 * Signature: (JLjava/util/Map;)F
 */
JNIEXPORT jfloat JNICALL Java_com_tencent_angel_pytorch_Torch_gatneBackward
        (JNIEnv *env, jclass jcls, jlong jptr, jobject jparams, jboolean sparse) {
    using namespace angel;
    DEFINE_MODEL_PTR(angel::TorchModel, jptr)
    // build inputs
    std::vector<torch::jit::IValue> inputs;
    std::vector<std::pair<std::string, void*>> ptrs;

    gcn_add_parameters(env, &inputs, &ptrs, jparams, sparse);
    gcn_set_weights(env, ptr, &ptrs, jparams);

    // get targets if defined
    at::Tensor dsts; // undefined targets
    if (jni_map_contain(env, jparams, "dsts")) {
        jboolean is_copy;
        jarray jtargets = (jarray) jni_map_get(env, jparams, "dsts");
        void* jtargets_cptr = env->GetPrimitiveArrayCritical(jtargets, &is_copy);
        dsts = torch::from_blob(jtargets_cptr, {env->GetArrayLength(jtargets)}, TORCH_OPTION_INT64);
        ptrs.push_back(std::make_pair("dsts", jtargets_cptr));
    }

    ptr->zero_grad();
    auto loss = ptr->backward(inputs, dsts);
    gcn_set_grads(env, ptr, ptrs, jparams);
    set_grads(env, inputs, ptrs, jparams);

    release_array(env, ptrs, jparams);
    return loss;
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    gcnExecMethod
 * Signature: (JLjava/lang/String;Ljava/util/Map;)Ljava/lang/Object;
 */
JNIEXPORT jobject JNICALL Java_com_tencent_angel_pytorch_Torch_gcnExecMethod(
    JNIEnv *env, jclass jcls, jlong jptr, jstring jmethod, jobject jparams, jboolean sparse) {
  using namespace angel;
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  // build inputs
  std::vector<torch::jit::IValue> inputs;
  std::vector<std::pair<std::string, void *>> ptrs;

  gcn_add_parameters(env, &inputs, &ptrs, jparams, sparse);
  gcn_set_weights(env, ptr, &ptrs, jparams);

  const char *method = env->GetStringUTFChars(jmethod, 0);
  auto output = ptr->exec_method(method, inputs).toTensor();
  env->ReleaseStringUTFChars(jmethod, method);

  jarray joutput;
  if (output.dtype().id() == caffe2::TypeIdentifier::Get<float>()) {
    joutput = env->NewFloatArray(output.numel());
    env->SetFloatArrayRegion((jfloatArray)joutput, 0, output.numel(),
                             output.data<float>());
  } else if (output.dtype().id() == caffe2::TypeIdentifier::Get<int64_t>()) {
    joutput = env->NewLongArray(output.numel());
    env->SetLongArrayRegion((jlongArray)joutput, 0, output.numel(),
                            output.data<int64_t>());
  } else
    throw std::logic_error("the dtype for predict should be float or int64");

  release_array(env, ptrs, jparams);
  return joutput;
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    setParameters
 * Signature: (J[F)V
 */
JNIEXPORT void JNICALL Java_com_tencent_angel_pytorch_Torch_setParameters(
    JNIEnv *env, jclass jcls, jlong jptr, jfloatArray jvals) {
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  jboolean is_copy;
  DEFINE_PRIMITIVE_ARRAY(jvals)
  int len = env->GetArrayLength(jvals);
  ptr->set_gcn_parameters(jvals_cptr, len);
  RELEASE_PRIMITIVE_ARRAY(jvals);
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getParameters
 * Signature: (J)[F
 */
JNIEXPORT jfloatArray JNICALL
Java_com_tencent_angel_pytorch_Torch_getParameters(JNIEnv *env, jclass jcls,
                                                   jlong jptr) {
  using namespace angel;
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  std::vector<at::Tensor> tensors = ptr->get_parameters();
  int size = 0;
  for (auto const &t : tensors)
    size += t.size(0);

  jfloatArray array = env->NewFloatArray(size);

  int start = 0;
  for (auto const &t : tensors) {
    env->SetFloatArrayRegion(array, start, static_cast<jsize>(t.size(0)),
                             t.data_ptr<float>());
    start += t.size(0);
  }

  return array;
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getMatsParameters
 * Signature: (J)[F
 */
JNIEXPORT jfloatArray JNICALL
Java_com_tencent_angel_pytorch_Torch_getMatsParameters(JNIEnv *env, jclass jcls,
                                                       jlong jptr) {
  using namespace angel;
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  std::vector<at::Tensor> tensors = ptr->get_mats_parameters();
  int size = 0;
  for (auto const &t : tensors)
    size += t.size(0);

  jfloatArray array = env->NewFloatArray(size);

  int start = 0;
  for (auto const &t : tensors) {
    env->SetFloatArrayRegion(array, start, static_cast<jsize>(t.size(0)),
                             t.data_ptr<float>());
    start += t.size(0);
  }
  return array;
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    gcnSave
 * Signature: (JLjava/util/Map;)V
 */
JNIEXPORT void JNICALL Java_com_tencent_angel_pytorch_Torch_gcnSave(
    JNIEnv *env, jclass jcls, jlong jptr, jobject jparams) {
  using namespace angel;
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  std::vector<std::pair<std::string, void *>> ptrs;
  gcn_set_weights(env, ptr, &ptrs, jparams);
  jstring jpath = (jstring)jni_map_get(env, jparams, "path");
  const char *path = env->GetStringUTFChars(jpath, 0);
  ptr->save(path);
  env->ReleaseStringUTFChars(jpath, path);
  release_array(env, ptrs, jparams);
}