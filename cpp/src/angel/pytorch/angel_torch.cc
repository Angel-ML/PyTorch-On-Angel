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
    auto output = ptr->forward(inputs);
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



/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    gcnForward
 * Signature: (JLjava/util/Map;)[F
 */
JNIEXPORT jlongArray JNICALL Java_com_tencent_angel_pytorch_Torch_gcnForward
  (JNIEnv *env, jclass jcls, jlong jptr, jobject jparams) {
  using namespace angel;
  DEFINE_MODEL_PTR(angel::TorchModel, jptr);
  // build inputs
  std::vector<torch::jit::IValue> inputs;
  std::vector<std::pair<std::string, void *>> ptrs;

  // calculate x dimention
  int feature_dim = jni_map_get_int(env, jparams, "feature_dim");
  jarray x = (jarray) jni_map_get(env, jparams, "x");
  int x_num = env->GetArrayLength(x) / feature_dim;
  add_input(env, &inputs, &ptrs, jparams, {x_num , feature_dim}, TORCH_OPTION_FLOAT, "x");

  // calculate edge_index dimension
  if (jni_map_contain(env, jparams, "edge_index")) {
    // full batch
    jarray edge_index = (jarray) jni_map_get(env, jparams, "edge_index");
    int edge_num = env->GetArrayLength(edge_index) / 2;
    add_input(env, &inputs, &ptrs, jparams, {2, edge_num}, TORCH_OPTION_INT64, "edge_index");
  } else if (jni_map_contain(env, jparams, "first_edge_index") && jni_map_contain(env, jparams, "second_edge_index")) {
    // mini-batch two-order
    jarray first_edge_index = (jarray) jni_map_get(env, jparams, "first_edge_index");
    jarray second_edge_index = (jarray) jni_map_get(env, jparams, "second_edge_index");
    int first_edge_num = env->GetArrayLength(first_edge_index) / 2;
    int second_edge_num = env->GetArrayLength(second_edge_index) / 2;
    add_input(env, &inputs, &ptrs, jparams, {2, first_edge_num}, TORCH_OPTION_INT64, "first_edge_index");
    add_input(env, &inputs, &ptrs, jparams, {2, second_edge_num}, TORCH_OPTION_INT64, "second_edge_index");
  }

  // set weights to torch models
  jboolean is_copy;
  jarray weights = (jarray) jni_map_get(env, jparams, "weights");
  int weights_length = env->GetArrayLength(weights);
  DEFINE_PRIMITIVE_ARRAY(weights);
  ptr->set_parameters(weights_cptr, weights_length);

  auto output = std::get<1>(ptr->forward(inputs).max(1));
  int output_size = output.view({-1}).size(0);
  jlongArray joutput = env->NewLongArray(output_size);
  env->SetLongArrayRegion(joutput, 0, output_size, output.data<int64_t>());

  RELEASE_PRIMITIVE_ARRAY(weights)
  release_array(env, ptrs, jparams);
  return joutput;
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

  // calculate x dimension
  int feature_dim = jni_map_get_int(env, jparams, "feature_dim");
  jarray x = (jarray) jni_map_get(env, jparams, "x");
  int x_num = env->GetArrayLength(x) / feature_dim;
  // add x to inputs
  add_input(env, &inputs, &ptrs, jparams, {x_num, feature_dim}, TORCH_OPTION_FLOAT, "x");

  // calculate edge_index dimension
  if (jni_map_contain(env, jparams, "edge_index")) {
    // full batch
    jarray edge_index = (jarray) jni_map_get(env, jparams, "edge_index");
    int edge_num = env->GetArrayLength(edge_index) / 2;
    add_input(env, &inputs, &ptrs, jparams, {2, edge_num}, TORCH_OPTION_INT64, "edge_index");
  } else if (jni_map_contain(env, jparams, "first_edge_index") && jni_map_contain(env, jparams, "second_edge_index")) {
    // mini-batch two-order
    jarray first_edge_index = (jarray) jni_map_get(env, jparams, "first_edge_index");
    jarray second_edge_index = (jarray) jni_map_get(env, jparams, "second_edge_index");
    int first_edge_num = env->GetArrayLength(first_edge_index) / 2;
    int second_edge_num = env->GetArrayLength(second_edge_index) / 2;
    add_input(env, &inputs, &ptrs, jparams, {2, first_edge_num}, TORCH_OPTION_INT64, "first_edge_index");
    add_input(env, &inputs, &ptrs, jparams, {2, second_edge_num}, TORCH_OPTION_INT64, "second_edge_index");
  }

  // set weights to torch models
  jboolean is_copy;
  jarray weights = (jarray) jni_map_get(env, jparams, "weights");
  DEFINE_PRIMITIVE_ARRAY(weights)
  ptr->set_parameters(weights_cptr, env->GetArrayLength(weights));

  // get targets
  jarray targets = (jarray) jni_map_get(env, jparams, "targets");
  DEFINE_PRIMITIVE_ARRAY(targets)
  DEFINE_JARRAY_TENSOR_DIM_INT64(targets);

  ptr->zero_grad();
  auto loss = ptr->backward(inputs, targets_tensor);
  ptr->set_grads(weights_cptr, env->GetArrayLength(weights));

  release_array(env, ptrs, jparams);
  RELEASE_PRIMITIVE_ARRAY(weights)
  RELEASE_PRIMITIVE_ARRAY(targets)
  return loss;


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


