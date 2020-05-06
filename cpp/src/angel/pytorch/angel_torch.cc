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
 * Method:    getEmbeddingsSize
 * Signature: (J)[I
 */
JNIEXPORT jintArray JNICALL Java_com_tencent_angel_pytorch_Torch_getEmbeddingsSize
        (JNIEnv *env, jclass jcls, jlong jptr) {
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
JNIEXPORT jlongArray JNICALL Java_com_tencent_angel_pytorch_Torch_getInputSizes
        (JNIEnv *env, jclass jcls, jlong jptr) {
    DEFINE_MODEL_PTR(angel::TorchModel, jptr);
    auto output = ptr->get_inputs_size();
    auto output_ptr = output.data();
    DEFINE_JLONGARRAY(output_ptr, static_cast<jsize>(output.size()));
    return output_ptr_jarray;
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

void add_inputs(JNIEnv *env,
                std::vector<torch::jit::IValue> *inputs,
                std::vector<std::pair<std::string, void *>> *ptrs,
                jobject jparams,
                angel::TorchModelType type) {
    using namespace angel;
    int len = env->GetArrayLength((jarray) angel::jni_map_get(env, jparams, "index"));
    size_t start_pos = inputs->size();
    switch (type) {
        case angel::TorchModelType::EMBEDDINGS_MATS:
            add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "mats", "mats_sizes");
            add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "embeddings_array", "embeddings_sizes", len);
            break;
        case angel::TorchModelType::BIAS_WEIGHT_EMBEDDING_MATS_FIELD:
            add_input(env, inputs, ptrs, jparams, TORCH_OPTION_INT64, "fields");
        case angel::TorchModelType::BIAS_WEIGHT_EMBEDDING_MATS:
            add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "mats", "mats_sizes");
        case angel::TorchModelType::BIAS_WEIGHT_EMBEDDING:
            add_input(env, inputs, ptrs, jparams, {len, angel::jni_map_get_int(env, jparams, "embedding_dim")},
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

void add_word2vec_inputs(JNIEnv* env,
                         std::vector<torch::jit::IValue> *inputs,
                         std::vector<std::pair<std::string, void*>> *ptrs,
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

std::vector<std::pair<std::string, int>> add_parameters(JNIEnv *env,
                                                        std::vector<torch::jit::IValue> *inputs,
                                                        std::vector<std::pair<std::string, void *>> *ptrs,
                                                        jobject jparams,
                                                        angel::TorchModelType type) {
    using namespace angel;

    std::vector<std::pair<std::string, int>> input_index;
    switch (type) {
        case angel::TorchModelType ::EMBEDDINGS_MATS:
            input_index.emplace_back("mats", inputs->size());
            add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "mats", "mats_sizes");
            input_index.emplace_back("embeddings", inputs->size());
            add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "embeddings_array", "embeddings_sizes", "inputs_sizes");
            break;
        case angel::TorchModelType::BIAS_WEIGHT_EMBEDDING_MATS:
        case angel::TorchModelType::BIAS_WEIGHT_EMBEDDING_MATS_FIELD:
            input_index.emplace_back("mats", inputs->size());
            add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "mats", "mats_sizes");
        case angel::TorchModelType::BIAS_WEIGHT_EMBEDDING:
            input_index.emplace_back("embedding", inputs->size());
            add_input(env, inputs, ptrs, jparams,
                      {angel::jni_map_get_int(env, jparams, "dim"), angel::jni_map_get_int(env, jparams, "embedding_dim")},
                      TORCH_OPTION_FLOAT_GRAD, "embedding");
        case angel::TorchModelType::BIAS_WEIGHT:
            input_index.emplace_back("weights", inputs->size());
            add_input(env, inputs, ptrs, jparams, TORCH_OPTION_FLOAT_GRAD, "weights");
            input_index.emplace_back("bias", inputs->size());
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
    ptrs.emplace_back(std::make_pair("batch_size", nullptr));
    add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_INT64, "index");
    add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_INT64, "feats");
    add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_FLOAT, "values");

    if (serving) {
        if (ptr->get_type() == angel::TorchModelType::BIAS_WEIGHT_EMBEDDING_MATS_FIELD) {
            add_input(env, &inputs, &ptrs, jparams, TORCH_OPTION_INT64, "fields");
        }
        auto output = ptr->serving_forward(inputs);
        auto output_ptr = output.to(at::kCPU).data_ptr();
        DEFINE_JFLOATARRAY(output_ptr, batch_size);

        // release java arrays
        release_array(env, ptrs, jparams);
        return output_ptr_jarray;
    } else {
        add_inputs(env, &inputs, &ptrs, jparams, ptr->get_type());
        auto output = ptr->forward(inputs).toTensor();
        auto output_ptr = output.to(at::kCPU).data_ptr();
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
 * Method:    word2vecBackward
 * Signature: (JLjava/util/Map;)F
 */
JNIEXPORT jfloat JNICALL Java_com_tencent_angel_pytorch_Torch_word2vecBackward
        (JNIEnv *env, jclass jcls, jlong jptr, jobject jparams) {
    using namespace angel;
    DEFINE_MODEL_PTR(angel::TorchModel, jptr);
    // build inputs
    std::vector<torch::jit::IValue> inputs;
    std::vector<std::pair<std::string, void*>> ptrs;
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
JNIEXPORT void JNICALL Java_com_tencent_angel_pytorch_Torch_save
        (JNIEnv *env, jclass jcls, jlong jptr, jobject jparams) {
    using namespace angel;
    DEFINE_MODEL_PTR(angel::TorchModel, jptr);
    // build parameters
    std::vector<torch::jit::IValue> parameters;
    std::vector<std::pair<std::string, void *>> ptrs;
    // parameters inputs
    auto input_index = add_parameters(env, &parameters, &ptrs, jparams, ptr->get_type());
    for (auto& pair: input_index) {
        ptr->set_parameter(pair.first, parameters[pair.second]);
    }
    jstring jpath = (jstring) jni_map_get(env, jparams, "path");
    const char* path = env->GetStringUTFChars(jpath, nullptr);
    ptr->save(path);
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
    for (const auto &x_key: x_keys)
        if (jni_map_contain(env, jparams, x_key)) {
            auto x = (jarray) jni_map_get(env, jparams, x_key);
            int x_num = env->GetArrayLength(x) / feature_dim;
            add_input(env, inputs, ptrs, jparams, {x_num, feature_dim}, TORCH_OPTION_FLOAT, x_key);
        }

    // add edge_index
    const std::vector<std::string> edge_keys = {"edge_index", "first_edge_index", "second_edge_index"};
    for (const auto &edge_key: edge_keys)
        if (jni_map_contain(env, jparams, edge_key)) {
            auto edge_index = (jarray) jni_map_get(env, jparams, edge_key);
            int edge_num = env->GetArrayLength(edge_index) / 2;
            add_input(env, inputs, ptrs, jparams, {2, edge_num}, TORCH_OPTION_INT64, edge_key);
        }

    // add edge_type
    const std::vector<std::string> types_keys = {"first_edge_type", "second_edge_type"};
    for (const auto &type_key: types_keys) {
        if (jni_map_contain(env, jparams, type_key)) {
            auto types = (jarray) jni_map_get(env, jparams, type_key);
            int type_num = env->GetArrayLength(types);
            add_input(env, inputs, ptrs, jparams, {type_num}, TORCH_OPTION_INT64, type_key);
        }
    }

    // add embedding
    if (jni_map_contain(env, jparams, "hidden_dim")) {
        int hidden_dim = jni_map_get_int(env, jparams, "hidden_dim");
        const std::vector<std::string> embedding_keys = {"embedding"};
        for (const auto &embedding_key: embedding_keys) {
            if (jni_map_contain(env, jparams, embedding_key)) {
                auto embed = (jarray) jni_map_get(env, jparams, embedding_key);
                int embed_num = env->GetArrayLength(embed) / hidden_dim;
                add_input(env, inputs, ptrs, jparams, {embed_num, hidden_dim}, TORCH_OPTION_FLOAT, embedding_key);
            }
        }
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
    model_ptr->set_gcn_parameters(weights_cptr, env->GetArrayLength(weights));
}

void gcn_set_grads(JNIEnv *env,
                   angel::TorchModel *model_ptr,
                   const std::vector<std::pair<std::string, void*>> &ptrs,
                   jobject jparams) {
    using namespace angel;
    jarray weights = (jarray) jni_map_get(env, jparams, "weights");
    for (auto item : ptrs)
        if (item.first == "weights")
            model_ptr->set_gcn_gradients(item.second, env->GetArrayLength(weights));
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
        targets = torch::from_blob(jtargets_cptr, {env->GetArrayLength(jtargets)}, TORCH_OPTION_FLOAT);
        ptrs.emplace_back("targets", jtargets_cptr);
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
        joutput = env->NewFloatArray(static_cast<jsize>(output.numel()));
        env->SetFloatArrayRegion((jfloatArray)joutput, 0, static_cast<jsize>(output.numel()), output.to(at::kCPU).data_ptr<float>());
    } else if (output.dtype().id() == caffe2::TypeIdentifier::Get<int64_t>()) {
        joutput = env->NewLongArray(static_cast<jsize>(output.numel()));
        env->SetLongArrayRegion((jlongArray)joutput, 0, static_cast<jsize>(output.numel()), output.to(at::kCPU).data_ptr<int64_t>());
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
JNIEXPORT void JNICALL Java_com_tencent_angel_pytorch_Torch_setParameters
        (JNIEnv *env, jclass jcls, jlong jptr, jfloatArray jvals) {
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
        env->SetFloatArrayRegion(array, start, static_cast<jsize>(t.size(0)), t.to(at::kCPU).data_ptr<float>());
        start += t.size(0);
    }

    return array;
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    getMatsParameters
 * Signature: (J)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_tencent_angel_pytorch_Torch_getMatsParameters
        (JNIEnv *env, jclass jcls, jlong jptr) {
    using namespace angel;
    DEFINE_MODEL_PTR(angel::TorchModel, jptr);
    std::vector<at::Tensor> tensors = ptr->get_mats_parameters();
    int size = 0;
    for (auto const& t: tensors)
        size += t.size(0);

    jfloatArray array = env->NewFloatArray(size);

    int start = 0;
    for (auto const& t: tensors) {
        env->SetFloatArrayRegion(array, start, static_cast<jsize>(t.size(0)), t.data_ptr<float>());
        start += t.size(0);
    }
    return array;
}

/*
 * Class:     com_tencent_angel_pytorch_Torch
 * Method:    gcnSave
 * Signature: (JLjava/util/Map;)V
 */
JNIEXPORT void JNICALL Java_com_tencent_angel_pytorch_Torch_gcnSave
        (JNIEnv *env, jclass jcls, jlong jptr, jobject jparams) {
    using namespace angel;
    DEFINE_MODEL_PTR(angel::TorchModel, jptr);
    std::vector<std::pair<std::string, void *>> ptrs;
    gcn_set_weights(env, ptr, &ptrs, jparams);
    jstring jpath = (jstring) jni_map_get(env, jparams, "path");
    const char* path = env->GetStringUTFChars(jpath, nullptr);
    ptr->save(path);
    env->ReleaseStringUTFChars(jpath, path);
    release_array(env, ptrs, jparams);
}