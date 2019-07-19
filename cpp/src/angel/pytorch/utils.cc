//
// Created by leleyu on 2019-05-17.
//

#include <angel/pytorch/utils.h>
#include <angel/map.h>

namespace angel {
std::vector<at::Tensor> make_mat_tensors(void* data_ptr, void* size_ptr, int n_data, int n_size) {
  auto* data_fptr = reinterpret_cast<float*>(data_ptr);
  auto* size_iptr = reinterpret_cast<int*>(size_ptr);
  int n_sum = 0;
  for (int i = 0; i < n_size; i += 2)
    n_sum += size_iptr[i] * size_iptr[i + 1];

  assert(n_sum == n_data);

  std::vector<at::Tensor> tensors;
  for (int i = 0; i < n_size; i += 2) {
    auto t = torch::from_blob(data_fptr, {size_iptr[i], size_iptr[i + 1]}, TORCH_OPTION_FLOAT_GRAD);
    tensors.push_back(t);
    data_fptr += size_iptr[i] * size_iptr[i + 1];
  }

  return tensors;
}

void set_mat_grads(JNIEnv* env, const std::vector<at::Tensor>& tensors, jfloatArray jmats,
                   void* size_ptr, int n_size) {
  auto* size_iptr = reinterpret_cast<int*>(size_ptr);
  int start = 0;
  for (int i = 0; i < n_size; i += 2) {
    int len = size_iptr[i] * size_iptr[i + 1];
    env->SetFloatArrayRegion(jmats, start, len, tensors[i / 2].data<float>());
    start += len;
  }
}

void add_input(JNIEnv* env,
  std::vector<torch::jit::IValue>* inputs,
  std::vector<std::pair<std::string, void*>> *ptrs,
  jobject params,
  const std::vector<int64_t>& sizes,
  const torch::TensorOptions& option,
  const std::string& key) {
  auto array = (jarray) angel::jni_map_get(env, params, env->NewStringUTF(key.data()));
  add_input(env, inputs, ptrs, array, sizes, option, key);
}

void add_input(JNIEnv* env,
               std::vector<torch::jit::IValue>* inputs,
               std::vector<std::pair<std::string, void*>> *ptrs,
               jobject params,
               const torch::TensorOptions& option,
               const std::string& key) {
  auto array = (jarray) angel::jni_map_get(env, params, env->NewStringUTF(key.data()));
  add_input(env, inputs, ptrs, array, {env->GetArrayLength(array)}, option, key);
}

void add_input(JNIEnv* env,
               std::vector<torch::jit::IValue>* inputs,
               std::vector<std::pair<std::string, void*>> *ptrs,
               jarray array,
               const std::vector<int64_t>& sizes,
               const torch::TensorOptions& option,
               const std::string& key) {
  jboolean is_copy;
  void* c_ptr = env->GetPrimitiveArrayCritical(array, &is_copy);
  auto tensor = torch::from_blob(c_ptr, sizes, option);
  inputs->emplace_back(tensor);
  ptrs->push_back(std::make_pair(key, c_ptr));
}

void add_input(JNIEnv* env,
               std::vector<torch::jit::IValue>* inputs,
               std::vector<std::pair<std::string, void*>> *ptrs,
               jobject params,
               const torch::TensorOptions& option,
               const std::string& data_key,
               const std::string& size_key) {
  auto data_array = (jarray) angel::jni_map_get(env, params, data_key);
  auto size_array = (jarray) angel::jni_map_get(env, params, size_key);
  jboolean is_copy;
  void* p_data = env->GetPrimitiveArrayCritical(data_array, &is_copy);
  auto* p_fdata = reinterpret_cast<float*>(p_data);
  void* p_size = env->GetPrimitiveArrayCritical(size_array, &is_copy);
  auto* p_isize = reinterpret_cast<int*>(p_size);
  int n_size = env->GetArrayLength(size_array);
  int n_sum = 0;
  for (int i = 0; i < n_size; i += 2)
    n_sum += p_isize[i] * p_isize[i + 1];
  assert(n_sum == env->GetArrayLength(data_array));

  std::vector<at::Tensor> tensors;
  for (int i = 0; i < n_size; i += 2) {
    auto t = torch::from_blob(p_fdata, {p_isize[i], p_isize[i + 1]}, option);
    tensors.push_back(t);
    p_fdata += p_isize[i] * p_isize[i + 1];
  }

  inputs->emplace_back(tensors);
  ptrs->push_back(std::make_pair(data_key, p_data));
  env->ReleasePrimitiveArrayCritical(size_array, p_size, 0);
}

void release_array(JNIEnv *env,
  const std::vector<std::pair<std::string, void*>>& ptrs,
  jobject params) {
  for (const auto& it: ptrs) {
    std::string key = it.first;
    void* ptr = it.second;
    if (ptr != nullptr) {
      auto array = (jarray) angel::jni_map_get(env, params, key);
      env->ReleasePrimitiveArrayCritical(array, ptr, 0);
    }
  }
}

void set_grads(JNIEnv *env,
  std::vector<torch::jit::IValue> &inputs,
  std::vector<std::pair<std::string, void*>> &ptrs,
  jobject params) {
  assert(inputs.size() == ptrs.size());
  size_t size = inputs.size();
  for (size_t i = 0; i < size; i++) {
    if (inputs[i].isTensor() && inputs[i].toTensor().grad().defined()) {
      auto array = (jarray) jni_map_get(env, params, ptrs[i].first);
      env->SetFloatArrayRegion((jfloatArray) array, 0, env->GetArrayLength(array),
        inputs[i].toTensor().grad().data<float>());
    } else if (inputs[i].isTensorList()) {
      auto list = inputs[i].toTensorList();
      auto array = (jarray) jni_map_get(env, params, ptrs[i].first);
      int start = 0;
      for (size_t pos =0; pos < list.size(); pos++) {
        int len = static_cast<int>(list.get(pos).view({-1}).size(0));
        env->SetFloatArrayRegion((jfloatArray)array, start, len, list.get(pos).grad().data<float>());
        start += len;
      }
    }
  }
}

} // namespace angel

