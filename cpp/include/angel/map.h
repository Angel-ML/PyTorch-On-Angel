//
// Created by leleyu on 2019-06-10.
//

#ifndef TORCH_ON_ANGEL_MAP_H
#define TORCH_ON_ANGEL_MAP_H

#include <jni.h>
#include <string>

namespace angel {

int jni_integer_int(jobject integer);
int jni_map_size(JNIEnv *env, jobject map);
jobject jni_map_get(JNIEnv *env, jobject map, jobject key);
jobject jni_map_get(JNIEnv *env, jobject map, const std::string& key);
int jni_map_get_int(JNIEnv *env, jobject map, jobject key);
int jni_map_get_int(JNIEnv *env, jobject map, const std::string& key);
jobject jni_map_new(JNIEnv *env);
void jni_map_set(JNIEnv *env, jobject map, const std::string& key, jobject value);
void jni_map_set(JNIEnv *env, jobject map, jobject key, jobject value);
void jni_map_set(JNIEnv *env, jobject map, const std::string& key, int value);
bool jni_map_contain(JNIEnv *env, jobject map, const std::string& key);
bool jni_map_contain(JNIEnv *env, jobject map, jobject key);

} // namespace angel

#endif //TORCH_ON_ANGEL_MAP_H
