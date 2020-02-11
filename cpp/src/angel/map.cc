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
// Created by leleyu on 2019-06-10.
//

#include <angel/map.h>

namespace angel {

    int jni_integer_int(JNIEnv *env, jobject integer) {
      jclass Integer = env->FindClass("java/lang/Integer");
      jmethodID Integer_intValue = env->GetMethodID(Integer, "intValue", "()I");
      return env->CallIntMethod(integer, Integer_intValue);
    }

    int jni_map_size(JNIEnv *env, jobject map) {
      jclass Map = env->FindClass("java/util/Map");
      jmethodID Map_size = env->GetMethodID(Map, "size", "()I");
      return env->CallIntMethod(map, Map_size);
    }

    jobject jni_map_get(JNIEnv* env, jobject map, jobject key) {
      jclass Map = env->FindClass("java/util/Map");
      jmethodID Map_get = env->GetMethodID(Map, "get", "(Ljava/lang/Object;)Ljava/lang/Object;");
      jobject value = env->CallObjectMethod(map, Map_get, key);
      return value;
    }

    jobject jni_map_get(JNIEnv* env, jobject map, const std::string& key) {
      return jni_map_get(env, map, env->NewStringUTF(key.data()));
    }

    int jni_map_get_int(JNIEnv *env, jobject map, jobject key) {
      jobject value = jni_map_get(env, map, key);
      return jni_integer_int(env, value);
    }

    int jni_map_get_int(JNIEnv *env, jobject map, const std::string& key) {
      return jni_map_get_int(env, map, env->NewStringUTF(key.data()));
    }

    jobject jni_map_new(JNIEnv *env) {
      jclass HashMap = env->FindClass("java/util/HashMap");
      jmethodID HashMap_init = env->GetMethodID(HashMap, "<init>", "()V");
      jobject map = env->NewObject(HashMap, HashMap_init);
      return map;
    }

    void jni_map_set(JNIEnv *env, jobject map, jobject key, jobject value) {
      jclass Map = env->FindClass("java/util/Map");
      jmethodID Map_put = env->GetMethodID(Map, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
      env->CallObjectMethod(map, Map_put, key, value);
    }

    void jni_map_set(JNIEnv *env, jobject map, const std::string& key, jobject value) {
      jni_map_set(env, map, env->NewStringUTF(key.data()), value);
    }

    void jni_map_set(JNIEnv *env, jobject map, const std::string& key, int value) {
      jclass Integer = env->FindClass("java/lang/Integer");
      jmethodID Integer_init = env->GetMethodID(Integer, "<init>", "(I)V");
      jobject val = env->NewObject(Integer, Integer_init, value);
      jni_map_set(env, map, key, val);
    }

    bool jni_map_contain(JNIEnv *env, jobject map, const std::string& key) {
      return jni_map_contain(env, map, env->NewStringUTF(key.data()));
    }

    bool jni_map_contain(JNIEnv *env, jobject map, jobject key) {
      jclass Map = env->FindClass("java/util/Map");
      jmethodID Map_containsKey = env->GetMethodID(Map, "containsKey", "(Ljava/lang/Object;)Z");
      return env->CallBooleanMethod(map, Map_containsKey, key);
    }

}