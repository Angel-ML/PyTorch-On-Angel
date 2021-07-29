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

package com.tencent.angel.pytorch;

import java.util.Map;

public class Torch {

  public static native String name(long ptr);

  public static native void train(long ptr);

  public static native void eval(long ptr);

  public static native long initPtr(String path);

  public static native void destroyPtr(long ptr);

  public static native String getType(long ptr);

  public static native int[] getMatsSize(long ptr);

  public static native int getInputDim(long ptr);

  public static native int getUserInputDim(long ptr);

  public static native int getItemInputDim(long ptr);

  public static native long[] getInputSizes(long ptr);

  public static native int getNumFields(long ptr);

  public static native int getUserNumFields(long ptr);

  public static native int getItemNumFields(long ptr);

  public static native int getEmbeddingDim(long ptr);

  public static native int getUserEmbeddingDim(long ptr);

  public static native int getItemEmbeddingDim(long ptr);

  public static native int[] getEmbeddingsSize(long ptr);

  public static native int getParametersTotalSize(long ptr);

  public static native int[] getDenseColNums(long ptr);

  public static native int[] getSparseColNums(long ptr);

  /* forward */
  public static native float[] forward(long ptr, Map<String, Object> params, boolean serving);

  /* backward */
  public static native float backward(long ptr, Map<String, Object> params);

  /* save module */
  public static native void save(long ptr, Map<String, Object> params);

  public static native float[] dssmForward(long ptr, Map<String, Object> params);

  public static native float dssmBackward(long ptr, Map<String, Object> params);

  /* graph backward */
  public static native float gcnBackward(long ptr, Map<String, Object> params, boolean sparse);

  /* graph exec */
  public static native Object gcnExecMethod(long ptr, String method, Map<String, Object> params);

  /* graph get all parameters */
  public static native float[] getParameters(long ptr);

  /* recommendation get dense mats parameters */
  public static native float[] getMatsParameters(long ptr);

  /* set parameters to torch */
  public static native void setParameters(long ptr, float[] values);

  public static native void gcnSave(long ptr, Map<String, Object> params);
}
