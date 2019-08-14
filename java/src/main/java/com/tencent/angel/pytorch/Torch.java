package com.tencent.angel.pytorch;

import java.util.Map;

public class Torch {

  public static native String name(long ptr);

  public static native void setNumThreads(int nThreads);

  public static native long initPtr(String path);

  public static native void destroyPtr(long ptr);

  public static native String getType(long ptr);

  public static native int[] getMatsSize(long ptr);

  public static native int getInputDim(long ptr);

  public static native int getNumFields(long ptr);

  public static native int getEmbeddingDim(long ptr);

  public static native int getParametersTotalSize(long ptr);

  /* set parameters to torch */
  public static native void setParameters(long ptr, float[] values);

  /* forward */
  public static native float[] forward(long ptr, Map<String, Object> params, boolean serving);

  /* backward */
  public static native float backward(long ptr, Map<String, Object> params);

  /* save module */
  public static native void save(long ptr, Map<String, Object> params);

  /* graph backward */
  public static native float gcnBackward(long ptr, Map<String, Object> params);

  /* graph exec */
  public static native Object gcnExecMethod(long ptr, String method, Map<String, Object> params);

  /* graph get all parameters */
  public static native float[] getParameters(long ptr);

}
