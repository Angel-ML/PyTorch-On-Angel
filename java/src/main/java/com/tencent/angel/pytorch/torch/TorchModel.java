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
package com.tencent.angel.pytorch.torch;

import com.tencent.angel.exception.AngelException;
import com.tencent.angel.ml.math2.matrix.CooLongFloatMatrix;
import com.tencent.angel.pytorch.Torch;
import com.tencent.angel.pytorch.model.TorchModelType;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.spark.SparkEnv;
import scala.Enumeration;

public class TorchModel implements Serializable {

  private static String path;
  private static BlockingQueue<TorchModel> modelsQueue = new LinkedBlockingQueue<TorchModel>();
  private static AtomicInteger counter = new AtomicInteger(0);
  private static int cores = SparkEnv.get().conf()
          .getInt("spark.executor.cores", 1);

  // load library of torch and torch_angel
  static {
    System.loadLibrary("torch_angel");
  }

  private long ptr = 0;

  private TorchModel() {
  }

  public static synchronized TorchModel get() throws InterruptedException {
    if (modelsQueue.isEmpty()) {
      if (path == null) {
        throw new AngelException("Please set path before accessing model");
      }
      TorchModel model = new TorchModel();
      model.init(path);
      modelsQueue.put(model);
      counter.addAndGet(1);
      if (counter.get() > cores + 2) {
        throw new AngelException("The size of torch model exceeds the cores");
      }
    }
    return modelsQueue.take();
  }

  public static void put(TorchModel torchModel) throws InterruptedException {
    modelsQueue.put(torchModel);
  }

  public static void setPath(String path) {
    TorchModel.path = path;
  }

  private void init(String path) {
    ptr = Torch.initPtr(path);
  }

  public String name() {
    return Torch.name(ptr);
  }

  public String getType() {
    return Torch.getType(ptr);
  }

  public int[] getMatsSize() {
    return Torch.getMatsSize(ptr);
  }

  public int getInputDim() {
    return Torch.getInputDim(ptr);
  }

  public int[] getDenseColNums() {
    return Torch.getDenseColNums(ptr);
  }

  public int[] getSparseColNums() {
    return Torch.getSparseColNums(ptr);
  }

  public long[] getInputSizes() {
    return Torch.getInputSizes(ptr);
  }

  public int getNumFields() {
    return Torch.getNumFields(ptr);
  }

  public int getEmbeddingDim() {
    return Torch.getEmbeddingDim(ptr);
  }

  public int[] getEmbeddingsSize() {
    return Torch.getEmbeddingsSize(ptr);
  }

  public int getParametersTotalSize() {
    return Torch.getParametersTotalSize(ptr);
  }

  public int getUserInputDim() { return Torch.getUserInputDim(ptr); }

  public int getItemInputDim() { return Torch.getItemInputDim(ptr); }

  public int getUserNumFields() { return Torch.getUserNumFields(ptr); }

  public int getItemNumFields() { return Torch.getItemNumFields(ptr); }

  public int getUserEmbeddingDim() { return Torch.getUserEmbeddingDim(ptr); }

  public int getItemEmbeddingDim() { return Torch.getItemEmbeddingDim(ptr); }

  public void save(float[] bias, float[] weights, String path) {
    Map<String, Object> params = buildParams(bias, weights);
    params.put("path", path);
    Torch.save(ptr, params);
  }

  public void save(float[] bias, float[] weights, float[] mats, int[] matSizes, String path) {
    Map<String, Object> params = buildParams(bias, weights);
    params.put("mats", mats);
    params.put("mats_sizes", matSizes);
    params.put("path", path);
    Torch.save(ptr, params);
  }

  public void save(float[] bias, float[] weights, float[] embeddings, int embeddingDim, int dim,
                   String path) {
    Map<String, Object> params = buildParams(bias, weights);
    params.put("embedding", embeddings);
    params.put("embedding_dim", embeddingDim);
    params.put("dim", dim);
    params.put("path", path);
    Torch.save(ptr, params);
  }

  public void save(float[] bias, float[] weights, float[] embeddings, int embeddingDim,
                   float[] mats, int[] matSizes, int dim, String path) {
    Map<String, Object> params = buildParams(bias, weights);
    params.put("embedding", embeddings);
    params.put("embedding_dim", embeddingDim);
    params.put("mats", mats);
    params.put("mats_sizes", matSizes);
    params.put("dim", dim);
    params.put("path", path);
    Torch.save(ptr, params);
  }

  public void save(float[] mats, int[] matSizes, float[][] embeddingsArray, int[] embeddingsSize,
                   long[] inputsSizes, String path) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("mats", mats);
    params.put("mats_sizes", matSizes);
    params.put("embeddings_array", embeddingsArray);
    params.put("embeddings_sizes", embeddingsSize);
    params.put("inputs_sizes", inputsSizes);
    params.put("path", path);
    Torch.save(ptr, params);
  }

  private Map<String, Object> buildParams(int batchSize, CooLongFloatMatrix batch) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("batch_size", batchSize);
    params.put("index", batch.getRowIndices());
    params.put("feats", batch.getColIndices());
    params.put("values", batch.getValues());
    return params;
  }

  private Map<String, Object> buildParams(float[] bias, float[] weights) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("bias", bias);
    params.put("weights", weights);
    return params;
  }

  private Map<String, Object> buildParams(int batchSize, CooLongFloatMatrix batch, float[] bias,
                                          float[] weights) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("batch_size", batchSize);
    params.put("index", batch.getRowIndices());
    params.put("feats", batch.getColIndices());
    params.put("values", batch.getValues());
    params.put("bias", bias);
    params.put("weights", weights);
    return params;
  }

  public float[] forward(int batchSize, CooLongFloatMatrix batch) {
    Map<String, Object> params = buildParams(batchSize, batch);
    return Torch.forward(ptr, params, true);
  }

  public float[] forward(int batchSize, CooLongFloatMatrix batch, long[] fields) {
    Map<String, Object> params = buildParams(batchSize, batch);
    params.put("fields", fields);
    return Torch.forward(ptr, params, true);
  }

  public float[] forward(int batchSize, CooLongFloatMatrix batch, float[] bias, float[] weights) {
    Map<String, Object> params = buildParams(batchSize, batch, bias, weights);
    return Torch.forward(ptr, params, false);
  }

  public float[] forward(int batchSize, CooLongFloatMatrix batch, float[] bias, float[] weights,
                         float[] mats, int[] matSizes) {
    Map<String, Object> params = buildParams(batchSize, batch, bias, weights);
    params.put("mats", mats);
    params.put("mats_sizes", matSizes);
    return Torch.forward(ptr, params, false);
  }

  public float[] forward(int batchSize, CooLongFloatMatrix batch, float[] mats, int[] matSizes,
                         float[][] embeddingsArray, int[] embeddingsSize) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("batch_size", batchSize);
    params.put("index", batch.getRowIndices());
    params.put("feats", batch.getColIndices());
    params.put("values", batch.getValues());
    params.put("mats", mats);
    params.put("mats_sizes", matSizes);
    params.put("embeddings_array", embeddingsArray);
    params.put("embeddings_sizes", embeddingsSize);
    return Torch.forward(ptr, params, false);
  }

  public float[] forward(int batchSize, CooLongFloatMatrix batch, float[] mats, int[] matSizes,
                         float[][] embeddingsArray, int[] embeddingsSize, int multiForwardOut) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("batch_size", batchSize);
    params.put("index", batch.getRowIndices());
    params.put("feats", batch.getColIndices());
    params.put("values", batch.getValues());
    params.put("mats", mats);
    params.put("mats_sizes", matSizes);
    params.put("embeddings_array", embeddingsArray);
    params.put("embeddings_sizes", embeddingsSize);
    params.put("multi_forward_out", multiForwardOut);
    return Torch.forward(ptr, params, false);
  }

  public float[] forward(int batchSize, CooLongFloatMatrix batch, float[] bias, float[] weights,
                         float[] embeddings, int embeddingDim) {
    Map<String, Object> params = buildParams(batchSize, batch, bias, weights);
    params.put("embedding", embeddings);
    params.put("embedding_dim", embeddingDim);
    return Torch.forward(ptr, params, false);
  }

  public float[] forward(int batchSize, CooLongFloatMatrix batch, float[] bias, float[] weights,
                         float[] embeddings, int embeddingDim, float[] mats, int[] matSizes) {
    Map<String, Object> params = buildParams(batchSize, batch, bias, weights);
    params.put("embedding", embeddings);
    params.put("embedding_dim", embeddingDim);
    params.put("mats", mats);
    params.put("mats_sizes", matSizes);
    return Torch.forward(ptr, params, false);
  }

  public float[] forward(int batchSize, CooLongFloatMatrix batch, float[] bias, float[] weights,
                         float[] embeddings, int embeddingDim, float[] mats, int[] matSizes, long[] fields) {
    Map<String, Object> params = buildParams(batchSize, batch, bias, weights);
    params.put("embedding", embeddings);
    params.put("embedding_dim", embeddingDim);
    params.put("mats", mats);
    params.put("mats_sizes", matSizes);
    params.put("fields", fields);
    return Torch.forward(ptr, params, false);
  }

  public float backward(int batchSize, CooLongFloatMatrix batch, float[] bias, float[] weights,
                        float[] targets) {
    Map<String, Object> params = buildParams(batchSize, batch, bias, weights);
    params.put("targets", targets);
    return Torch.backward(ptr, params);
  }

  public float backward(int batchSize, CooLongFloatMatrix batch, float[] mats, int[] matSizes,
                        float[][] embeddingsArray, int[] embeddingsSize, float[] targets) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("batch_size", batchSize);
    params.put("index", batch.getRowIndices());
    params.put("feats", batch.getColIndices());
    params.put("values", batch.getValues());
    params.put("mats", mats);
    params.put("mats_sizes", matSizes);
    params.put("embeddings_array", embeddingsArray);
    params.put("embeddings_sizes", embeddingsSize);
    params.put("targets", targets);
    return Torch.backward(ptr, params);
  }

  public float backward(int batchSize, CooLongFloatMatrix batch, float[] bias, float[] weights,
                        float[] embeddings, int embeddingDim, float[] targets) {
    Map<String, Object> params = buildParams(batchSize, batch, bias, weights);
    params.put("embedding", embeddings);
    params.put("embedding_dim", embeddingDim);
    params.put("targets", targets);
    return Torch.backward(ptr, params);
  }

  public float backward(int batchSize, CooLongFloatMatrix batch, float[] bias, float[] weights,
                        float[] embeddings, int embeddingDim, float[] mats, int[] matSizes, float[] targets) {
    Map<String, Object> params = buildParams(batchSize, batch, bias, weights);
    params.put("embedding", embeddings);
    params.put("embedding_dim", embeddingDim);
    params.put("mats", mats);
    params.put("mats_sizes", matSizes);
    params.put("targets", targets);
    return Torch.backward(ptr, params);
  }

  public float backward(int batchSize, CooLongFloatMatrix batch, float[] bias, float[] weights,
                        float[] embeddings, int embeddingDim, float[] mats, int[] matSizes, long[] fields,
                        float[] targets) {
    Map<String, Object> params = buildParams(batchSize, batch, bias, weights);
    params.put("embedding", embeddings);
    params.put("embedding_dim", embeddingDim);
    params.put("mats", mats);
    params.put("mats_sizes", matSizes);
    params.put("fields", fields);
    params.put("targets", targets);
    return Torch.backward(ptr, params);
  }

  public float[] dssmForward(int batchSize, long[] index, float[] values, float[] mats, int[] matSizes,
                             float[][] embeddingsArray, int[] embeddingsSize, int[] inputsSizes) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("batch_size", batchSize);
    params.put("index", index);
    params.put("values", values);
    params.put("mats", mats);
    params.put("mats_sizes", matSizes);
    params.put("embeddings_array", embeddingsArray);
    params.put("embeddings_sizes", embeddingsSize);
    params.put("inputs_sizes", inputsSizes);
    return Torch.dssmForward(ptr, params);
  }

  public float dssmBackward(int batchSize, long[] index, float[] values, float[] mats, int[] matSizes,
                            float[][] embeddingsArray, int[] embeddingsSize, int[] inputsSizes, float[] targets) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("batch_size", batchSize);
    params.put("index", index);
    params.put("values", values);
    params.put("mats", mats);
    params.put("mats_sizes", matSizes);
    params.put("embeddings_array", embeddingsArray);
    params.put("embeddings_sizes", embeddingsSize);
    params.put("inputs_sizes", inputsSizes);
    params.put("targets", targets);
    return Torch.dssmBackward(ptr, params);
  }

  /* forward/backward/predict for gcn models */

  public float[] gcnForward(int batchSize, float[] x, int featureDim, long[] firstEdgeIndex,
                            long[] secondEdgeIndex, float[] weights) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("batch_size", batchSize);
    params.put("x", x);
    params.put("feature_dim", featureDim);
    params.put("first_edge_index", firstEdgeIndex);
    params.put("second_edge_index", secondEdgeIndex);
    params.put("weights", weights);
    return (float[]) Torch.gcnExecMethod(ptr, "forward_", params);
  }

  public float[] gcnForward(Map<String, Object> params) {
    return (float[]) Torch.gcnExecMethod(ptr, "forward_", params);
  }

  public float[] gcnPred(Map<String, Object> params) {
    return (float[]) Torch.gcnExecMethod(ptr, "embedding_predict_", params);
  }

  public long[] gcnPredict(int batchSize, float[] x, int featureDim, long[] firstEdgeIndex,
                           long[] secondEdgeIndex, float[] weights) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("batch_size", batchSize);
    params.put("x", x);
    params.put("feature_dim", featureDim);
    params.put("first_edge_index", firstEdgeIndex);
    params.put("second_edge_index", secondEdgeIndex);
    params.put("weights", weights);
    return (long[]) Torch.gcnExecMethod(ptr, "predict_", params);
  }

  public Object gcnPredict(Map<String, Object> params) {
    return Torch.gcnExecMethod(ptr, "predict_", params);
  }

  public float gcnBackward(int batchSize, float[] x, int featureDim, long[] firstEdgeIndex,
                           long[] secondEdgeIndex, float[] weights, long[] targets) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("batch_size", batchSize);
    params.put("x", x);
    params.put("feature_dim", featureDim);
    params.put("first_edge_index", firstEdgeIndex);
    params.put("second_edge_index", secondEdgeIndex);
    params.put("weights", weights);
    params.put("targets", targets);
    return Torch.gcnBackward(ptr, params, false);
  }

  public float gcnBackward(Map<String, Object> params, boolean sparse) {
    return Torch.gcnBackward(ptr, params, sparse);
  }

  public float[] gcnEmbedding(int batchSize, float[] x, int featureDim, long[] firstEdgeIndex,
                              long[] secondEdgeIndex, float[] weights) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("batch_size", batchSize);
    params.put("x", x);
    params.put("feature_dim", featureDim);
    params.put("first_edge_index", firstEdgeIndex);
    params.put("weights", weights);
    if (secondEdgeIndex != null) {
      params.put("second_edge_index", secondEdgeIndex);
    }
    return (float[]) Torch.gcnExecMethod(ptr, "embedding_", params);
  }

  public float[] gcnEmbedding(Map<String, Object> params) {
    return (float[]) Torch.gcnExecMethod(ptr, "embedding_", params);
  }

  public float[] gcnBiEmbedding(Map<String, Object> params, String method) {
    return (float[]) Torch.gcnExecMethod(ptr, method, params);
  }

  public float dgiBackward(int batchSize, float[] pos_x, float[] neg_x, int featureDim,
                           long[] firstEdgeIndex, long[] secondEdgeIndex, float[] weights) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("batch_size", batchSize);
    params.put("pos_x", pos_x);
    params.put("neg_x", neg_x);
    params.put("feature_dim", featureDim);
    params.put("first_edge_index", firstEdgeIndex);
    params.put("weights", weights);
    if (secondEdgeIndex != null) {
      params.put("second_edge_index", secondEdgeIndex);
    }
    return Torch.gcnBackward(ptr, params, false);
  }

  public float[] getParameters() {
    return Torch.getParameters(ptr);
  }

  public float[] getMatsParameters() {
    return Torch.getMatsParameters(ptr);
  }

  public void gcnSave(String path, float[] weights) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("path", path);
    params.put("weights", weights);
    Torch.gcnSave(ptr, params);
  }


  @Override
  public String toString() {
    Enumeration.Value type = TorchModelType.withName(getType());
    StringBuilder builder = new StringBuilder();
    builder.append("type: ").append(getType()).append(" ");
    builder.append("name:").append(name()).append(" ");
    if (type == TorchModelType.BIAS_WEIGHT()) {
      builder.append("input_dim: ").append(getInputDim()).append(" ");
    } else if (type == TorchModelType.BIAS_WEIGHT_EMBEDDING()) {
      builder.append("embedding_dim: ").append(getEmbeddingDim()).append(" ");
    } else if (type == TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS()
            || type == TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS_FIELD()) {
      builder.append("mats_dims: ");
      int[] sizes = getMatsSize();
      for (int i = 0; i < sizes.length - 1; i++) {
        builder.append(sizes[i] + ",");
      }
      builder.append(sizes[sizes.length - 1]);
    }
    return builder.toString();
  }

}
