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
import org.apache.spark.SparkEnv;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;

public class TorchModel implements Serializable {

  // load library of torch and torch_angel
  static {
    System.loadLibrary("torch_angel");
  }

  private long ptr = 0;
  private static String path;
  private static BlockingQueue<TorchModel> modelsQueue = new LinkedBlockingQueue<TorchModel>();
  private static AtomicInteger counter = new AtomicInteger(0);
  private static int cores = SparkEnv.get().conf()
    .getInt("spark.executor.cores", 1);

  private TorchModel() {}

  private void init(String path) {
    ptr = Torch.initPtr(path);
  }

  public static synchronized TorchModel get() throws InterruptedException {
    if (modelsQueue.isEmpty()) {
      if (path == null)
        throw new AngelException("Please set path before accessing model");
      TorchModel model = new TorchModel();
      model.init(path);
      modelsQueue.put(model);
      counter.addAndGet(1);
      if (counter.get() > cores + 2)
        throw new AngelException("The size of torch model exceeds the cores");
    }
    return modelsQueue.take();
  }

  public static void addModel(TorchModel torchModel) throws InterruptedException {
    TorchModel.modelsQueue.put(torchModel);
  }

  public static void setPath(String path) {
    TorchModel.path = path;
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

  public int getNumFields() {
      return Torch.getNumFields(ptr);
  }

  public int getEmbeddingDim() {
      return Torch.getEmbeddingDim(ptr);
  }

  public int getParametersTotalSize() {
    return Torch.getParametersTotalSize(ptr);
  }

  public void setParameters(float[] values) {
    Torch.setParameters(ptr, values);
  }

  public void save(float[] bias, float[] weights) {
    Map<String, Object> params = buildParams(bias, weights);
    Torch.save(ptr, params);
  }

  public void save(float[] bias, float[] weights,
                   float[] embeddings, int embeddingDim,
                   int dim) {
      Map<String, Object> params = buildParams(bias, weights);
      params.put("embedding", embeddings);
      params.put("embedding_dim", embeddingDim);
      params.put("dim", dim);
      Torch.save(ptr, params);
  }

  public void save(float[] bias, float[] weights,
                   float[] embeddings, int embeddingDim,
                   float[] mats, int[] matSizes, int dim) {
      Map<String, Object> params = buildParams(bias, weights);
      params.put("embedding", embeddings);
      params.put("embedding_dim", embeddingDim);
      params.put("mats", mats);
      params.put("mats_sizes", matSizes);
      params.put("dim", dim);
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

  private Map<String, Object> buildParams(int batchSize, CooLongFloatMatrix batch, float[] bias, float[] weights) {
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

  public float[] forward(int batchSize, CooLongFloatMatrix batch, float[] bias, float[] weights, float[] embeddings, int embeddingDim) {
    Map<String, Object> params = buildParams(batchSize, batch, bias, weights);
    params.put("embedding", embeddings);
    params.put("embedding_dim", embeddingDim);
    return Torch.forward(ptr, params, false);
  }

  public float[] forward(int batchSize, CooLongFloatMatrix batch, float[] bias, float[] weights, float[] embeddings, int embeddingDim, float[] mats, int[] matSizes) {
    Map<String, Object> params = buildParams(batchSize, batch, bias, weights);
    params.put("embedding", embeddings);
    params.put("embedding_dim", embeddingDim);
    params.put("mats", mats);
    params.put("mats_sizes", matSizes);
    return Torch.forward(ptr, params, false);
  }

  public float[] forward(int batchSize, CooLongFloatMatrix batch, float[] bias, float[] weights, float[] embeddings, int embeddingDim, float[] mats, int[] matSizes, long[] fields) {
    Map<String, Object> params = buildParams(batchSize, batch, bias, weights);
    params.put("embedding", embeddings);
    params.put("embedding_dim", embeddingDim);
    params.put("mats", mats);
    params.put("mats_sizes", matSizes);
    params.put("fields", fields);
    return Torch.forward(ptr, params, false);
  }

  public float backward(int batchSize, CooLongFloatMatrix batch, float[] bias, float[] weights, float[] targets) {
    Map<String, Object> params = buildParams(batchSize, batch, bias, weights);
    params.put("targets", targets);
    return Torch.backward(ptr, params);
  }

  public float backward(int batchSize, CooLongFloatMatrix batch, float[] bias, float[] weights, float[] embeddings, int embeddingDim, float[] targets) {
    Map<String, Object> params = buildParams(batchSize, batch, bias, weights);
    params.put("embedding", embeddings);
    params.put("embedding_dim", embeddingDim);
    params.put("targets", targets);
    return Torch.backward(ptr, params);
  }

  public float backward(int batchSize, CooLongFloatMatrix batch, float[] bias, float[] weights, float[] embeddings, int embeddingDim, float[] mats, int[] matSizes, float[] targets) {
    Map<String, Object> params = buildParams(batchSize, batch, bias, weights);
    params.put("embedding", embeddings);
    params.put("embedding_dim", embeddingDim);
    params.put("mats", mats);
    params.put("mats_sizes", matSizes);
    params.put("targets", targets);
    return Torch.backward(ptr, params);
  }

  public float backward(int batchSize, CooLongFloatMatrix batch, float[] bias, float[] weights, float[] embeddings, int embeddingDim, float[] mats, int[] matSizes, long[] fields, float[] targets) {
    Map<String, Object> params = buildParams(batchSize, batch, bias, weights);
    params.put("embedding", embeddings);
    params.put("embedding_dim", embeddingDim);
    params.put("mats", mats);
    params.put("mats_sizes", matSizes);
    params.put("fields", fields);
    params.put("targets", targets);
    return Torch.backward(ptr, params);
  }

  /* forward/backward/predict for gcn models */

  public float[] gcnForward(int batchSize, float[] x, int featureDim, long[] firstEdgeIndex, long[] secondEdgeIndex, float[] weights) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("batch_size", batchSize);
    params.put("x", x);
    params.put("feature_dim", featureDim);
    params.put("first_edge_index", firstEdgeIndex);
    params.put("second_edge_index", secondEdgeIndex);
    params.put("weights", weights);
    return (float[]) Torch.gcnExecMethod(ptr, "forward_", params);
  }

  public long[] gcnPredict(int batchSize, float[] x, int featureDim, long[] firstEdgeIndex, long[] secondEdgeIndex, float[] weights) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("batch_size", batchSize);
    params.put("x", x);
    params.put("feature_dim", featureDim);
    params.put("first_edge_index", firstEdgeIndex);
    params.put("second_edge_index", secondEdgeIndex);
    params.put("weights", weights);
    return (long[]) Torch.gcnExecMethod(ptr, "predict_", params);
  }

  public float gcnBackward(int batchSize, float[] x, int featureDim, long[] firstEdgeIndex, long[] secondEdgeIndex, float[] weights, long[] targets) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("batch_size", batchSize);
    params.put("x", x);
    params.put("feature_dim", featureDim);
    params.put("first_edge_index", firstEdgeIndex);
    params.put("second_edge_index", secondEdgeIndex);
    params.put("weights", weights);
    params.put("targets", targets);
    return Torch.gcnBackward(ptr, params);
  }

  public float[] gcnEmbedding(int batchSize, float[] x, int featureDim, long[] firstEdgeIndex, long[] secondEdgeIndex, float[] weights) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("batch_size", batchSize);
    params.put("x", x);
    params.put("feature_dim", featureDim);
    params.put("first_edge_index", firstEdgeIndex);
    params.put("weights", weights);
    if (secondEdgeIndex != null)
      params.put("second_edge_index", secondEdgeIndex);
    return (float[]) Torch.gcnExecMethod(ptr, "embedding_", params);
  }

  public float dgiBackward(int batchSize, float[] pos_x, float[] neg_x, int featureDim, long[] firstEdgeIndex, long[] secondEdgeIndex, float[] weights) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("batch_size", batchSize);
    params.put("pos_x", pos_x);
    params.put("neg_x", neg_x);
    params.put("feature_dim", featureDim);
    params.put("first_edge_index", firstEdgeIndex);
    params.put("weights", weights);
    if (secondEdgeIndex != null)
      params.put("second_edge_index", secondEdgeIndex);
    return Torch.gcnBackward(ptr, params);
  }

  public float[] getParameters() {
    return Torch.getParameters(ptr);
  }

}
