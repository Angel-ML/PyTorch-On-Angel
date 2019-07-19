package com.tencent.angel.pytorch.torch;

import com.tencent.angel.exception.AngelException;
import com.tencent.angel.ml.math2.matrix.CooLongFloatMatrix;
import com.tencent.angel.pytorch.Torch;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

public class TorchModel implements Serializable {

  // load library of torch and torch_angel
  static {
    System.loadLibrary("torch_angel");
  }

  private long ptr = 0;
  private static String path;
  private static TorchModel model = null;

  private TorchModel() {}

  private void init(String path) {
    ptr = Torch.initPtr(path);
  }

  public static TorchModel get() {
    if (model == null) {
      if (path == null)
        throw new AngelException("Please set path before accessing model");
      model = new TorchModel();
      model.init(path);
    }
    return model;
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

  /* forward/backward for gcn models */
  public long[] gcnForward(int batchSize, float[] x, int featureDim, long[] edgeIndex, float[] weights) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("batchSize", batchSize);
    params.put("x", x);
    params.put("feature_dim", featureDim);
    params.put("edge_index", edgeIndex);
    params.put("weights", weights);
    return Torch.gcnForward(ptr, params);
  }

  public long[] gcnForward(int batchSize, float[] x, int featureDim, long[] firstEdgeIndex, long[] secondEdgeIndex, float[] weights) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("batchSize", batchSize);
    params.put("x", x);
    params.put("feature_dim", featureDim);
    params.put("first_edge_index", firstEdgeIndex);
    params.put("second_edge_index", secondEdgeIndex);
    params.put("weights", weights);
    return Torch.gcnForward(ptr, params);
  }

  public float gcnBackward(int batchSize, float[] x, int featureDim, long[] edgeIndex, float[] weights, long[] targets) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("batchSize", batchSize);
    params.put("x", x);
    params.put("feature_dim", featureDim);
    params.put("edge_index", edgeIndex);
    params.put("weights", weights);
    params.put("targets", targets);
    return Torch.gcnBackward(ptr, params);
  }

  public float gcnBackward(int batchSize, float[] x, int featureDim, long[] firstEdgeIndex, long[] secondEdgeIndex, float[] weights, long[] targets) {
    Map<String, Object> params = new HashMap<String, Object>();
    params.put("batchSize", batchSize);
    params.put("x", x);
    params.put("feature_dim", featureDim);
    params.put("first_edge_index", firstEdgeIndex);
    params.put("second_edge_index", secondEdgeIndex);
    params.put("weights", weights);
    params.put("targets", targets);
    return Torch.gcnBackward(ptr, params);
  }

  public float[] getParameters() {
    return Torch.getParameters(ptr);
  }

}
