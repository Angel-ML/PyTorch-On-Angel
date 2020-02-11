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
package com.tencent.angel.pytorch.data;

import com.tencent.angel.exception.AngelException;
import com.tencent.angel.ml.math2.MFactory;
import com.tencent.angel.ml.math2.VFactory;
import com.tencent.angel.ml.math2.matrix.CooLongFloatMatrix;
import com.tencent.angel.ml.math2.vector.IntFloatVector;
import com.tencent.angel.pytorch.model.TorchModelType;
import it.unimi.dsi.fastutil.floats.FloatArrayList;
import it.unimi.dsi.fastutil.longs.LongArrayList;
import scala.Tuple2;
import scala.Tuple3;

public class SampleParser {

  public static Tuple3<CooLongFloatMatrix, long[], float[]> parse(String[] lines, String type) {
    if (TorchModelType.withName(type) == TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS_FIELD())
      return parseLIBFFM(lines);
    else {
      Tuple2<CooLongFloatMatrix, float[]> tuple2 = parseLIBSVM(lines);
      return new Tuple3<CooLongFloatMatrix, long[], float[]>(tuple2._1, null, tuple2._2);
    }
  }

  public static Tuple3<CooLongFloatMatrix, long[], String[]> parsePredict(String[] lines, String type) {
    if (TorchModelType.withName(type) == TorchModelType.BIAS_WEIGHT_EMBEDDING_MATS_FIELD())
      return parseLIBFFMPredict(lines);
    else {
      Tuple2<CooLongFloatMatrix, String[]> tuple2 = parseLIBSVMPredict(lines);
      return new Tuple3<CooLongFloatMatrix, long[], String[]>(tuple2._1, null, tuple2._2);
    }
  }

  private static Tuple2<CooLongFloatMatrix, float[]> parseLIBSVM(String[] lines) {
    LongArrayList rows = new LongArrayList();
    LongArrayList cols = new LongArrayList();
    LongArrayList fields = null;
    FloatArrayList vals = new FloatArrayList();
    float[] targets = new float[lines.length];

    int index = 0;
    for (int i = 0; i < lines.length; i++) {
      String[] parts = lines[i].split(" ");
      float label = Float.parseFloat(parts[0]);
      targets[i] = label;

      for (int j = 1; j < parts.length; j++) {
        String[] kv = parts[j].split(":");
        long key = Long.parseLong(kv[0]) - 1;
        float val = Float.parseFloat(kv[1]);

        rows.add(index);
        cols.add(key);
        vals.add(val);
      }

      index++;
    }

    CooLongFloatMatrix coo = MFactory.cooLongFloatMatrix(rows.toLongArray(),
            cols.toLongArray(), vals.toFloatArray(), null);

    return new Tuple2<CooLongFloatMatrix, float[]>(coo, targets);
  }

  private static Tuple3<CooLongFloatMatrix, long[], float[]> parseLIBFFM(String[] lines) {
    LongArrayList rows = new LongArrayList();
    LongArrayList cols = new LongArrayList();
    LongArrayList fields = new LongArrayList();
    FloatArrayList vals = new FloatArrayList();
    float[] targets = new float[lines.length];

    int index = 0;
    for (int i = 0; i < lines.length; i++) {
      String[] parts = lines[i].split(" ");
      float label = Float.parseFloat(parts[0]);
      targets[i] = label;

      for (int j = 1; j < parts.length; j++) {
        String[] fkv = parts[j].split(":");
        long field = Long.parseLong(fkv[0]);
        long key = Long.parseLong(fkv[1]) - 1;
        float val = Float.parseFloat(fkv[2]);

        rows.add(index);
        fields.add(field);
        cols.add(key);
        vals.add(val);
      }

      index++;
    }

    CooLongFloatMatrix coo = MFactory.cooLongFloatMatrix(rows.toLongArray(),
            cols.toLongArray(), vals.toFloatArray(), null);

    return new Tuple3<CooLongFloatMatrix, long[], float[]>(coo, fields.toLongArray(), targets);
  }

  private static Tuple2<CooLongFloatMatrix, String[]> parseLIBSVMPredict(String[] lines) {
    LongArrayList rows = new LongArrayList();
    LongArrayList cols = new LongArrayList();
    FloatArrayList vals = new FloatArrayList();
    String[] targets = new String[lines.length];

    int index = 0;
    for (int i = 0; i < lines.length; i++) {
      String[] parts = lines[i].split(" ");
      targets[i] = parts[0];

      for (int j = 1; j < parts.length; j++) {
        String[] kv = parts[j].split(":");
        long key = Long.parseLong(kv[0]) - 1;
        float val = Float.parseFloat(kv[1]);

        rows.add(index);
        cols.add(key);
        vals.add(val);
      }

      index++;
    }

    CooLongFloatMatrix coo = MFactory.cooLongFloatMatrix(rows.toLongArray(),
            cols.toLongArray(), vals.toFloatArray(), null);

    return new Tuple2<CooLongFloatMatrix, String[]>(coo, targets);
  }

  private static Tuple3<CooLongFloatMatrix, long[], String[]> parseLIBFFMPredict(String[] lines) {
    LongArrayList rows = new LongArrayList();
    LongArrayList cols = new LongArrayList();
    LongArrayList fields = new LongArrayList();
    FloatArrayList vals = new FloatArrayList();
    String[] targets = new String[lines.length];

    int index = 0;
    for (int i = 0; i < lines.length; i++) {
      String[] parts = lines[i].split(" ");
      targets[i] = parts[0];

      for (int j = 1; j < parts.length; j++) {
        String[] fkv = parts[j].split(":");
        long field = Long.parseLong(fkv[0]);
        long key = Long.parseLong(fkv[1]) - 1;
        float val = Float.parseFloat(fkv[2]);

        rows.add(index);
        fields.add(field);
        cols.add(key);
        vals.add(val);
      }

      index++;
    }

    CooLongFloatMatrix coo = MFactory.cooLongFloatMatrix(rows.toLongArray(),
            cols.toLongArray(), vals.toFloatArray(), null);

    return new Tuple3<CooLongFloatMatrix, long[], String[]>(coo, fields.toLongArray(), targets);
  }

  public static Tuple2<Long, IntFloatVector> parseNodeFeature(String line, int dim, String format) {
    if (format.equals("sparse"))
      return parseSparseNodeFeature(line, dim);
    if (format.equals("dense"))
      return parseDenseNodeFeature(line, dim);
    throw new AngelException("data format should be sparse or dense");
  }

  public static Tuple2<Long, IntFloatVector> parseSparseNodeFeature(String line, int dim) {
    if (line.length() == 0)
      return null;

    String[] parts = line.split(" ");
    if (parts.length < 2)
      return null;

    long node = Long.parseLong(parts[0]);
    int[] keys = new int[parts.length - 1];
    float[] vals = new float[parts.length - 1];
    for (int i = 1; i < parts.length; i++) {
      String[] kv = parts[i].split(":");
      keys[i - 1] = Integer.parseInt(kv[0]);
      vals[i - 1] = Float.parseFloat(kv[1]);
    }
    IntFloatVector feature = VFactory.sortedFloatVector(dim, keys, vals);
    return new Tuple2<Long, IntFloatVector>(node, feature);
  }

  public static Tuple2<Long, IntFloatVector> parseDenseNodeFeature(String line, int dim) {
    if (line.length() == 0)
      return null;

    String[] parts = line.split(" ");
    if (parts.length != dim + 1)
      throw new AngelException("number elements of data should be equal dim");

    long node = Long.parseLong(parts[0]);
    float[] values = new float[parts.length - 1];
    for (int i = 1; i < parts.length; i++)
      values[i - 1] = Float.parseFloat(parts[i]);

    IntFloatVector feature = VFactory.denseFloatVector(values);
    return new Tuple2<Long, IntFloatVector>(node, feature);
  }

  public static IntFloatVector parseFeature(String line, int dim, String format) {
    if (format.equals("sparse"))
      return parseSparseIntFloat(line, dim);
    if (format.equals("dense"))
      return parseDenseIntFloat(line, dim);

    throw new AngelException("format should be sparse or dense");
  }

  public static IntFloatVector parseSparseIntFloat(String line, int dim) {
    String[] parts = line.split(" ");

    int[] keys = new int[parts.length];
    float[] vals = new float[parts.length];
    for (int i = 0; i < parts.length; i++) {
      String[] kv = parts[i].split(":");
      keys[i] = Integer.parseInt(kv[0]);
      if (keys[i] >= dim)
        throw new AngelException("feature index should be less than dim");
      vals[i] = Float.parseFloat(kv[1]);
    }

    return VFactory.sortedFloatVector(dim, keys, vals);
  }

  public static IntFloatVector parseDenseIntFloat(String line, int dim) {
    String[] parts = line.split(" ");
    if (parts.length != dim)
      throw new AngelException("number elements " + parts.length + " should be equal with dim " + dim);

    float[] vals = new float[parts.length];
    for (int i = 0; i < parts.length; i++)
      vals[i] = Float.parseFloat(parts[i]);

    return VFactory.denseFloatVector(vals);
  }
}
