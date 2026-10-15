## GNN Algorithm

> PyTorch on Angel provides the ability to run graph convolutional network algorithms. We
> define the graph convolutional networks based on [Pytorch-Geometric](https://github.com/rusty1s/pytorch_geometric)
> while using the parameter server of Angel to store the network structure and features of nodes.


### 1 Introduction

#### 1.1 How to predict

- Change the paths of input and output.
- Set `actionType` to 'predict'.
- Download `hdfs://modelPath/xx.pt` to local and use it as the model file, or use the hdfs path directly with
  `--files hdfs://modelPath/xx.pt`, in which case `torchModelPath` can be omitted.

#### 1.2 How to train incrementally

- Change the paths of input and output, or use the same data to train incrementally.
- Set `actionType` to 'train'.
- Download `hdfs://modelPath/xx.pt` to local and use it as the model file, or use the hdfs path directly with
  `--files hdfs://modelPath/xx.pt`, in which case `torchModelPath` can be omitted.

#### 1.3 How to calculate the resource

To determine how to allocate resources, first figure out where the data is saved, then calculate how much
storage space it needs, and finally set the resource size to 2~3 times the storage size. For the detailed calculation
method, refer to [Resource Calculation Method](./resource_calculation_method.md).

### 2 Algorithm Examples

In this part, most of the algorithms support two usage modes, `script-mode` and `yaml-mode`, where the properties
are configured in the submit script and in a `.yaml` file respectively.

The following configuration is shared by all the submit scripts below and is denoted as `<COMMON_SCRIP_PART>`:

   ```bash
source ./spark-on-angel-env.sh  # see bin/spark-on-angel-env.sh
$SPARK_HOME/bin/spark-submit \
--master yarn-cluster\
--conf spark.ps.instances=5 \
--conf spark.ps.cores=1 \
--conf spark.ps.jars=$SONA_ANGEL_JARS \
--conf spark.ps.memory=5g \
--conf spark.ps.log.level=INFO \
--conf spark.driver.extraJavaOptions=-Djava.library.path=$JAVA_LIBRARY_PATH:.:./torch/angel_libtorch \
--conf spark.executor.extraJavaOptions=-Djava.library.path=$JAVA_LIBRARY_PATH:.:./torch/angel_libtorch \
--conf spark.executor.extraLibraryPath=./torch/angel_libtorch \
--conf spark.driver.extraLibraryPath=./torch/angel_libtorch \
--conf spark.executorEnv.OMP_NUM_THREADS=2 \
--conf spark.executorEnv.MKL_NUM_THREADS=2 \
--jars $SONA_SPARK_JARS  \
--archives hdfs://path/of/torch.zip#torch \  # c++ library files
--queue $queue \
--name $algorithmName \
--driver-memory 5g \
--num-executors 5 \
--executor-cores 1 \
--executor-memory 5g \
   ```

#### 2.1 GraphSage

[GraphSage](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) generates embeddings by sampling and
aggregating features from a node’s local neighborhood. Here we give an example of how to run GraphSage on PyTorch on Angel.

**1. script-mode**

**(1) Generate pytorch script model**

Run the following command in the `python/graph/homogeneous/supervised` directory.

  - dense/low-sparse

  ```bash
  python graphsage.py --input_dim 1433 --hidden_dim 128 --output_dim 7 --output_file graphsage_cora.pt
  ```

  - high-sparse

  ```bash
  python graphsage.py --input_dim 32 --input_embedding_dim 8 --input_field_num 20 --encode one-hot --hidden_dim 128 --output_dim 7 --output_file graphsage_sparse.pt
  ```

This script uses [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of GraphSage. After that, you will obtain a model file named "graphsage_cora.pt". Here we use the Cora dataset as an example, where the feature dimension for each node is 1433 with 7 different classes. For a detailed parameter introduction, see [Python Model Parameters](./public_parameters_gnn.md).

**(2) Preparing input data**

  GraphSage requires three inputs: the edge table, the node feature table, and the node label table. For details, see [Data Format](./data_format_gnn.md).

**(3) Submit model to cluster**

  Once the model file and inputs are ready, submit the task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

  - dense/low-sparse

  ```bash
  <COMMON_SCRIP_PART>
  --files graphsage_cora.pt \   #path to the TorchScript model
  --class com.tencent.angel.pytorch.examples.supervised.cluster.GraphSageExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \   # jar from Compiling the examples submodule
  edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath\
  torchModelPath:graphsage_cora.pt featureDim:1433 stepSize:0.01\
  optimizer:adam numEpoch:10 testRatio:0.5\
  numPartitions:50 format:sparse samples:10 batchSize:128\
  predictOutputPath:$predictOutputPath embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5
  ```

  - high-sparse

  ```bash
  <COMMON_SCRIP_PART>
  --files graphsage_cora.pt \   #path to the TorchScript model
  --class com.tencent.angel.pytorch.examples.supervised.cluster.GraphSageExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \   # jar from Compiling the examples submodule
  edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath\
  torchModelPath:graphsage_sparse.pt featureDim:32 stepSize:0.01\
  optimizer:adam numEpoch:10 testRatio:0.5 fieldNum:20 featEmbedDim:8 \
  numPartitions:50 format:sparse samples:10 batchSize:128\
  predictOutputPath:$predictOutputPath embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5
  ```

  For a full list of parameters and output results, see [details](./public_parameters_gnn.md).

  **Notes:**

   - The model file, graphsage_cora.pt, should be uploaded to the Spark driver and every executor, so use `--files` to upload it.

**2. yaml-mode**

- demo.yaml

  ```yaml
  graph:
    name: "myGraph"
    nodes:
      - name: "user"
        feature:
          format: "dense"
          path: $userPath
        label:
          path: $labelPath
          validate_path: ""
    edges:
      base_config:
        weighted: false
        sep: "space"
      link_config:
        - name: "u2u"
          src: "user"
          dst: "user"
          path: $edgePath
  model:
    model_name: "graphsage"
    input_dim: 602
    hidden_dim: 128
    output_dim: 41
    second: true
    encode: "dense"
  trainer:
    epoch: 10
    batch_size: 128
    learning_rate: 0.01
    decay: 0.01
    test_ratio: 0.5
    optimizer: "adam"
    sample_num: 5
    batch_init_num: 5
    eval_metrics: "acc"
    sample_method: "random"
    validate_periods: 1
    periods: 10
    save_checkpoint: false
    use_shared_samples: false
    model_save_path: $savePath
  predictor:
    batch_size_multiplier: 5
    node_embedding_output_path: $embeddingPath
    predict_output_path: $predictOutputPath
  ```

- script

  ```bash
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.homogeneous.supervised.GraphSageExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \
  psPartitionNum:50 dataPartitionNum:100 psPartitionNumFactor:2 dataPartitionNumFactor:3 \
  storageLevel:MEMORY_AND_DISK actionType:train
  ```

#### 2.2 DGI/Unsupervised GraphSage

Here we give an example of how to run DGI on PyTorch on Angel.

**1. script-mode**

**(1) Generate pytorch script model**

  Run the following command in the `python/graph/homogeneous/unsupervised` directory.

  - DGI pt model

  ```bash
  python dgi.py --input_dim 1433 --hidden_dim 128 --output_dim 128 --output_file dgi_cora.pt
  ```

  - Unsupervised GraphSage pt model

  ```bash
  python unsupervised_graphsage.py --input_dim 1433 --hidden_dim 128 --output_dim 128 --output_file unsupervised_graphsage_cora.pt
  ```

  This script uses [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of dgi. After that, you will obtain a model file named "dgi_cora.pt". Here we use the Cora dataset
  as an example, where the feature dimension for each node is 1433. 
  For a detailed parameter introduction, see [Python Model Parameters](./public_parameters_gnn.md).

**(2) Preparing input data**

  DGI requires two inputs: the edge table and the node feature table.
  For details, see [Data Format](./data_format_gnn.md).

**(3) Submit model to cluster**

  Once the model file and inputs are ready, submit the task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).
  The only difference between **DGI** and Unsupervised **GraphSage** is the .pt model; the submit scripts are the same.

  ```bash
  <COMMON_SCRIP_PART>
  --files dgi_cora.pt \ #path to the TorchScript model
  --class com.tencent.angel.pytorch.examples.unsupervised.cluster.DGIExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \ # jar from Compiling the examples submodule
  edgePath:$edgePath featurePath:$featurePath\
  torchModelPath:dgi_cora.pt featureDim:1433 stepSize:0.01\
  optimizer:adam numEpoch:10 \
  numPartitions:50 format:sparse samples:10 batchSize:128\
  embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5
  ```

  For a full list of parameters and output results, see [details](./public_parameters_gnn.md).

**Notes:**

- The model file, dgi_cora.pt, should be uploaded to the Spark driver and every executor, so use `--files` to upload it.

**2. yaml-mode**

- demo.yaml

  ```yaml
  graph:
    name: "myGraph"
    nodes:
      - name: "user"
        feature:
          format: "dense"
          path: $userPath
        label:
          path: $labelPath
          validate_path: ""
    edges:
      base_config:
        weighted: false
        sep: "space"
      link_config:
        - name: "u2u"
          src: "user"
          dst: "user"
          path: $edgePath
  model:
    model_name: "dgi"
    input_dim: 1433
    hidden_dim: 128
    output_dim: 128
    neg_sampling: true
    second: true # only support second-order
  trainer:
    epoch: 10
    batch_size: 128
    learning_rate: 0.01
    decay: 0.01
    train_sample_ratio: 0.8
    optimizer: "adam"
    sample_num: 10
    batch_init_num: 5
    sample_method: "random"
    periods: 10
    model_save_path: $savePath
  predictor:
    batch_size_multiplier: 5
    node_embedding_output_path: $embeddingPath
  ```

- script

  ```bash
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.homogeneous.unsupervised.DGIExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \
  actionType:train \
  dataPartitionNum:10 \
  psNumPartition:10 \
  storageLevel:MEMORY_ONLY \
  ```


#### 2.3 Relation GCN (RGCN)

Relation GCN is a semi-supervised graph convolutional network which can utilize the types of edges. The difference between
RGCN and GCN is that each edge can have different types.

Here we give an example of using RGCN on PyTorch on Angel.

**1. script-mode**

**(1) Generate pytorch script model**

  Run the following command in the `python/graph/homogeneous/supervised` directory.

  ```bash
  python rgcn.py --input_dim 32 --hidden_dim 16 --n_class 2 --output_file rgcn_mutag.pt --n_relations 46 --n_bases 30
  ```

  This script uses [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of rgcn. After that, you will obtain a model file named "rgcn_mutag.pt". Where n_class is the
  number of classes, n_relations is the number of types for edges and n_bases is a parameter of RGCN to avoid
  overfitting.
  For a detailed parameter introduction, see [Python Model Parameters](./public_parameters_gnn.md).

**(2) Preparing input data**

  RGCN requires three inputs: the edge table with type, the node feature table, and the node label table.

  For details, see [Data Format](./data_format_gnn.md).

**(3) Submit model to cluster**

  Once the model file and inputs are ready, submit the task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

  ```bash
  <COMMON_SCRIP_PART>
  --files rgcn_mutag.pt \   #path to the TorchScript model
  --class com.tencent.angel.pytorch.examples.supervised.cluster.RGCNExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \
  edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath\
  torchModelPath:rgcn_mutag.pt featureDim:32 stepSize:0.01\
  optimizer:adam numEpoch:10 testRatio:0.5\
  numPartitions:50 format:sparse samples:10 batchSize:128\
  predictOutputPath:$predictOutputPath embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5
  ```

  For a full list of parameters and output results, see [details](./public_parameters_gnn.md).

  - edgePath: the input path (hdfs) of edge table, which contains src, dst and type

**Notes:**

- The model file, rgcn_mutag.pt, should be uploaded to the Spark driver and every executor, so use `--files` to upload it.

**2. yaml-mode**

- demo.yaml

  ```yaml
  graph:
    name: "myGraph"
    nodes:
      - name: "user"
        feature:
          format: "dense"
          path: $userPath
        label:
          path: $labelPath
          validate_path: ""
    edges:
      base_config:
        weighted: false
        sep: "space"
      link_config:
        - name: "u2u"
          src: "user"
          dst: "user"
          path: $edgePath
  model:
    model_name: "rgcn"
    input_dim: 32
    hidden_dim: 128
    n_relations: 267
    output_dim: 11
    second: true
    encode: "dense"
  trainer:
    epoch: 10
    batch_size: 128
    learning_rate: 0.001
    decay: 0.01
    test_ratio: 0.5
    optimizer: "adam"
    sample_num: 5
    batch_init_num: 5
    eval_metrics: "acc"
    sample_method: "random"
    validate_periods: 1
    periods: 10
    model_save_path: $savePath
  predictor:
    batch_size_multiplier: 5
    node_embedding_output_path: $embeddingPath
    predict_output_path: $predictOutputPath
  ```

- script

  ```bash
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.homogeneous.supervised.RGCNExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \
  actionType:train \
  dataPartitionNum:10 \
  psNumPartition:10 \
  storageLevel:MEMORY_ONLY \
  ```


#### 2.4 EdgeProp

[EdgeProp](https://arxiv.org/abs/1906.05546) is an end-to-end Graph Convolution Network (GCN)-based algorithm to learn
the embeddings of the nodes and edges of a large-scale time-evolving graph. It considers not only node information but also edge-side information.

Here we give an example of using EdgeProp on PyTorch on Angel.

**1. script-mode**

**(1) Generate pytorch script model**

  Run the following command in the `python/graph` directory:

  ```bash
  python edgeprop.py --input_dim 23 --input_edge_dim 7 --hidden_dim 128 --output_dim 7 --output_file edgeprop_eth.pt
  ```

  This script uses [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of edgeProp. After that, you will obtain a model file named "edgeprop_eth.pt". Where
  edge_input_dim is the dimension of edge feature, other parameters are same as GraphSage.
  For a detailed parameter introduction, see [Python Model Parameters](./public_parameters_gnn.md).

**(2) Preparing input data**

  EdgeProp requires three inputs: the edge table with edge feature, the node feature table, and the node label table.

  For details, see [Data Format](./data_format_gnn.md).

**(3) Submit model to cluster**

  Once the model file and inputs are ready, submit the task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

  ```bash
  <COMMON_SCRIP_PART>
  --files edgeprop_eth.pt \   #path to the TorchScript model
  --class com.tencent.angel.pytorch.examples.supervised.cluster.EdgePropGCNExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \   # jar from Compiling the examples submodule
  edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath\
  torchModelPath:edgeprop_eth.pt featureDim:23 edgeFeatureDim:7 stepSize:0.01\
  optimizer:adam numEpoch:10 testRatio:0.5\
  numPartitions:50 format:sparse samples:10 batchSize:128\
  predictOutputPath:$predictOutputPath embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5
  ```

  Here is a short description of the parameters in the submit script:

  - edgePath: the input path (hdfs) of edge table, which contains src, dst and edge feature

**Notes:**

- The model file, edgeprop_eth.pt, should be uploaded to the Spark driver and every executor, so use `--files` to upload it.

**2. yaml-mode**

- demo.yaml

  ```yaml
  graph:
    name: "myGraph"
    nodes:
      - name: "user"
        feature:
          format: "dense"
          path: $userPath
        label:
          path: $labelPath
          validate_path: ""
    edges:
      base_config:
        weighted: false
        sep: "space"
      link_config:
        - name: "u2u"
          src: "user"
          dst: "user"
          path: $edgePath
  model:
    model_name: "edgeprop"
    input_dim: 23
    input_edge_dim: 7
    hidden_dim: 128
    output_dim: 7
    second: true
    encode: "dense"
  trainer:
    epoch: 10
    batch_size: 128
    learning_rate: 0.01
    decay: 0.01
    test_ratio: 0.5
    optimizer: "adam"
    sample_num: 5
    batch_init_num: 5
    eval_metrics: "acc"
    sample_method: "random"
    validate_periods: 1
    periods: 10
    model_save_path: $savePath
  predictor:
    batch_size_multiplier: 5
    node_embedding_output_path: $embeddingPath
  ```

- script

  ```bash
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.homogeneous.supervised.EdgePropGCNExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \
  actionType:train \
  dataPartitionNum:10 \
  psNumPartition:10 \
  storageLevel:MEMORY_ONLY \
  ```

#### 2.5 GAT

Here we give an example of how to run GAT on PyTorch on Angel.

**1. script-mode**

**(1) Generate pytorch script model**

  Run the following command in the `python/graph/homogeneous/supervised` directory:

  ```bash
  python gat.py --input_dim 32 --hidden_dim 128 --output_dim 11 --output_file gat_am.pt
  ```

  This script uses [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of GAT. After that, you will obtain a model file named "gat_am.pt". Here we use the AM dataset as an example, where the feature dimension for each node is 32 with 11 different classes.
  For a detailed parameter introduction, see [Python Model Parameters](./public_parameters_gnn.md).

**(2) Preparing input data**

  GAT requires three inputs: the edge table, the node feature table, and the node label table.

  For details, see [Data Format](./data_format_gnn.md).

**(3) Submit model to cluster**

  Once the model file and inputs are ready, submit the task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

  ```bash
  <COMMON_SCRIP_PART>
  --files gat_am.pt \   #path to the TorchScript model
  --class com.tencent.angel.pytorch.examples.supervised.cluster.GATExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \   # jar from Compiling the examples submodule
  edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath\
  torchModelPath:gat_am.pt featureDim:32 stepSize:0.01\
  optimizer:adam numEpoch:10 testRatio:0.5\
  numPartitions:50 format:sparse samples:10 batchSize:128\
  predictOutputPath:$predictOutputPath embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5
  ```

  For a full list of parameters and output results, see [details](./public_parameters_gnn.md).

**Notes:**

- The model file, gat_am.pt, should be uploaded to the Spark driver and every executor, so use `--files` to upload it.

**2. yaml-mode**

- demo.yaml

  ```yaml
  graph:
    name: "myGraph"
    nodes:
      - name: "user"
        feature:
          format: "dense"
          path: $userPath
        label:
          path: $labelPath
          validate_path: ""
    edges:
      base_config:
        weighted: false
        sep: "space"
      link_config:
        - name: "u2u"
          src: "user"
          dst: "user"
          path: $edgePath
  model:
    model_name: "gat"
    input_dim: 602
    hidden_dim: 128
    output_dim: 42
    second: true
    encode: "dense"
  trainer:
    epoch: 10
    batch_size: 128
    learning_rate: 0.01
    decay: 0.01
    test_ratio: 0.5
    optimizer: "adam"
    sample_num: 5
    batch_init_num: 5
    eval_metrics: "acc"
    sample_method: "random"
    validate_periods: 1
    periods: 10
    save_checkpoint: false
    use_shared_samples: false
    model_save_path: $savePath
  predictor:
    batch_size_multiplier: 5
    node_embedding_output_path: $embeddingPath
  ```

- script

  ```bash
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.homogeneous.supervised.GATExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \
  actionType:train \
  dataPartitionNum:300 \
  psNumPartition:100 \
  storageLevel:MEMORY_ONLY \
  ```


#### 2.6 HAN

[HAN](https://arxiv.org/pdf/1903.07293.pdf) is a semi-supervised graph convolutional network for heterogeneous graphs. In
order to capture the heterogeneous information, HAN defined two different attentions: node-level and semantic level.
Here a simplified version of HAN is implemented, which accepts bipartite graph in the form of "user-item", where item
nodes could have multiple types. In other words, the input graph has multiple meta-paths in the form of "
user-item-user".
HAN classifies user nodes, and outputs their embeddings if needed.

Here we give an example of using HAN on PyTorch on Angel.

**1. script-mode**

**(1) Generate pytorch script model**

  Run the following command in the `python/graph/bipartite/supervised` directory: 

  - dense/low-sparse

  ```bash
  python han.py --m 64 --input_dim 32 --hidden_dim 16 --output_dim 2 --item_types 5  --output_file han.pt
  ```

  - high-sparse

   ```bash
   python han.py --m 64 --input_dim 32 --input_embedding_dim 8 --input_field_num 20 --encode one-hot --hidden_dim 16 --output_dim 2 --item_types 5 --output_file han_sparse.pt
   ```

  This script uses [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of han. After that, you will obtain a model file named "han.pt".
  For a detailed parameter introduction, see [Python Model Parameters](./public_parameters_gnn.md).

**(2) Preparing input data**

  HAN requires three inputs: the edge table with type, the node feature table, and the node label table.

  HAN requires an edge file which contains three columns including the source node column, the destination column and
  the node type column. The third column indicates the destination nodes' types, each type indicates a meta-path of "
  A-B-A".

  For details, see [Data Format](./data_format_gnn.md).

**(3) Submit model to cluster**

  Once the model file and inputs are ready, submit the task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

  - dense/low-sparse

  ```bash
   <COMMON_SCRIP_PART>
   --files han.pt \ #path to the TorchScript model
   --class com.tencent.angel.pytorch.examples.supervised.cluster.HANExample \
   ./pytorch-on-angel-examples-${VERSION}.jar \ # jar from Compiling the examples submodule
   edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath\
   torchModelPath:han.pt featureDim:32 itemTypes:5 stepSize:0.01\
   optimizer:adam numEpoch:10 testRatio:0.5\
   numPartitions:50 format:sparse samples:10 batchSize:128\
   predictOutputPath:$predictOutputPath embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
   actionType:train numBatchInit:5
  ```

  - high-sparse

  ```bash
  <COMMON_SCRIP_PART>
  --files han.pt \   #path to the TorchScript model
  --class com.tencent.angel.pytorch.examples.supervised.cluster.HANExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \   # jar from Compiling the examples submodule
  edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath\
  torchModelPath:han.pt featureDim:32 itemTypes:5 stepSize:0.01\
  optimizer:adam numEpoch:10 testRatio:0.5 fieldNum:20 featEmbedDim:8 \
  numPartitions:50 format:sparse samples:10 batchSize:128\
  predictOutputPath:$predictOutputPath embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5
  ```

  For a full list of parameters and output results, see [details](./public_parameters_gnn.md).

  - edgePath: the input path (hdfs) of edge table, which contains src, dst and type

**Notes:**

- The model file, han.pt, should be uploaded to the Spark driver and every executor, so use `--files` to upload it.

**2. yaml-mode**

- demo.yaml

  ```yaml
  graph:
    name: "myGraph"
    nodes:
      - name: "user"
        feature:
          format: "dense"
          path: $userPath
        label:
          path: $labelPath
          validate_path: ""
    edges:
      base_config:
        weighted: false
        sep: "space"
      link_config:
        - name: "u2u"
          src: "user"
          dst: "user"
          path: $edgePath
  model:
    model_name: "han"
    input_dim: 32
    hidden_dim: 128
    output_dim: 11
    item_types: 267
    m: 64
    second: true
  trainer:
    epoch: 10
    batch_size: 128
    learning_rate: 0.01
    decay: 0.01
    test_ratio: 0.5
    optimizer: "adam"
    sample_num: 5
    batch_init_num: 5
    eval_metrics: "acc"
    sample_method: "random"
    validate_periods: 1
    periods: 10
    model_save_path: $savePath
  predictor:
    batch_size_multiplier: 5
    node_embedding_output_path: $embeddingPath
  ```

- script

  ```bash
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.bipartite.supervised.HANExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \
  actionType:train \
  dataPartitionNum:300 \
  psNumPartition:100 \
  storageLevel:MEMORY_ONLY \
  ```

#### 2.7 Semi Bipartite GraphSage

[Semi Bipartite GraphSage](https://conferences.computer.org/icde/2020/pdfs/ICDE2020-5acyuqhpJ6L9P042wmjY1p/290300b677/290300b677.pdf)
is a semi-supervised graph convolutional network for bipartite graphs.

Here we give an example of using Semi Bipartite GraphSage on PyTorch on Angel.

**1. script-mode**

**(1) Generate pytorch script model**

  Run the following command in the `python/graph/bipartite/supervised` directory:  
  - dense/low-sparse

  ```bash
  python bipartite_graphsage.py --input_user_dim 2 --input_item_dim 19 --hidden_dim 128 --output_dim 2 --output_file semi_bipartite_graphsage.pt --task_type classification
  ```

  - high-sparse

   ```bash
   python bipartite_graphsage.py --input_user_dim 10 --input_item_dim 10 --hidden_dim 128 --output_dim 2 --output_file semi_bipartite_graphsage_sparse.pt --task_type classification --input_user_field_num 3 --input_item_field_num 3 --input_user_embedding_dim 8 --input_item_embedding_dim 16
   ```

  This script uses [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of Semi Bipartite GraphSage. After that, you will obtain a model file named "
  semi_bipartite_graphsage.pt".
  For a detailed parameter introduction, see [Python Model Parameters](./public_parameters_gnn.md).

**(2) Preparing input data**

  Semi Bipartite GraphSage requires four inputs: the edge table, the user node feature table, the item node feature
  table, and the label table for user nodes.

  For details, see [Data Format](./data_format_gnn.md).

**(3) Submit model to cluster**

  Once the model file and inputs are ready, submit the task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md). 
  - dense/low-sparse

  ```bash
  <COMMON_SCRIP_PART>
  --files semi_bipartite_graphsage.pt \ #path to the TorchScript model
  --class com.tencent.angel.pytorch.examples.supervised.cluster.BiGCNExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \ # jar from Compiling the examples submodule
  edgePath:$edgePath userFeaturePath:$featurePath labelPath:$labelPath\
  torchModelPath:semi_bipartite_graphsage.pt userFeatureDim:2 itemFeatureDim:19 stepSize:0.01\
  optimizer:adam numEpoch:10 testRatio:0.5\
  numPartitions:50 format:sparse userNumSamples:10 itemNumSamples:10 batchSize:128\
  predictOutputPath:$predictOutputPath userEmbeddingPath:$userEmbeddingPath outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5\
  ```

  - high-sparse

   ```bash
   <COMMON_SCRIP_PART>
   --files semi_bipartite_graphsage_sparse.pt \   #path to the TorchScript model
   --class com.tencent.angel.pytorch.examples.supervised.cluster.BiGCNExample \
   ./pytorch-on-angel-examples-${VERSION}.jar \   # jar from Compiling the examples submodule
   edgePath:$edgePath userFeaturePath:$featurePath labelPath:$labelPath\
   torchModelPath:semi_bipartite_graphsage_sparse.pt userFeatureDim:10 itemFeatureDim:10 stepSize:0.01\
   optimizer:adam numEpoch:10 testRatio:0.5 userFieldNum:3 itemFieldNum:3 userFeatEmbedDim:8 itemFeatEmbedDim:16\
   numPartitions:50 format:sparse userNumSamples:10 itemNumSamples:10 batchSize:128\
   predictOutputPath:$predictOutputPath userEmbeddingPath:$userEmbeddingPath outputModelPath:$outputModelPath\
   actionType:train numBatchInit:5
   ```

  For a full list of parameters and output results, see [details](./public_parameters_gnn.md).

**Notes:**

- The model file, semi_bipartite_graphsage.pt, should be uploaded to the Spark driver and every executor, so use `--files` to upload it.

**2. yaml-mode**

Not supported.


#### 2.8 Unsupervised Bipartite GraphSage

[Unsupervised Bipartite GraphSage](https://conferences.computer.org/icde/2020/pdfs/ICDE2020-5acyuqhpJ6L9P042wmjY1p/290300b677/290300b677.pdf)
is an unsupervised graph convolutional network for bipartite graphs.

Here we give an example of using Unsupervised Bipartite GraphSage on PyTorch on Angel.

**1. script-mode**

**(1) Generate pytorch script model**

  Run the following command in the `python/graph/bipartite/unsupervised` directory:

  ```bash
  python bipartite_graphsage.py --input_user_dim 2 --input_item_dim 19 --hidden_dim 128 --output_dim 128 --output_file un_bipartite_graphsage.pt
  ```

  This script uses [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of Unsupervised Bipartite GraphSage. After that, you will obtain a model file named "
  un_bipartite_graphsage.pt".
  For a detailed parameter introduction, see [Python Model Parameters](./public_parameters_gnn.md).

**(2) Preparing input data**

  Unsupervised Bipartite GraphSage requires three inputs: the edge table, the user node feature table, and the item
  node feature table.

  For details, see [Data Format](./data_format_gnn.md).

**(3) Submit model to cluster**
  Once the model file and inputs are ready, submit the task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

  ```bash
  <COMMON_SCRIP_PART>
  --files unsupervised_bipartite_graphsage.pt \   #path to the TorchScript model
  --class com.tencent.angel.pytorch.examples.unsupervised.cluster.BiGraphSageExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \   # jar from Compiling the examples submodule
  edgePath:$edgePath userFeaturePath:$featurePath labelPath:$labelPath\
  torchModelPath:unsupervised_bipartite_graphsage.pt userFeatureDim:2 itemFeatureDim:19 stepSize:0.01\
  optimizer:adam numEpoch:10\
  numPartitions:50 format:sparse userNumSamples:10 itemNumSamples:10 batchSize:128\
  predictOutputPath:$predictOutputPath userEmbeddingPath:$userEmbeddingPath itemEmbeddingPath:$itemEmbeddingPath outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5
  ```

  For a full list of parameters and output results, see [details](./public_parameters_gnn.md).

**Notes:**

- The model file, unsupervised_bipartite_graphsage.pt, should be uploaded to the Spark driver and every executor, so use `--files` to upload it.

**2. yaml-mode**

- demo.yaml

  ```yaml
  graph:
    name: "myGraph"
    nodes:
      - name: "user"
        feature:
          format: "dense"
          path: $userPath
        label:
          path: $labelPath
          validate_path: ""
    edges:
      base_config:
        weighted: false
        sep: "space"
      link_config:
        - name: "u2u"
          src: "user"
          dst: "user"
          path: $edgePath
  model:
    model_name: "bipartite_graphsage"
    input_user_dim: 128
    input_item_dim: 128
    negative_size: 32
    hidden_dim: 32
    output_dim: 64
    encode: "dense"
    neg_sampling: false
    second: true
  trainer:
    epoch: 10
    batch_size: 128
    learning_rate: 0.001
    decay: 0.01
    user_sample_num: 10
    item_sample_num: 10
    train_sample_ratio: 1.0
    optimizer: "adam"
    multi_hot_field: false
    batch_init_num: 5
    sample_method: "random"
    validate_periods: 1
    periods: 10
    model_save_path: $savePath
  predictor:
    node_embedding_output_path: $embeddingPath
  ```

- script

  ```bash
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.bipartite.unsupervised.BiGraphSageExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \
  actionType:train \
  dataPartitionNum:1 \
  dataPartitionNumFactor:1 \
  psPartitionNum:1 \
  psPartitionNumFactor:1 \
  storageLevel:MEMORY_ONLY \
  ```


#### 2.9 Unsupervised HGAT

[Heterogeneous Graph Attention Network](https://arxiv.org/pdf/1903.07293.pdf), HGAT, is an unsupervised graph attention
convolution network for bipartite graphs.

Here we give an example of using HGAT on PyTorch on Angel.

**1. script-mode**

**(1) Generate pytorch script model**

  Run the following command in the `python/graph` directory:
  - dense/low-sparse

  ```bash
  python unsupervised_heterogeneous_gat.py --input_user_dim 64 --input_item_dim 64 --hidden_dim 64 --output_dim 64 --output_file hgat_dense.pt --negative_size 32 --heads 2
  ```

  - high-sparse

  ```bash
  python unsupervised_heterogeneous_gat.py --input_user_dim 32 --input_item_dim 32 --hidden_dim 8 --output_dim 64 --output_file hgat_sparse.pt --input_user_field_num 4 --input_item_field_num 2 --input_user_embedding_dim 8 --input_item_embedding_dim 16 --negative_size 32 --heads 2 --encode multi-hot
  ```

  This script uses [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of HGAT. After that, you will obtain a model file named `hgat_dense.pt` or `hgat_sparse.pt`.
  For a detailed parameter introduction, see [Python Model Parameters](./public_parameters_gnn.md).

**(2) Preparing input data**

  HGAT requires three inputs: the edge table, the user feature table, and the item node feature table.

  For details, see [Data Format](./data_format_gnn.md).

**(3) Submit model to cluster**

  Once the model file and inputs are ready, submit the task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

  - dense/low-sparse

  ```bash
  <COMMON_SCRIP_PART>
  --files hgat_dense.pt \ #path to the TorchScript model
  --class com.tencent.angel.pytorch.examples.unsupervised.cluster.HGATExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \ # jar from Compiling the examples submodule
  edgePath:$edgePath userFeaturePath:$featurePath labelPath:$labelPath\
  torchModelPath:hgat_dense.pt userFeatureDim:64 itemFeatureDim:64 stepSize:0.0001 decay:0.001\
  optimizer:adam numEpoch:10 testRatio:0.5 \
  numPartitions:50 format:dense userNumSamples:5 itemNumSamples:5 batchSize:128\
  predictOutputPath:$predictOutputPath userEmbeddingPath:$userEmbeddingPath itemEmbeddingPath:$itemEmbeddingPath
  outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5
  ```

  - high-sparse

  ```bash
  <COMMON_SCRIP_PART>
  --files hgat_sparse.pt \ #path to the TorchScript model
  --class com.tencent.angel.pytorch.examples.unsupervised.cluster.HGATExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \ # jar from Compiling the examples submodule
  edgePath:$edgePath userFeaturePath:$featurePath labelPath:$labelPath\
  torchModelPath:hgat_sparse.pt userFeatureDim:25000000 itemFeatureDim:80000 stepSize:0.0001 decay:0.001 fieldMultiHot:true \
  optimizer:adam numEpoch:10 testRatio:0.5 userFieldNum:4 itemFieldNum:2 userFeatEmbedDim:8 itemFeatEmbedDim:16\
  numPartitions:50 format:sparse userNumSamples:5 itemNumSamples:5 batchSize:128\
  predictOutputPath:$predictOutputPath userEmbeddingPath:$userEmbeddingPath itemEmbeddingPath:$itemEmbeddingPath
  outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5
  ```

  For a full list of parameters and output results, see [details](./public_parameters_gnn.md).

**Notes:**

- The model file, hgat_sparse.pt, should be uploaded to the Spark driver and every executor, so use `--files` to upload it.

**2. yaml-mode**

- demo.yaml

  ```yaml
  graph:
    name: "myGraph"
    nodes:
      - name: "user"
        feature:
          format: "dense"
          path: $userPath
        label:
          path: $labelPath
          validate_path: ""
    edges:
      base_config:
        weighted: false
        sep: "space"
      link_config:
        - name: "u2u"
          src: "user"
          dst: "user"
          path: $edgePath
  model:
    model_name: "bipartite_gat"
    input_user_dim: 128
    input_item_dim: 128
    negative_size: 32
    hidden_dim: 32
    output_dim: 64
    encode: "dense"
    neg_sampling: false
    second: true # only support second order
  trainer:
    epoch: 10
    batch_size: 128
    learning_rate: 0.01
    decay: 0.01
    user_sample_num: 5
    item_sample_num: 5
    train_sample_ratio: 1.0
    optimizer: "adam"
    multi_hot_field: true
    batch_init_num: 5
    eval_metrics: "acc"
    sample_method: "random"
    validate_periods: 1
    periods: 10
    model_save_path: $savePath
  predictor:
    node_embedding_output_path: $embeddingPath
  ```

- script

  ```bash
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.bipartite.unsupervised.HGATExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \
  actionType:train \
  dataPartitionNum:10 \
  psNumPartition:10 \
  storageLevel:MEMORY_ONLY \
  ```

#### 2.10 IGMC

[Inductive Matrix Completion Based On Graph Neural Networks](https://arxiv.org/pdf/1904.12058.pdf), IGMC,
trains a graph neural network (GNN) based purely on 1-hop subgraphs
around (user, item) pairs generated from the rating matrix, and maps these subgraphs to their corresponding ratings.

Here we give an example of using IGMC on PyTorch on Angel.

**1. script-mode**

**(1) Generate pytorch script model**

  Run the following command in the `python/graph/bipartite/supervised` directory:  

  - classification

  ```bash
  python igmc.py --input_user_dim 23 --input_item_dim 18 --hidden_dim 32 --edge_types 5 --output_dim 5 --output_file igmc_ml_class.pt
  ```

  - regression

  ```bash
  python igmc.py --input_user_dim 23 --input_item_dim 18 --hidden_dim 32 --edge_types 5 --output_dim 5 --method regression --output_file igmc_ml_reg.pt
  ```

This script uses [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of IGMC. After that, you will obtain a model file named `igmc_ml_class.pt` or `igmc_ml_reg.pt`.
For a detailed parameter introduction, see [Python Model Parameters](./public_parameters_gnn.md).

**(2) Preparing input data**

  IGMC requires three inputs: the edge table (with rating), the user node feature table, and the item node feature table.

  For details, see [Data Format](./data_format_gnn.md).

**(3) Submit model to cluster**

Once the model file and inputs are ready, submit the task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).
The .pt models differ between classification and regression, but the submit scripts are the same.

  ```bash
  <COMMON_SCRIP_PART>
  --files igmc_ml_class.pt \ #path to the TorchScript model
  --class com.tencent.angel.pytorch.examples.supervised.cluster.IGMCExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \ # jar from Compiling the examples submodule
  edgePath:$edgePath userFeaturePath:$userFeaturePath itemFeaturePath:$itemFeaturePath\
  torchModelPath:igmc_ml_class.pt userFeatureDim:23 itemFeatureDim:18 stepSize:0.0001 decay:0.001\
  optimizer:adam numEpoch:10 testRatio:0.5 \
  numPartitions:50 format:dense batchSize:128\
  predictOutputPath:$predictOutputPath outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5
  ```

  For a full list of parameters and output results, see [details](./public_parameters_gnn.md).

  - edgePath: the input path (hdfs) of edge table, which contains src, dst and type

**Notes:**

- The model file, igmc_ml_class.pt, should be uploaded to the Spark driver and every executor, so use `--files` to upload it.

**2. yaml-mode**

- demo.yaml

  ```yaml
  graph:
    name: "myGraph"
    nodes:
      - name: "user"
        feature:
          format: "dense"
          path: $userPath
        label:
          path: $labelPath
          validate_path: ""
    edges:
      base_config:
        weighted: false
        sep: "space"
      link_config:
        - name: "u2u"
          src: "user"
          dst: "user"
          path: $edgePath
  model:
    model_name: "igmc"
    input_user_dim: 23
    input_item_dim: 18
    hidden_dim: 128
    edge_types: 6
    output_dim: 6
    task_type: "classification"
    second: false # igmc does not support second-order
    encode: "dense"
  trainer:
    epoch: 10
    batch_size: 128
    learning_rate: 0.01
    decay: 0.01
    test_ratio: 0.5
    optimizer: "adam"
    user_sample_num: 5
    item_sample_num: 5
    batch_init_num: 5
    eval_metrics: "acc"
    sample_method: "random"
    validate_periods: 1
    periods: 10
    model_save_path: $savePath
  predictor:
    batch_size_multiplier: 10
    node_embedding_output_path: $embeddingPath
  ```

- script

  ```bash
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.bipartite.supervised.IGMCExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \
  actionType:train \
  dataPartitionNum:300 \
  psNumPartition:100 \
  storageLevel:MEMORY_ONLY \
  ```


#### 2.11 GAMLP

Following the routine of decoupled GNNs, [GAMLP](https://arxiv.org/pdf/2108.10097.pdf) executes feature propagation
during pre-computation, which helps it maintain high scalability.

GAMLP contains two independent modules:

- GAMLP-aggregator, the feature propagation and aggregation module. This module only needs to be run once in the preprocessing stage to produce the feature input for subsequent GNN model training; it can also run as an independent feature propagation component shared by other GNN algorithms
- GAMLP-training, the GNN model training module, loads the features produced by the aggregation module for training. In this stage, node sampling and pulling node features no longer incur communication overhead, which greatly improves the model training efficiency.

Here we give an example of using GAMLP on PyTorch on Angel.

##### 2.11.1 GAMLP-aggregator

**1. script-mode**

**(1) Generate pytorch script model**

  Run the following command in the `python/graph/homogeneous/supervised` directory:

  ```bash
  python aggregator.py --aggregation_type mean --output_file aggregator.pt
  ```

  This script uses [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of GAMLP-aggregator. After that, you will obtain a model file named "aggregator.pt". Here we use
  the Cora dataset as an example, where the feature dimension for each node is 1433 with 7 different classes. 
  For a detailed parameter introduction, see [Python Model Parameters](./public_parameters_gnn.md).

**(2) Preparing input data**

  The GAMLP-aggregator requires two inputs: the edge table and the node feature table.

  For details, see [Data Format](./data_format_gnn.md).

**(3) Submit model to cluster**

  Once the model file and inputs are ready, submit the task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

  - dense/low-sparse

  ```bash
  <COMMON_SCRIP_PART>
  --files aggregator.pt \
  --class com.tencent.angel.pytorch.examples.supervised.cluster.AggregatorExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \ # jar from Compiling the examples submodule
  edgePath:$edgePath featurePath:$featurePath hops:4 sampleMethod:aliasTable \
  upload_torchModelPath:aggregator.pt featureDim:1433 sep:tab \
  numPartitions:100 format:dense samples:10 batchSize:128 numBatchInit:128 \
  predictOutputPath:$predictOutputPath periods:10 \
  checkpointInterval:10 psNumPartition:100 useBalancePartition:false
  ```

**2. yaml-mode**

- demo.yaml

  ```yaml
  graph:
    name: "myGraph"
    nodes:
      - name: "user"
        feature:
          format: "dense"
          path: $userPath
        label:
          path: $labelPath
          validate_path: ""
    edges:
      base_config:
        weighted: false
        sep: "space"
      link_config:
        - name: "u2u"
          src: "user"
          dst: "user"
          path: $edgePath
  model:
    model_name: "aggregator"
    input_dim: 602
    aggregation_type: "mean"
    second: false #aggregator does not support second
  trainer:
    epoch: 10
    batch_size: 128
    learning_rate: 0.01
    decay: 0.01
    test_ratio: 0.5
    optimizer: "adam"
    sample_num: 5
    batch_init_num: 5
    eval_metrics: "acc"
    sample_method: "random"
    validate_periods: 1
    periods: 10
    model_save_path: $savePath
  predictor:
    batch_size_multiplier: 5
    node_embedding_output_path: $embeddingPath
  ```

- script

  ```bash
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.homogeneous.supervised.AggregatorExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \
  actionType:train \
  dataPartitionNum:300 \
  psNumPartition:100 \
  storageLevel:MEMORY_ONLY \
  ```


##### 2.11.2 GAMLP-training

**1. script-mode**

**(1) Generate pytorch script model**

  Run the following command in the `python/graph/homogeneous/supervised` directory:

  ```bash
  python gamlp.py --input_dim 1433 --hidden_dim 128 --output_dim 7 --hops 4 --output_file gamlp.pt
  ```

  Note that the --hops must be the same as in the GAMLP-aggregator module.
  This script uses [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of GAMLP. After that, you will obtain a model file named `gamlp.pt`. Here we use
  the Cora dataset as an example, where the feature dimension for each node is 1433 with 7 different classes. 
  For a detailed parameter introduction, see [Python Model Parameters](./public_parameters_gnn.md).

**(2) Preparing input data**

  GAMLP-training requires three inputs: the edge table, the node feature table, and the node label table.
  Note that the feature comes from the predictOutputPath of the previous GAMLP-aggregator module.
  For details, see [Data Format](./data_format_gnn.md).

**(3) Submit model to cluster**

  Once the model file and inputs are ready, submit the task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

  - dense/low-sparse

  ```bash
  <COMMON_SCRIP_PART>
  --files gamlp.pt \
  --class com.tencent.angel.pytorch.examples.supervised.cluster.GAMLPExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \
  edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath hops:4\
  featureDim:1433 sep:tab labelsep:tab stepSize:0.001\
  optimizer:adam numEpoch:100 testRatio:0.4 validatePeriods:1 evals:acc,f1 \
  numPartitions:10 format:dense samples:10 batchSize:128 numBatchInit:128 \
  embeddingPath:$output periods:10 outputModelPath:$outputModelPath \
  checkpointInterval:10 predictOutputPath:$predictOutputPath psNumPartition:10 useBalancePartition:false
  ```

  For a full list of parameters and output results, see [details](./public_parameters_gnn.md).

**Notes:**

- The model file, aggregator.pt, should be uploaded to the Spark driver and every executor, so use `--files` to upload it.
- The --hops value must be the same in all parameter settings.

**2. yaml-mode**

- demo.yaml

  ```yaml
  graph:
    name: "myGraph"
    nodes:
      - name: "user"
        feature:
          format: "dense"
          path: $userPath
        label:
          path: $labelPath
          validate_path: ""
    edges:
      base_config:
        weighted: false
        sep: "space"
      link_config:
        - name: "u2u"
          src: "user"
          dst: "user"
          path: $edgePath
  model:
    model_name: "gamlp"
    input_dim: 602
    hidden_dim: 128
    output_dim: 41
    hops: 2
    second: false # gamlp does not support second
    encode: "dense"
  trainer:
    epoch: 10
    batch_size: 128
    learning_rate: 0.01
    decay: 0.01
    test_ratio: 0.5
    optimizer: "adam"
    sample_num: 5
    batch_init_num: 5
    eval_metrics: "acc"
    sample_method: "random"
    validate_periods: 1
    periods: 10
    model_save_path: $savePath
  predictor:
    batch_size_multiplier: 5
    node_embedding_output_path: $embeddingPath
  ```

- script

  ```bash
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.homogeneous.supervised.GAMLPExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \
  actionType:train \
  dataPartitionNum:300 \
  psNumPartition:100 \
  storageLevel:MEMORY_ONLY \
  ```

#### 2.12 HGAMLP

HGAMLP is an extension of GAMLP for heterogeneous graphs. It contains two independent modules:

- HGAMLP-aggregator, the feature propagation and aggregation module for heterogeneous graphs. It aggregates different types of nodes according to the metapaths specified by the user, and outputs the aggregated features of each node type.
- HGAMLP-training, the GNN model training module, which is the same as GAMLP-training.

Here we give an example of using HGAMLP on PyTorch on Angel.


##### HGAMLP-aggregator

**1. script-mode**

**(1) Generate pytorch script model**

  Run the following command in the `python/graph` directory:

```bash
python aggregator.py --aggregation_type mean --output_file aggregator.pt
```

This script uses [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of GAMLP-aggregator. After that, you will obtain a model file named "aggregator.pt". Here we use
the Cora dataset as an example, where the feature dimension for each node is 1433 with 7 different classes. 
For a detailed parameter introduction, see [Python Model Parameters](./public_parameters_gnn.md).

**(2) Preparing input data**

  The HGAMLP-aggregator requires two inputs: the edge table and the node feature table.

  For details, see [Data Format](./data_format_gnn.md).

**(3) Submit model to cluster**

  Once the model file and inputs are ready, submit the task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

  - dense/low-sparse

  ```bash
  <COMMON_SCRIP_PART>
  --files aggregator.pt \
  --class com.tencent.angel.pytorch.examples.supervised.cluster.HeteAggregatorExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \ # jar from Compiling the examples submodule
  edgePaths:$edgePaths featurePaths:$featurePaths metapaths:$metapaths isWeighted:false \
  upload_torchModelPath:aggregator.pt sep:tab featureSep:tab featureDims:$featureDims aggregator_in_scala:true \
  numPartitions:2 format:dense samples:10 batchSize:128 numBatchInit:128 useWeightedAggregate:true \
  embeddingOutputPaths:$embeddingOutputPaths periods:10 sampleMethod:random \
  checkpointInterval:10 psNumPartition:2 useBalancePartition:false
  ```

**2. yaml-mode**

Not supported.


##### HGAMLP-training

**(1) Generate pytorch script model**

  Run the following command in the `python/graph` directory:

  ```bash
  python gamlp.py --input_dim 1433 --hidden_dim 128 --output_dim 7 --hops 4 --output_file gamlp.pt
  ```

  Note that the --hops must be the same as the HGAMLP-aggregator metapath length.
  This script uses [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of GAMLP. After that, you will obtain a model file named `gamlp.pt`. Here we use
the Cora dataset as an example, where the feature dimension for each node is 1433 with 7 different classes.
  For a detailed parameter introduction, see [Python Model Parameters](./public_parameters_gnn.md).

**(2) Preparing input data**

  HGAMLP-training requires three inputs: the edge table, the node feature table, and the node label table.
  Note that the feature comes from the predictOutputPath of the previous GAMLP-aggregator module.
  For details, see [Data Format](./data_format_gnn.md).

**(3) Submit model to cluster**

  Once the model file and inputs are ready, submit the task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

  - dense/low-sparse

  ```bash
  <COMMON_SCRIP_PART>
  --files gamlp.pt \
  --class com.tencent.angel.pytorch.examples.supervised.cluster.GAMLPExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \
  edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath hops:4\
  featureDim:1433 sep:tab labelsep:tab stepSize:0.001\
  optimizer:adam numEpoch:100 testRatio:0.4 validatePeriods:1 evals:acc,f1 \
  numPartitions:10 format:dense samples:10 batchSize:128 numBatchInit:128 \
  embeddingPath:$output periods:10 outputModelPath:$outputModelPath \
  checkpointInterval:10 predictOutputPath:$predictOutputPath psNumPartition:10 useBalancePartition:false
  ```

For a full list of parameters and output results, see [details](./public_parameters_gnn.md).

**Notes:**

- The model file, aggregator.pt and gamlp.pt, should be uploaded to the Spark driver and every executor, so use `--files` to upload it.
- The --hops value must be the same in all parameter settings.

**2. yaml-mode**

- demo.yaml

  ```yaml
  graph:
    name: "myGraph"
    nodes:
      - name: "user"
        feature:
          format: "dense"
          path: $userPath
        label:
          path: $labelPath
          validate_path: ""
    edges:
      base_config:
        weighted: false
        sep: "space"
      link_config:
        - name: "u2u"
          src: "user"
          dst: "user"
          path: $edgePath
  model:
    model_name: "gamlp"
    input_dim: 602
    hidden_dim: 128
    output_dim: 41
    hops: 2
    second: false # gamlp does not support second
    encode: "dense"
  trainer:
    epoch: 10
    batch_size: 128
    learning_rate: 0.01
    decay: 0.01
    test_ratio: 0.5
    optimizer: "adam"
    sample_num: 5
    batch_init_num: 5
    eval_metrics: "acc"
    sample_method: "random"
    validate_periods: 1
    periods: 10
    model_save_path: $savePath
  predictor:
    batch_size_multiplier: 5
    node_embedding_output_path: $embeddingPath
  ```

- script

  ```bash
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.homogeneous.supervised.GAMLPExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \
  actionType:train \
  dataPartitionNum:300 \
  psNumPartition:100 \
  storageLevel:MEMORY_ONLY \
  ```

#### 2.13 GATNE

GATNE is an algorithm framework for embedding large-scale Attributed Multiplex Heterogeneous Networks, implemented
on [Representation Learning for Attributed Multiplex Heterogeneous Network](https://arxiv.org/abs/1905.01669). For a
heterogeneous graph with nodes and edges of multiple types, it can learn an embedding for every node given the node
features.

Here we give an example of using GATNE on PyTorch on Angel.

**1. script-mode**

**(1) Generate pytorch script model**

  Run the following command in the `python/graph/heterogeneous/unsupervised` directory:

  ```bash
  python gatne.py --node_types 0,1,2 --edge_types 0,1 --schema 0-0-1,0-1-2 --in_dims 0:32,1:16,2:128 --input_split_idxs 0:32,1:32,2:128 --embedding_dim 100
  ```

  This script uses [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of GATNE. After that, you will obtain a model file named "gatne.pt". Here we use the Cora dataset
  as an example. 
  For a detailed parameter introduction, see [Python Model Parameters](./public_parameters_gnn.md).

**(2) Preparing input data**

  GATNE requires four inputs: the typed edge table, the node type table, the node feature tables, and the random
  walk table.

  - edgePath: the typed edge table, which contains src, dst and edge type
  - nodeTypePath: the node type table, which contains node and node type
  - featurePaths: the feature tables for each node type, in the format `type0:path0,type1:path1,...`
  - walkPath: the random walk sequences used as training samples

  For details, see [Data Format](./data_format_gnn.md).

**(3) Submit model to cluster**

  ```bash
  <COMMON_SCRIP_PART>
  --files gatne.pt \   #path to the TorchScript model
  --class com.tencent.angel.pytorch.examples.unsupervised.cluster.GATNEExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \
  edgePath:$edgePath testEdgePath:$testEdgePath walkPath:$walkPath nodeTypePath:$nodeTypePath \
  featurePaths:0:$userFeatPath,1:$itemFeatPath1,2:$itemFeatPath2 \
  stepSize:0.1 optimizer:adam numEpoch:5 decay:0.0 evals:auc validatePeriods:1 \
  contextDim:32 window:10 negative:10 eachNumSamples:0:9,1:8,2:7 \
  numPartitions:10 psNumPartition:10 format:dense batchSize:50 \
  embeddingOutputPath:$embeddingOutputPath outputModelPath:$outputModelPath \
  actionType:train useBalancePartition:false
  ```

**Notes:**

- The model file, gatne.pt, should be uploaded to the Spark driver and every executor, and it is auto-detected from the working directory. Therefore, use `--files` to upload the model file.

**2. yaml-mode**

- demo.yaml

  ```yaml
  graph:
    name: "myGraph"
    nodes:
       - name: "0"
         feature:
           format: "dense"
           sep: "tab"
           path: $node0Path
       - name: "1"
         feature:
           format: "dense"
           sep: "tab"
           path: $node1Path
       - name: "2"
         feature:
           format: "dense"
           sep: "tab"
           path: $node2Path
    edges:
      base_config:
        sep: "space"
      link_config:
        - name: "all_edges"
          path: $edgePath
  model:
    model_name: "gatne"
    in_dims: "0:1156,1:1156,2:1156"
    embedding_dim: 200
    embedding_u_dim: 10
    dim_a: 20
    node_types: "0,1,2"
    edge_types: "0,1,2,3,4,5,6,7,8"
    schema: "0-0-0,0-1-1,0-2-2,1-3-0,1-4-1,1-5-2,2-6-0,2-7-1,2-8-2"
    aggregation_type: "sum"
    encode: "dense"
    dropout: 0.3
  trainer:
    epoch: 10
    batch_size: 5000
    learning_rate: 0.0015
    decay: 0.5
    nodes_sample_num: "0:9,1:8,2:7,3:6,4:5,5:6,6:8,7:8,8:9"
    window_size: 5
    negative_sample_num: 5
    optimizer: "adam"
    eval_metrics: "auc"
    validate_periods: 1
    model_save_interval: 10
    init_method: randomNormal
    mean: 0
    std: 1
    save_checkpoint: true
    evaluate_by_edge_type: true
    output_embedding_by_node_type: true
    output_embedding_by_edge_type: false
    test_edge_path: $testEdgePath
    walk_path: $walkPath
    node_type_path: $typePath
    model_save_path: $savePath
    context_embedding_load_path: ""
    feat_embedding_load_path: ""
  predictor:
    node_embedding_output_path: $embeddingPath
  ```

- script

  ```bash
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.heterogeneous.unsupervised.GATNEExample \
  ./pytorch-on-angel-examples-${VERSION}.jar \
  actionType:train \
  dataPartitionNum:300 \
  psNumPartition:100 \
  storageLevel:MEMORY_ONLY
  ```


### FAQ

1. To use GAT or HGAT, PyTorch >= v1.5.0 is required.
2. If the loss is NaN or does not converge, decrease the learning rate, e.g. to 0.001, 0.0001, or lower.
3. If you encounter the error `file not found model.json`, check whether the versions of model.pt and PyTorch match.
4. If you encounter the error `java.lang.UnsupportedOperationException: empty collection`, check whether the input data is empty.
5. If you encounter the error `ERROR AngelYarnClient: submit application to yarn failed.`, the PS failed to apply for resources; switch to another cluster or try later.
6. If you encounter the error `java.lang.UnsatisfiedLinkError: no torch_angel in java.library.path`, check whether the torch path is correct.