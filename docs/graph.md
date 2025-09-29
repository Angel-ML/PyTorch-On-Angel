## GNN Algorithm

> Pytorch on Angel provides the ability to run graph convolution network algorithm. We
> defines the graph convolution networks based on [Pytorch-Geometric](https://github.com/rusty1s/pytorch_geometric)
> while
> using the parameter server of Angel to store the network structure and features of nodes.


### 1 Introduction

#### 1.1 How to predict

- change the path of input and output.
- assign `actionType` value as 'predict'.
- you can get `hdfs://modelPath/xx.pt` to local, then use it as predict; Or you can use the hdfs path, and
  set `--files hdfs://modelPath/xx.pt`, in this way the `torchModelPath` can be removed.

#### 1.2 How to train incrementally

- change the path of input and output, or you can use the same data to train incrementally.
- assign `actionType` value as 'train'.
- you can get `hdfs://modelPath/xx.pt` to local, then use it as training; Or you can use the hdfs path, and
  set `--files hdfs://modelPath/xx.pt`, in this way the `torchModelPath` can be removed.

#### 1.3 How to calculate the resource

In order to know how to set the resources, you should figure out where the data saved firstly, and then calculate how
much data storage space, finally set 2~3 times of data storage. The detail calculation method refer
to [Resource Calculation Method](./resource_calculation_method.md).

### 2 Algorithm Examples

In this part, many of algorithm has two different usage mode, `yaml-mode` and `script-mode`, while the properties
would be configured in `.yaml` file
and script file respectively.

Same content for following script configuration as `<COMMON_SCRIP_PART>`:

   ```$xslt
source ./spark-on-angel-env.sh  #see in bin/spark-on-angel-env.sh
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
--archives hdfs://path/of/torch.zip#torch\  #c++ library files
--queue $queue
--name $algorithmName
--driver-memory 5g \
--num-executors 5 \
--executor-cores 1 \
--executor-memory 5g \
   ```

#### 2.1 GraphSage

[GraphSage](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) generates embeddings by sampling and
aggregating features from a node’s local neighborhood. Here we give an example of how to run GraphSage algorithm beyond
Pytorch on Angel.

**1. script-mode**

**(1) Generate pytorch script model**

Running the following command under `python/graph` path.

  - dense/low-sparse

  ```$xslt
  python graphsage.py --input_dim 1433 --hidden_dim 128 --output_dim 7 --output_file graphsage_cora.pt
  ```

  - high-sparse

  ```$xslt
  python graphsage.py --input_dim 32 --input_embedding_dim 8 --input_field_num 20 --encode one-hot --hidden_dim 128 --output_dim 7 --output_file graphsage_sparse.pt
  ```

​	This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of graphsage. After that, you will 	obtain a model file named "graphsage_cora.pt". Here we use the Cora dataset as an example, where the feature dimension 	for each node is 1433 with 7 different classes.  Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

**(2) Preparing input data**

  There are three inputs required for graphsage, including the edge table, the node feature table and the node label table. The detail info see [Data Format](./data_format_gnn.md).

**(3) Submit model to cluster**

  After obtaining the model file and the inputs, we can submit a task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

  - dense/low-sparse

  ```$xslt
  <COMMON_SCRIP_PART>
  --files graphsage_cora.pt \   #path to pytorch script model
  --class com.tencent.angel.pytorch.examples.unsupervised.cluster.GraphSageExample \
  ./pytorch-on-angel-${VERSION}.jar \   # jar from Compiling java submodule
  edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath\
  torchModelPath:graphsage_cora.pt featureDim:1433 stepSize:0.01\
  optimizer:adam numEpoch:10 testRatio:0.5\
  numPartitions:50 format:sparse samples:10 batchSize:128\
  predictOutputPath:$predictOutputPath embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5
  ```

- high-sparse

  ```$xslt
  <COMMON_SCRIP_PART>
  --files graphsage_cora.pt \   #path to pytorch script model
  --class com.tencent.angel.pytorch.examples.supervised.cluster.GraphSageExample \
  ./pytorch-on-angel-${VERSION}.jar \   # jar from Compiling java submodule
  edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath\
  torchModelPath:graphsage_sparse.pt featureDim:32 stepSize:0.01\
  optimizer:adam numEpoch:10 testRatio:0.5 fieldNum:20 featEmbedDim:8 \
  numPartitions:50 format:sparse samples:10 batchSize:128\
  predictOutputPath:$predictOutputPath embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5
  ```

  Here we give a short description for the parameters in the submit script. Detailed parameters and the output result
  see [details](./public_parameters_gnn.md).

  **Notes:**

   - The model file, graphsage_cora.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need use ``--files`` to upload the model file.

**2. yaml-mode**

- demo.yaml

  ```$xslt
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
        path: $edgePath    model:
   model_name: "gat"
   input_dim: 32
   hidden_dim: 16
   output_dim: 100
   second: true
   encode: "dense"   trainer:
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
   model_save_path: $savePath    predictor:
   batch_size_multiplier: 5
   node_embedding_output_path: $embeddingPath
   predict_output_path: "$predictPath
  ```

- script

  ```$xslt
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.homogeneous.supervised.GraphSageExample \
  ./pytorch-on-angel-${VERSION}.jar \
  psPartitionNum:50 dataPartitionNum:100 psPartitionNumFactor:2 dataPartitionNumFactor:3 \
  storageLevel:MEMORY_AND_DISK actionType:train
  ```

#### 2.2 DGI/Unsupervised GraphSage

Here we give an example of how to run DGI algorithm beyond Pytorch on Angel.

**1. script-mode**

**(1) Generate pytorch script model**

  First, running the following command under `python/graph` path.

- DGI pt model

  ```$xslt
  python dgi.py --input_dim 1433 --hidden_dim 128 --output_dim 128 --output_file dgi_cora.pt
  ```

  - Unsupervised GraphSage pt model

  ```$xslt
  python unsupervised_graphsage.py --input_dim 1433 --hidden_dim 128 --output_dim 128 --output_file unsupervised_graphsage_cora.pt
  ```

  This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains
  the dataflow graph of dgi. After that, you will obtain a model file named "dgi_cora.pt". Here we use the Cora dataset
  as an example, where the feature dimension for each node is 1433.  
  Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md).

**(2) Preparing input data**

  There are two inputs required for dgi, including the edge table and the node feature table.
  The detail info see [Data Format](./data_format_gnn.md)

**(3) Submit model to cluster**

  After obtaining the model file and the inputs, we can submit a task
  through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).
  The only difference between **DGI** and Unsupervised **GraphSage** is pt model, the submit scripts same.

  ```$xslt
  <COMMON_SCRIP_PART>
  --files dgi_cora.pt \ #path to pytorch script model
  --class com.tencent.angel.pytorch.examples.unsupervised.cluster.DGIExample \
  ./pytorch-on-angel-${VERSION}.jar \ # jar from Compiling java submodule
  edgePath:$edgePath featurePath:$featurePath\
  torchModelPath:dgi_cora.pt featureDim:1433 stepSize:0.01\
  optimizer:adam numEpoch:10 \
  numPartitions:50 format:sparse samples:10 batchSize:128\
  embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5
  ```

  Here we give a short description for the parameters in the submit script. Detailed parameters and the output result
  see [details](./public_parameters_gnn.md).

**Notes:**

- The model file, dgi_cora.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need
  use ``--files`` to upload the model file.

**2. yaml-mode**

- demo.yaml

  ```$xslt
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

  ```$xslt
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.homogeneous.supervised.RGCNExample \
  ./pytorch-on-angel-${VERSION}.jar \
  actionType:train \
  dataPartitionNum:10 \
  psNumPartition:10 \
  storageLevel:MEMORY_ONLY \
  ```


#### 2.3 Relation GCN (RGCN)

Relation GCN is semi-supervised graph convolution network which can utilize the types of edges. The difference between 
RGCN and GCN is that each edge can has different types.

Here we give an example of using RGCN over pytorch on angel.

**1. script-mode**

**(1) Generate pytorch script model**

  First, go to directory of python/graph and execute the following command:

  ```$xslt
  python rgcn.py --input_dim 32 --hidden_dim 16 --n_class 2 --output_file rgcn_mutag.pt --n_relations 46 --n_bases 30
  ```

  This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains
  the dataflow graph of rgcn. After that, you will obtain a model file named "rgcn_mutag.pt". Where n_class is the
  number of classes, n_relations is the number of types for edges and n_bases is a parameter of RGCN to avoid
  overfitting.
  Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

**(2) Preparing input data**

  There are three inputs required for graphsage, including the edge table with type, the node feature table and the
  node label table.

  The detail info see [Data Format](./data_format_gnn.md)

**(3) Submit model to cluster**

  After obtaining the model file and the inputs, we can submit a task
  through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

  ```$xslt
  <COMMON_SCRIP_PART>
  --files rgcn_mutag.pt \   #path to pytorch script model
  --class com.tencent.angel.pytorch.examples.supervised.cluster.RGCNExample \
  ./pytorch-on-angel-${VERSION}.jar \
  edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath\
  torchModelPath:rgcn_mutag.pt featureDim:32 stepSize:0.01\
  optimizer:adam numEpoch:10 testRatio:0.5\
  numPartitions:50 format:sparse samples:10 batchSize:128\
  predictOutputPath:$predictOutputPath embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5
  ```

  Here we give a short description for the parameters in the submit script. Detailed parameters and the output result
  see [details](./public_parameters_gnn.md).

  - edgePath: the input path (hdfs) of edge table, which contains src, dst and type

**Notes:**

- The model file, rgcn_mutag.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need
  use ``--files`` to upload the model file.

**2. yaml-mode**

- demo.yaml

  ```$xslt
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
    predict_output_path: $predictPath
  ```

- script

  ```$xslt
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.homogeneous.supervised.RGCNExample \
  ./pytorch-on-angel-${VERSION}.jar \
  actionType:train \
  dataPartitionNum:10 \
  psNumPartition:10 \
  storageLevel:MEMORY_ONLY \
  ```


#### 2.4 EdgeProp

[EdgeProp](https://arxiv.org/abs/1906.05546) is an end-to-end Graph Convolution Network (GCN)-based algorithm to learn
the embeddings of the nodes and edges of a large-scale time-evolving graph. It consider not only node information and
also edge side information.

Here we give an example of using EdgeProp over pytorch on angel.

**1. script-mode**

**(1) Generate pytorch script model**

  First, go to directory of python/graph and execute the following command:

  ```$xslt
  python edgeprop.py --input_dim 23 --edge_input_dim 7 --hidden_dim 128 --output_dim 7 --output_file edgeprop_eth.pt
  ```

  This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains
  the dataflow graph of edgeProp. After that, you will obtain a model file named "edgeprop_eth.pt". Where
  edge\_input\_dim is the dimension of edge feature, other parameters are same as GraphSAGE.
  Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

**(2) Preparing input data**

  There are three inputs required for graphsage, including the edge table with edge feature, the node feature table and
  the node label table.

  The detail info see [Data Format](./data_format_gnn.md)

**(3) Submit model to cluster**

  After obtaining the model file and the inputs, we can submit a task
  through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

  ```$xslt
  <COMMON_SCRIP_PART>
  --files edgeprop_eth.pt \   #path to pytorch script model
  --class com.tencent.angel.pytorch.examples.supervised.cluster.EdgePropGCNExample \
  ./pytorch-on-angel-${VERSION}.jar \   # jar from Compiling java submodule
  edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath\
  torchModelPath:edgeprop_eth.pt featureDim:23 edgeFeatureDim:7 stepSize:0.01\
  optimizer:adam numEpoch:10 testRatio:0.5\
  numPartitions:50 format:sparse samples:10 batchSize:128\
  predictOutputPath:$predictOutputPath embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5
  ```

  Here we give a short description for the parameters in the submit script.

  - edgePath: the input path (hdfs) of edge table, which contains src, dst and edge feature

**Notes:**

- The model file, rgcn_mutag.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need
  use ``--files`` to upload the model file.

**2. yaml-mode**

- demo.yaml

  ```$xslt
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

  ```$xslt
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.homogeneous.supervised.EdgePropGCNExample \
  ./pytorch-on-angel-${VERSION}.jar \
  actionType:train \
  dataPartitionNum:10 \
  psNumPartition:10 \
  storageLevel:MEMORY_ONLY \
  ```

#### 2.5 GAT

Here we give an example of how to run GAT algorithm beyond Pytorch on Angel.

**1. script-mode**

**(1) Generate pytorch script model**

  First, go to directory of python/graph and execute the following command:

  ```$xslt
  python gat.py --input_dim 32 --hidden_dim 128 --output_dim 11 --output_file gat_am.pt
  ```

  This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains
  the dataflow graph of GAT. After that, you will obtain a model file named "gat_am.pt". Here we use the am dataset as
  an example, where the feature dimension for each node is 32 with 11 different classes.
  Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

**(2) Preparing input data**

  There are three inputs required for graphsage, including the edge table, the node feature table and the node label
  table.

  The detail info see [Data Format](./data_format_gnn.md)

**(3) Submit model to cluster**

  After obtaining the model file and the inputs, we can submit a task
  through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

  ```$xslt
  <COMMON_SCRIP_PART>
  --files gat_am.pt \   #path to pytorch script model
  --class com.tencent.angel.pytorch.examples.supervised.cluster.GATExample \
  ./pytorch-on-angel-${VERSION}.jar \   # jar from Compiling java submodule
  edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath\
  torchModelPath:gat_am.pt featureDim:32 stepSize:0.01\
  optimizer:adam numEpoch:10 testRatio:0.5\
  numPartitions:50 format:sparse samples:10 batchSize:128\
  predictOutputPath:$predictOutputPath embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5
  ```

  Here we give a short description for the parameters in the submit script. Detailed parameters and the output result
  see [details](./public_parameters_gnn.md)

**Notes:**

- The model file, gat_am.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need
  use ``--files`` to upload the model file.

**2. yaml-mode**

- demo.yaml

  ```$xslt
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

  ```$xslt
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.homogeneous.supervised.GATExample \
  ./pytorch-on-angel-${VERSION}.jar \
  actionType:train \
  dataPartitionNum:300 \
  psNumPartition:100 \
  storageLevel:MEMORY_ONLY \
  ```


#### 2.6 HAN

[HAN](https://arxiv.org/pdf/1903.07293.pdf) is a semi-supervised graph convolution network for heterogeneous graph. In
order to capture the heterogeneous information, HAN defined two different attentions: node-level and semantic level.
Here a simplified version of HAN is implemented, which accepts bipartite graph in the form of "user-item", where item
nodes could have multiple types. In another words, the input graph has multiple meta-paths in the form of "
user-item-user".
HAN classifies user nodes, and outputs their embeddings if needed.

Here we give an example of using HAN over pytorch on angel.

**1. script-mode**

**(1) Generate pytorch script model**

  First, go to directory of python/graph and execute the following command: 

  - dense/low-sparse

  ```$xslt
  python semi_han.py --m 64 --input_dim 32 --hidden_dim 16 --output_dim 2 --item_types 5  --output_file han.pt
  ```

  - high-sparse

   ```$xslt
   python semi_han.py --m 64 --input_dim 32 --input_embedding_dim 8 --input_field_num 20 --encode one-hot --hidden_dim 16 --output_dim 2 --item_types 5 --output_file han_sparse.pt
   ```

  This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains
  the dataflow graph of han. After that, you will obtain a model file named "han.pt".
  Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

**(2) Preparing input data**

  There are three inputs required for han, including the edge table with type, the node feature table and the node
  label table.

  HAN requires an edge file which contains three columns including the source node column, the destination column and
  the node type column. The third column indicates the destination nodes' types, each type indicates a meta-path of "
  A-B-A".

  The detail info see [Data Format](./data_format_gnn.md)

**(3) Submit model to cluster**

  After obtaining the model file and the inputs, we can submit a task
  through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

  - dense/low-sparse

  ```$xslt
   <COMMON_SCRIP_PART>
   --files han.pt \ #path to pytorch script model
   --class com.tencent.angel.pytorch.examples.supervised.cluster.HANExample \
   ./pytorch-on-angel-${VERSION}.jar \ # jar from Compiling java submodule
   edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath\
   torchModelPath:han.pt featureDim:32 temTypes:5 stepSize:0.01\
   optimizer:adam numEpoch:10 testRatio:0.5\
   numPartitions:50 format:sparse samples:10 batchSize:128\
   predictOutputPath:$predictOutputPath embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
   actionType:train numBatchInit:5
  ```

- high-sparse

  ```$xslt
  <COMMON_SCRIP_PART>
  --files han.pt \   #path to pytorch script model
  --class com.tencent.angel.pytorch.examples.supervised.cluster.HANExample \
  ./pytorch-on-angel-${VERSION}.jar \   # jar from Compiling java submodule
  edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath\
  torchModelPath:han.pt featureDim:32 temTypes:5 stepSize:0.01\
  optimizer:adam numEpoch:10 testRatio:0.5 fieldNum:20 featEmbedDim:8 \
  numPartitions:50 format:sparse samples:10 batchSize:128\
  predictOutputPath:$predictOutputPath embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5
  ```

  Here we give a short description for the parameters in the submit script. Detailed parameters and the output result
  see [details](./public_parameters_gnn.md)

  - edgePath: the input path (hdfs) of edge table, which contains src, dst and type

**Notes:**

- The model file, rgcn_mutag.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need
  use ``--files`` to upload the model file.

**2. yaml-mode**

- demo.yaml

  ```$xslt
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

  ```$xslt
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.bipartite.supervised.HANExample \
  ./pytorch-on-angel-${VERSION}.jar \
  actionType:train \
  dataPartitionNum:300 \
  psNumPartition:100 \
  storageLevel:MEMORY_ONLY \
  ```

#### 2.7 Semi Bipartite GraphSage

[Semi Bipartite GraphSage](https://conferences.computer.org/icde/2020/pdfs/ICDE2020-5acyuqhpJ6L9P042wmjY1p/290300b677/290300b677.pdf)
is a semi-supervised graph convolution network for Bipartite graph.

Here we give an example of using Semi Bipartite GraphSage over pytorch on angel.

**1. script-mode**

**(1) Generate pytorch script model**

  First, go to directory of python/graph and execute the following command:  
  - dense/low-sparse

  ```$xslt
  python semi_bipartite_graphsage.py --input_user_dim 2 --input_item_dim 19 --hidden_dim 128 --output_dim 2 --output_file semi_bipartite_graphsage.pt --task_type classification
  ```

- high-sparse

   ```$xslt
   python semi_bipartite_graphsage.py --input_user_dim 10 --input_item_dim 10 --hidden_dim 128 --output_dim 2 --output_file semi_bipartite_graphsage_sparse.pt --task_type classification --input_user_field_num 3 --input_item_field_num 3 --input_user_embedding_dim 8 --input_item_embedding_dim 16
   ```

  This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains
  the dataflow graph of Semi Bipartite GraphSage. After that, you will obtain a model file named "
  semi_bipartite_graphsage.pt".
  Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

**(2) Preparing input data**

  There are three inputs required for han, including the edge table, the user node feature table, the item node feature
  table and the label table for user node.

  The detail info see [Data Format](./data_format_gnn.md)

**(3) Submit model to cluster**

  After obtaining the model file and the inputs, we can submit a task
  through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).  
  - dense/low-sparse

  ```$xslt
  <COMMON_SCRIP_PART>
  --files semi_bipartite_graphsage.pt \ #path to pytorch script model
  --class com.tencent.angel.pytorch.examples.supervised.cluster.BiGCNExample \
  ./pytorch-on-angel-${VERSION}.jar \ # jar from Compiling java submodule
  edgePath:$edgePath userFeaturePath:$featurePath labelPath:$labelPath\
  torchModelPath:semi_bipartite_graphsage.pt userFeatureDim:2 itemFeatureDim:19 stepSize:0.01\
  optimizer:adam numEpoch:10 testRatio:0.5\
  numPartitions:50 format:sparse userNumSamples:10 itemNumSamples:10 batchSize:128\
  predictOutputPath:$predictOutputPath userEmbeddingPath:$userEmbeddingPath outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5\
  ```

- high-sparse

   ```$xslt
   <COMMON_SCRIP_PART>
   --files semi_bipartite_graphsage_sparse.pt \   #path to pytorch script model
   --class com.tencent.angel.pytorch.examples.supervised.cluster.BiGCNExample \
   ./pytorch-on-angel-${VERSION}.jar \   # jar from Compiling java submodule
   edgePath:$edgePath userFeaturePath:$featurePath labelPath:$labelPath\
   torchModelPath:semi_bipartite_graphsage_sparse.pt userFeatureDim:10 itemFeatureDim:10 stepSize:0.01\
   optimizer:adam numEpoch:10 testRatio:0.5 userFieldNum:3 itemFieldNum:3 userFeatEmbedDim:8 itemFeatEmbedDim:16\
   numPartitions:50 format:sparse userNumSamples:10 itemNumSamples:10 batchSize:128\
   predictOutputPath:$predictOutputPath userEmbeddingPath:$userEmbeddingPath outputModelPath:$outputModelPath\
   actionType:train numBatchInit:5
   ```

  Here we give a short description for the parameters in the submit script. Detailed parameters and the output result
  see [details](./public_parameters_gnn.md)

**Notes:**

- The model file, semi\_bipartite\_graphsage.pt, should be uploaded to Spark Driver and each Executor. Therefore, we
  need use ``--files`` to upload the model file.

**2. yaml-mode**

Not supported.


#### 2.8 Unsupervised Bipartite GraphSage

[Unsupervised Bipartite GraphSage](https://conferences.computer.org/icde/2020/pdfs/ICDE2020-5acyuqhpJ6L9P042wmjY1p/290300b677/290300b677.pdf)
is an unsupervised graph convolution network for Bipartite graph.

Here we give an example of using Unsupervised Bipartite GraphSage over pytorch on angel.

**1. script-mode**

**(1) Generate pytorch script model**

  First, go to directory of python/graph and execute the following command:

  ```$xslt
  python unsupervised_bipartite_graphsage.py --input_user_dim 2 --input_item_dim 19 --hidden_dim 128 --output_dim 128 --output_file un_bipartite_graphsage.pt
  ```

  This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains
  the dataflow graph of Unsupervised Bipartite GraphSage. After that, you will obtain a model file named "
  un_bipartite_graphsage.pt".
  Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

**(2) Preparing input data**

  There are three inputs required for Unsupervised Bipartite GraphSage, including the edge table, the user node feature
  table，and item node feature table.

  The detail info see [Data Format](./data_format_gnn.md)

**(3) Submit model to cluster**
  After obtaining the model file and the inputs, we can submit a task
  through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

  ```$xslt
  <COMMON_SCRIP_PART>
  --files unsupervised_bipartite_graphsage.pt \   #path to pytorch script model
  --class com.tencent.angel.pytorch.examples.unsupervised.cluster.BiGraphSageExample \
  ./pytorch-on-angel-${VERSION}.jar \   # jar from Compiling java submodule
  edgePath:$edgePath userFeaturePath:$featurePath labelPath:$labelPath\
  torchModelPath:unsupervised_bipartite_graphsage.pt userFeatureDim:2 itemFeatureDim:19 stepSize:0.01\
  optimizer:adam numEpoch:10\
  numPartitions:50 format:sparse userNumSamples:10 itemNumSamples:10 batchSize:128\
  predictOutputPath:$predictOutputPath userEmbeddingPath:$userEmbeddingPath itemEmbeddingPath:$itemEmbeddingPath outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5
  ```

  Here we give a short description for the parameters in the submit script. Detailed parameters and the output result
  see [details](./public_parameters_gnn.md)

**Notes:**

- The model file, unsupervised\_bipartite\_graphsage.pt, should be uploaded to Spark Driver and each Executor.
  Therefore, we need use ``--files`` to upload the model file.

**2. yaml-mode**

- demo.yaml

  ```$xslt
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

  ```$xslt
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.bipartite.unsupervised.BiGraphSageExample \
  ./pytorch-on-angel-${VERSION}.jar \
  actionType:train \
  dataPartitionNum:1 \
  dataPartitionNumFactor:1 \
  psPartitionNum:1 psPartitionNumFactor:1 \
  storageLevel:MEMORY_ONLY \
  ```


#### 2.9 Unsupervised HGAT

[Heterogeneous Graph Attention Network](https://arxiv.org/pdf/1903.07293.pdf), HGAT,  is a unsupervised graph attention 
convolution network for Bipartite graph.

Here we give an example of using HGAT over pytorch on angel.

**1. script-mode**

**(1) Generate pytorch script model**

  First, go to directory of python/graph and execute the following command:
  - dense/low-sparse

  ```$xslt
  python unsupervised_heterogeneous_gat.py --input_user_dim 64 --input_item_dim 64 --hidden_dim 64 --output_dim 64 --output_file hgat_dense.pt --negative_size 32 --heads 2
  ```

- high-sparse

  ```$xslt
  python unsupervised_heterogeneous_gat.py --input_user_dim 32 --input_item_dim 32 --hidden_dim 8 --output_dim 64 --output_file hgat_sparse.pt --input_user_field_num 4 --input_item_field_num 2 --input_user_embedding_dim 8 --input_item_embedding_dim 16 --negative_size 32 --heads 2 --encode multi-hot
  ```

  This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains
  the dataflow graph of Unsupervised Bipartite GraphSage. After that, you will obtain a model file named "hgat_dense.pt
  or hgat_sparse.pt".
  Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

**(2) Preparing input data**

  There are three inputs required for HGAT, including the edge table, the user feature node table，and item node feature
  table.

  The detail info see [Data Format](./data_format_gnn.md)

**(3) Submit model to cluster**

  After obtaining the model file and the inputs, we can submit a task
  through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

  - dense/low-sparse

  ```$xslt
  <COMMON_SCRIP_PART>
  --files hgat_dense.pt \ #path to pytorch script model
  --class com.tencent.angel.pytorch.examples.unsupervised.cluster.HGATExample \
  ./pytorch-on-angel-${VERSION}.jar \ # jar from Compiling java submodule
  edgePath:$edgePath userFeaturePath:$featurePath labelPath:$labelPath\
  torchModelPath:hgat_dense.pt userFeatureDim:64 itemFeatureDim:64 stepSize:0.0001 decay:0.001\
  optimizer:adam numEpoch:10 testRatio:0.5 \
  numPartitions:50 format:dense userNumSamples:5 itemNumSamples:5 batchSize:128\
  predictOutputPath:$predictOutputPath userEmbeddingPath:$userEmbeddingPath itemEmbeddingPath:$itemEmbeddingPath
  outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5
  ```

- high-sparse

  ```$xslt
  <COMMON_SCRIP_PART>
  --files hgat_sparse.pt \ #path to pytorch script model
  --class com.tencent.angel.pytorch.examples.unsupervised.cluster.HGATExample \
  ./pytorch-on-angel-${VERSION}.jar \ # jar from Compiling java submodule
  edgePath:$edgePath userFeaturePath:$featurePath labelPath:$labelPath\
  torchModelPath:hgat_sparse.pt userFeatureDim:25000000 itemFeatureDim:80000 stepSize:0.0001 decay:0.001 fieldMultiHot:
  true \
  optimizer:adam numEpoch:10 testRatio:0.5 userFieldNum:4 itemFieldNum:2 userFeatEmbedDim:8 itemFeatEmbedDim:16\
  numPartitions:50 format:sparse userNumSamples:5 itemNumSamples:5 batchSize:128\
  predictOutputPath:$predictOutputPath userEmbeddingPath:$userEmbeddingPath itemEmbeddingPath:$itemEmbeddingPath
  outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5
  ```

  Here we give a short description for the parameters in the submit script. Detailed parameters and the output result
  see [details](./public_parameters_gnn.md)

**Notes:**

- The model file, hgat_sparse.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need
  use ``--files`` to upload the model file.

**2. yaml-mode**

- demo.yaml

  ```$xslt
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

  ```$xslt
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.bipartite.unsupervised.HGATExample \
  ./lib/pytorch-on-angel-examples-0.4.0.jar \
  actionType:train \
  dataPartitionNum:10 \
  psNumPartition:10 \
  storageLevel:MEMORY_ONLY \
  ```

#### 2.10 IGMC

[Inductive Matrix Completion Base On Graph Nural NEetworks](https://arxiv.org/pdf/1904.12058.pdf) , IGMC, 
trains a graph neural network (GNN) based purely on 1-hop subgraphs
around (user, item) pairs generated from the rating matrix and maps these subgraphs to their corresponding ratings

Here we give an example of using IGMC over pytorch on angel.

**1. script-mode**

**(1) Generate pytorch script model**

  First, go to directory of python/graph and execute the following command:  

  - classification

  ```$xslt
  supervised_igmc.py --input_user_dim 23 --input_item_dim 18 --hidden_dim 32 --edge_types 5 --output_dim 5 --output_file igmc_ml_class.pt
  ```

  - regression

  ```$xslt
  python supervised_igmc.py --input_user_dim 23 --input_item_dim 18 --hidden_dim 32 --edge_types 5 --output_dim 5 --method regression --output_file igmc_ml_reg.pt
  ```

This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains
the dataflow graph of Unsupervised Bipartite GraphSage. After that, you will obtain a model file named "
igmc_ml_class.pt or igmc_ml_reg.pt".
Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

**(2) Preparing input data**

  There are three inputs required for IGMC, including the edge table(with rating), the node feature table.

  The detail info see [Data Format](./data_format_gnn.md)

**(3) Submit model to cluster**

After obtaining the model file and the inputs, we can submit a task
through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).
The pt models are difference between classification and regression, but the scripts is same.

  ```$xslt
  <COMMON_SCRIP_PART>
  --files igmc_ml_class.pt \ #path to pytorch script model
  --class com.tencent.angel.pytorch.examples.supervised.cluster.IGMCExample \
  ./pytorch-on-angel-${VERSION}.jar \ # jar from Compiling java submodule
  edgePath:$edgePath userFeaturePath:$userFeaturePath itemFeaturePath:$itemFeaturePath\
  torchModelPath:igmc_ml_class.pt userFeatureDim:23 itemFeatureDim:18 stepSize:0.0001 decay:0.001\
  optimizer:adam numEpoch:10 testRatio:0.5 \
  numPartitions:50 format:dense batchSize:128\
  predictOutputPath:$predictOutputPath outputModelPath:$outputModelPath\
  actionType:train numBatchInit:5
  ```

  Here we give a short description for the parameters in the submit script. Detailed parameters and the output result
  see [details](./public_parameters_gnn.md)

  - edgePath: the input path (hdfs) of edge table, which contains src, dst and type

**Notes:**

- The model file, igmc\_ml\_class.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need
  use ``--files`` to upload the model file.

**2. yaml-mode**

- demo.yaml

  ```$xslt
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

  ```$xslt
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.bipartite.supervised.IGMCExample \
  ./lib/pytorch-on-angel-examples-0.4.0.jar \
  actionType:train \
  dataPartitionNum:300 \
  psNumPartition:100 \
  storageLevel:MEMORY_ONLY \
  ```


#### 2.11 GAMLP

[GAMLP](https://arxiv.org/pdf/2108.10097.pdf) Following the routine of decoupled GNNs, the feature propagation in GAMLP
is executed during pre-computation, which helps it maintain high scalability.

GAMLP contains two independent modules:

- GAMLP-aggregator, feature propagation aggregation module. This module only needs to be run once in the preprocessing stage as the feature input for subsequent GNN model training; the module can also be run as an independent component as a feature propagation aggregation component common to other GNN algorithms
- GAMLP-training, the GNN model training module, loads the features of the aggregation module for training. In this training stage, it is no longer necessary to do node sampling, pulling node feature aggregation and other communication overhead operations, which greatly improves the model training efficiency.

Here we give an example of using GAMLP over pytorch on angel.

##### 2.11.1 GAMLP-aggregator

**1. script-mode**

**(1) Generate pytorch script model**

  First, go to directory of python/graph and execute the following command:

  ```$xslt
  python aggregator.py --aggregation_type mean --output_file aggregator.pt
  ```

  This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains
  the dataflow graph of GAMLP-aggregator. After that, you will obtain a model file named "aggregator.pt". Here we use
  the Cora dataset as an example, where the feature dimension for each node is 1433 with 7 different classes.  
  Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

**(2) Preparing input data**

  There are two inputs required for GAMLP-aggregator, including the edge table and the node feature table.

  The detail info see [Data Format](./data_format_gnn.md)

**(3) Submit model to cluster**

  After obtaining the model file and the inputs, we can submit a task
  through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

  - dense/low-sparse

  ```$xslt
  <COMMON_SCRIP_PART>
  --files aggregator.pt\
  --class com.tencent.angel.pytorch.examples.supervised.cluster.AggregatorExample \
  ./pytorch-on-angel-${VERSION}.jar \ # jar from Compiling java submodule
  edgePath:$edgePath featurePath:$featurePath hops:4 sampleMethod:aliasTable \
  upload_torchModelPath:aggregator.pt featureDim:1433 sep:tab \
  numPartitions:100 format:dense samples:10 batchSize:128 numBatchInit:128 \
  predictOutputPath:$predictOutputPath periods:10 \
  checkpointInterval:10 psNumPartition:100 useBalancePartition:false
  ```

**2. yaml-mode**

- demo.yaml

  ```$xslt
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

  ```$xslt
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.homogeneous.supervised.AggregatorExample \
  ./lib/pytorch-on-angel-examples-0.4.0.jar \
  actionType:train \
  dataPartitionNum:300 \
  psNumPartition:100 \
  storageLevel:MEMORY_ONLY \
  ```


##### 2.11.2 GAMLP-training

**1. script-mode**

**(1) Generate pytorch script model**

  First, go to directory of python/graph and execute the following command:

  ```$xslt
  python gamlp.py --input_dim 1433 --hidden_dim 128 --output_dim 7 --hops 4 --output_file gamlp.pt
  ```

  Note that the --hops must be same with the GAMLP-aggregator module.
  This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains
  the dataflow graph of GAMLP-aggregator. After that, you will obtain a model file named "aggregator.pt". Here we use
  the Cora dataset as an example, where the feature dimension for each node is 1433 with 7 different classes.  
  Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

**(2) Preparing input data**

  There are three inputs required for GAMLP-training, including the edge table, the node feature table and the node
  label table.
  Note that the feature comes from the predictOutputPath of the previous GAMLP-aggregator module.
  The detail info see [Data Format](./data_format_gnn.md)

**(3) Submit model to cluster**

  After obtaining the model file and the inputs, we can submit a task
  through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

  - dense/low-sparse

  ```$xslt
  <COMMON_SCRIP_PART>
  --files gamlp.pt\
  --class com.tencent.angel.pytorch.examples.supervised.cluster.GAMLPExample \
  ./pytorch-on-angel-${VERSION}.jar \
  edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath hops:4\
  featureDim:1433 sep:tab labelsep:tab stepSize:0.001\
  optimizer:adam numEpoch:100 testRatio:0.4 validatePeriods:1 evals:acc,f1 \
  numPartitions:10 format:dense samples:10 batchSize:128 numBatchInit:128 \
  embeddingPath:$output periods:10 outputModelPath:$outputModelPath \
  checkpointInterval:10 predictOutputPath:$predictOutputPath psNumPartition:10 useBalancePartition:false
  ```

  Here we give a short description for the parameters in the submit script. Detailed parameters and the output result
  see [details](./public_parameters_gnn.md)

**Notes:**

- The model file, aggregator.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need
  use ``--files`` to upload the model file.
- The --hops must be same in all parameter settings.

**2. yaml-mode**

- demo.yaml

  ```$xslt
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
    validatePeriods: 1
    periods: 10
    model_save_path: $savePath
  predictor:
    batch_size_multiplier: 5
    node_embedding_output_path: $embeddingPath
  ```

- script

  ```$xslt
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.homogeneous.supervised.GAMLPExample \
  ./lib/pytorch-on-angel-examples-0.4.0.jar \
  actionType:train \
  dataPartitionNum:300 \
  psNumPartition:100 \
  storageLevel:MEMORY_ONLY \
  ```

#### 2.12 HGAMLP

HGAMLP is an extension of GAMLP for heterogeneous graphs, it contains two independent modules:

- HGAMLP-aggregator, Feature propagation aggregation module for heterogeneous graphs. According to the metapaths passed in by the user, aggregate different types of nodes, and output the aggregated features of each type of nodes.
- HGAMLP-training, the GNN model training module, which is same as GAMLP-training.

Here we give an example of using HGAMLP over pytorch on angel.


##### HGAMLP-aggregator

**1. script-mode**

**(1) Generate pytorch script model**

  First, go to directory of python/graph and execute the following command:

```$xslt
    python aggregator.py --aggregation_type mean --output_file aggregator.pt
```

This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains
the dataflow graph of GAMLP-aggregator. After that, you will obtain a model file named "aggregator.pt". Here we use
the Cora dataset as an example, where the feature dimension for each node is 1433 with 7 different classes.  
Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

**(2) Preparing input data**

  There are two inputs required for HGAMLP-aggregator, including the edge table and the node feature table.

  The detail info see [Data Format](./data_format_gnn.md)

**(3) Submit model to cluster**

  After obtaining the model file and the inputs, we can submit a task
  through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

  - dense/low-sparse

  ```$xslt
  <COMMON_SCRIP_PART>
  --files aggregator.pt\
  --class com.tencent.angel.pytorch.examples.supervised.cluster.HeteAggregatorExample  \
  ./pytorch-on-angel-${VERSION}.jar \ # jar from Compiling java submodule
  edgePaths:$edgePaths featurePaths:$featurePaths metapaths:$metapaths isWeighted:false \
  upload_torchModelPath:aggregator.pt sep:tab featureSep:tab featureDims:$featureDims aggregator_in_scala:true \
  numPartitions:2 format:dense samples:10 batchSize:128 numBatchInit:128 useWeightedAggregate:true \
  embeddingOutputPaths:$embeddingOutputPaths periods:10 sampleMethod:randm \
  checkpointInterval:10 psNumPartition:2 useBalancePartition:false
  ```

**2. yaml-mode**

Not supported.


##### HGAMLP-training

**(1) Generate pytorch script model**

  First, go to directory of python/graph and execute the following command:

  ```$xslt
  python gamlp.py --input_dim 1433 --hidden_dim 128 --output_dim 7 --hops 4 --output_file gamlp.pt
  ```

  Note that the --hops must be same with the HGAMLP-aggregator metapaths length.
  This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains
  the dataflow graph of GAMLP-aggregator. After that, you will obtain a model file named "aggregator.pt". Here we use
  the Cora dataset as an example, where the feature dimension for each node is 1433 with 7 different classes.  
  Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

**(2) Preparing input data**

  There are three inputs required for GAMLP-training, including the edge table, the node feature table and the node
  label table.
  Note that the feature comes from the predictOutputPath of the previous GAMLP-aggregator module.
  The detail info see [Data Format](./data_format_gnn.md)

**(3) Submit model to cluster**

  After obtaining the model file and the inputs, we can submit a task
  through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

  - dense/low-sparse

  ```$xslt
  <COMMON_SCRIP_PART>
  --files gamlp.pt\
  --class com.tencent.angel.pytorch.examples.supervised.cluster.GAMLPExample \
  ./pytorch-on-angel-${VERSION}.jar \
  edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath hops:4\
  featureDim:1433 sep:tab labelsep:tab stepSize:0.001\
  optimizer:adam numEpoch:100 testRatio:0.4 validatePeriods:1 evals:acc,f1 \
  numPartitions:10 format:dense samples:10 batchSize:128 numBatchInit:128 \
  embeddingPath:$output periods:10 outputModelPath:$outputModelPath \
  checkpointInterval:10 predictOutputPath:$predictOutputPath psNumPartition:10 useBalancePartition:false
  ```

Here we give a short description for the parameters in the submit script. Detailed parameters and the output result
see [details](./public_parameters_gnn.md)

**Notes:**

- The model file, aggregator.pt and gamlp.pt, should be uploaded to Spark Driver and each Executor. Therefore, we
  need use ``--files`` to upload the model file.
- The --hops must be same in all parameter settings.

**2. yaml-mode**

- demo.yaml

  ```$xslt
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
    validatePeriods: 1
    periods: 10
    model_save_path: $savePath
  predictor:
    batch_size_multiplier: 5
    node_embedding_output_path: $embeddingPath
  ```

- script

  ```$xslt
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.homogeneous.supervised.GAMLPExample \
  ./lib/pytorch-on-angel-examples-0.4.0.jar \
  actionType:train \
  dataPartitionNum:300 \
  psNumPartition:100 \
  storageLevel:MEMORY_ONLY \
  ```

#### 2.13 GATNE

GATNE is a algorithms framework for embedding large-scale Attributed Multiplex Heterogeneous Networks(AMHW), implemented
on the [Representation Learning for Attributed Multiplex Heterogeneous Network](https://arxiv.org/abs/1905.01669). A
heterogeneous graph with nodes and edges of multiple types, it can learn embedding for every node by given features of
node.

Here we give an example of using GATNE over pytorch on angel.

**1. script-mode**

**(1) Generate pytorch script model**

  First, go to directory of python/graph and execute the following command:

  ```$xslt
  python gatne.py --node_types 0,1,2 --edge_types 0,1 --schema 0-0-1,0-1-2 --in_dims 0:32,1:16,2:128 --input_split_idxs 0:32,1:32,2:128 --embedding_dim 100
  ```

  This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains
  the dataflow graph of GATNE. After that, you will obtain a model file named "gatne.pt". Here we use the Cora dataset
  as an example.  
  Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

**(2) Preparing input data**

  There are three inputs required for GATNE, including the edge table, the node feature table and the node
  label table.
  Note that the feature comes from the predictOutputPath of the previous GATNE module.
  The detail info see [Data Format](./data_format_gnn.md)

**(3) Submit model to cluster**

  ```$xslt
  <COMMON_SCRIP_PART>
  --files gamlp.pt\
  --class com.tencent.angel.pytorch.examples.unsupervised.cluster.GATNEExample \
  ./pytorch-on-angel-${VERSION}.jar \
  edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath hops:4\
  featureDim:1433 sep:tab labelsep:tab stepSize:0.001\
  optimizer:adam numEpoch:100 testRatio:0.4 validatePeriods:1 evals:acc,f1 \
  numPartitions:10 format:dense samples:10 batchSize:128 numBatchInit:128 \
  embeddingPath:$output periods:10 outputModelPath:$outputModelPath \
  checkpointInterval:10 predictOutputPath:$predictOutputPath psNumPartition:10 useBalancePartition:false
  ```

**Notes:**

- The GATNE.yaml of application configuration file should be uploaded to Spark Driver and each Executor on training
  mode, while the model `.pt` file is indispensable on predict mode.
- Therefore, we need use `--files` to upload all files.

**2. yaml-mode**

- demo.yaml

  ```$xslt
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
    save_checkpoint: "true"
    evaluate_by_edge_type: "true"
    output_embedding_by_node_type: "true"
    output_embedding_by_edge_type: "false"
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

  ```$xslt
  <COMMON_SCRIP_PART>
  --files demo.yaml \
  --class com.tencent.angel.pytorch.examples.graph.heterogeneous.unsupervised.GATNEExample \
  ./lib/pytorch-on-angel-examples-0.4.0.jar \
  actionType:train \
  dataPartitionNum:300 \
  psNumPartition:100 \
  storageLevel:MEMORY_ONLY
  ```


### FAQ

1. If you want to use GAT or HGAT, pytorch >= v1.5.0.
2. If you found loss is NAN or does not converge, you can decrease the learning rate, such as: 0.001,0.0001 or lower.
3. If you encounter the error `file not found model.json`, please check the version of model.pt and the version of
   pytorch, whether them are matched.
4. If you encounter the error `java.lang.UnsupportedOperationException: empty collection`, please check whether the
   input data is empty.
5. If you encounter the error `ERROR AngelYarnClient: submit application to yarn failed.`, ps did not apply for
   resources, change another cluster or try later.
6. If you encounter the error `java.lang.UnsatisfiedLinkError: no torch_angel in java.library.path`, please check
   whether the torch path is correct.
   `