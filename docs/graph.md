## GraphSage/GCN/DGI/EdgeProp

Pytorch on Angel provides the ability to run graph convolution network algorithm. We follow [Pytorch-Geometric](https://github.com/rusty1s/pytorch_geometric) to define the graph convolution networks while using the parameter server of Angel to store the network structure and features of nodes.


### Introduction
#### How to predict
1. change the input data path and output path
2. change `actionType` to `predict`
3. you can get hdfs://modelPath/xx.pt to local, then use it as training; Or you can use the hdfs path, and set `--files hdfs://modelPath/xx.pt`, in this way the `torchModelPath` can be remove


#### How to train incrementally
1. change the input data path and output path, or you can use the same data to train incrementally
2. set `actionType` as `train` 
3. you can get hdfs://modelPath/xx.pt to local, then use it as training; Or you can use the hdfs path, and set `--files hdfs://modelPath/xx.pt`, in this way the `torchModelPath` can be remove

#### How to calculate the resource

##### Dense data or low dimentition sparse data (dim less than 2000 generally)
- Angel PS resource: in order to ensure that Angel does not hang up, it is necessary to configure memory that is about twice the size of the model. The formula for calculating the size of the graph feature is: node_num * feature_dim * 4Byte, such as: 1kw nodes, feature_dim is 100, the size of graph feature is about 4G, then set ps.instances=3, ps.memory=4G is reasonable(total memory is 2~3 times of data saved in ps.). Of course, this is only a simple case. In many algorithms, there are many types of data saved in ps, such as: edge, node feature, node label etc.
	- `Semi GraphSage/GAT`: edge, node feature, node label, the model size is: (edge_num * 2 * 8Byte + node_num * feature_dim * 4Byte + the num of node with label * 4Byte)
	- `RGCN/HAN`: edge with type(dst type), node feature, node label, the model size is: (edge_num * 2 * 8Byte + edge_num * 4Byte + node_num * feature_dim * 4Byte + the num of node with label * 4Byte)
	- `DGI/Unsupervised GraphSage`: edge, node feature, the model size is: (edge_num * 2 * 8Byte + node_num * feature_dim * 4Byte)
	- `Semi Bipartite GraphSage`: edge * 2, user feature, item feature, user label, the model size is: (edge_num * 2 * 8Byte * 2 + user_num * user_feature_dim * 4Byte + item_num * item_feature_dim * 4Byte + user_label_num * 4Byte)
	- `Semi Heterogeneous GraphSage`: edge * 2, user feature, item feature, user label, item type, the model size is: (edge_num * 2 * 8Byte * 2 + user_num * user_feature_dim * 4Byte + item_num * item_feature_dim * 4Byte + user_label_num * 4Byte + edge_num * 4Byte * 2)
	- `Unsupervised GraphSage`: edge, node feature, the model size is: (edge_num * 2 * 8Byte + node_num * feature_dim * 4Byte)
	- `Unsupervised Bipartite GraphSage`: edge * 2, user feature, item feature, the model size is: (edge_num * 2 * 8Byte * 2 + user_num * user_feature_dim * 4Byte + item_num * item_feature_dim * 4Byte)
	- `IGMC`: edge with type, user feature, item feature, themodel size is: (edge_num * 2 * 8Byte + edge_num * 4Byte + user_num * user_feature_dim * 4Byte + item_num * item_feature_num * 4Byte)

- Spark Executor Resource: the configuration of Spark resources is mainly considered from the aspect of training data(Edge data is usually saved on Spark Executor), and it is best to save 2 times the input data. If the memory is tight, 1x is acceptable, but it will be relatively slow. For example, a 10 billion edge set is about 200G in size, and a 30G * 20 configuration is sufficient. 

##### High-Sparse data(data has field)
Only support for four algorithms, such as: `Semi GraphSage, Semi Bipartite GraphSage, HGAT and HAN`.
Resources of Angel PS and Spark Executor is similar to Dense data, the only difference is that there is low-dimension embedding matrix for high-sparse data.
low-dimension embedding: input_embedding_dim * slots * input_dim * 4Byte
- input_embedding_dim: the dimension of embedding, which is usually low, such as:8; (As for Bipartite GNN or Heterogeneous GNN, the parameter is input_user_embedding_dim or input_item_embedding_dim)
- slots: related to the optimizer, the default optimizer adam, slots = 3
- input_dim: the dim of input feature; (As for Bipartite GNN or Heterogeneous GNN, the parameter is input_user_dim or input_item_dim, which is the dim of user's feature or item's feature)
- input_field_num: the number of features with value; (As for Bipartite GNN or Heterogeneous GNN, the parameter is input_user_field_num or input_item_field_num, which is the number of user's feature with value or item's feature with value)

Angel PS Resource:in order to ensure that Angel does not hang up, it is necessary to configure memory that is about twice the size of the model.
	- `Semi GraphSage`: edge, node feature, node label, embedding matrix, the model size is: (edge_num * 2 * 8Byte + node_num * `field_num` * 4Byte + the num of node with label * 4Byte + `input_embedding_dim * slots * input_dim * 4Byte`)
	- `Semi Bipartite GraphSage/HGAT`: edge * 2, user feature, item feature, user label, user embedding matrix, item embedding matrix, the model size is: (edge_num * 2 * 8Byte * 2 + user_num * `user_field_num` * 4Byte + item_num * `item_field_num` * 4Byte + user_label_num * 4Byte+ `user_embedding_dim * slots * user_feature_dim * 4Byte + item_embedding_dim * slots * item_feature_dim * 4Byte`)
	

Spark Executor Resource:the configuration of Spark resources is mainly considered from the aspect of training data(Edge data is usually saved on Spark Executor), and it is best to save 2 times the input data. If the memory is tight, 1x is acceptable, but it will be relatively slow. For example, a 10 billion edge set is about 200G in size, and a 30G * 20 configuration is sufficient. 


### Example of GraphSage

Here we give an example of how to run GraphSage algorithm beyond Pytorch on Angel.

1. **Generate pytorch sciprt model**
    First, go to directory of python/graph and execute the following command:
    ```$xslt
    python graphsage.py --input_dim 1433 --hidden_dim 128 --output_dim 7 --output_file graphsage_cora.pt
    ```
    This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of graphsage. After that, you will obtain a model file named "graphsage_cora.pt". Here we use the Cora dataset as an example, where the feature dimension for each node is 1433 with 7 different classes.

2. **Preparing input data**
    There are three inputs required for graphsage, including the edge table, the feature table and the label table.

    The edge table is a file or directory which exists on hdfs. Each line of the file is an edge composed with a source node and a destination node seperated by space/comma/tab. Each node is encoded with a Long type numeric.

    The feature table is a file or directory from hdfs. Each line specifies the feature of one node. The format can be sparse or dense. 

    For sparse format, each line is formated as follows:
    ```
    node\tf1:v1 f2:v2 f3:v3
    ```
    The separator between ``node`` and ``features`` is tab while space is used as separator between different feature indices.

    For dense format, it is:
    ```
    node\tv1 v2 v3
    ```

    The label table contains a set of node-label pairs. Since graphsage is a semi-supervised model, the label table may only contain a small set of node-label pairs. Each line of the label file is a node-label pair where space is used as the separator between node and label.

    Note that, each node contained in the edge table should has a feature line in the feature table file.

2. **Submit model to cluster**
    After obtaining the model file and the inputs, we can submit a task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).
    ```$xslt
    source ./spark-on-angel-env.sh  
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
          --queue $queue \
          --name "graphsage-angel" \
          --jars $SONA_SPARK_JARS  \
          --archives angel_libtorch.zip#torch\  #path to c++ library files
          --files graphsage_cora.pt \   #path to pytorch script model
          --driver-memory 5g \
          --num-executors 5 \
          --executor-cores 1 \
          --executor-memory 5g \
          --class com.tencent.angel.pytorch.example.supervised.cluster.GraphSageExample \
          ./pytorch-on-angel-${VERSION}.jar \   # jar from Compiling java submodule
          edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath\
          torchModelPath:graphsage_cora.pt featureDim:1433 stepSize:0.01\
          optimizer:adam numEpoch:10 testRatio:0.5\
          numPartitions:50 format:sparse samples:10 batchSize:128\
          predictOutputPath:$predictOutputPath embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
          actionType:train numBatchInit:5
    ```
    Here we give a short description for the parameters in the submit script. 

    - edgePath: the input path (hdfs) of edge table
    - featurePath: the input path (hdfs) of feature table
    - labelPath: the input path (hdfs) of label table
    - torchModelPath: the name of the model file, graphsage_cora.pt in this example
    - featureDim: the dimension for the feature for each node, which should be equal with the number when generate the model file
    - stepSize: the learning rate when training
    - optimizer: adam/momentum/sgd/adagrad
    - numEpoch: number of epoches you want to run 
    - testRatio: use how many nodes from the label file for testing
    - numPartitions: partition the data into how many partitions
    - format: should be sparse/dense
    - samples: the number of samples when sampling neighbors in graphsage
    - batchSize: batch size for each optimizing step
    - predictOutputPath: hdfs path to save the predict label for all nodes in the graph, set it if you need the label
    - embeddingPath: hdfs path to save the embedding for all nodes in the graph, set it if you need the embedding vectors
    - outputModelPath: hdfs path to save the training model file, which is also a torch model pt file, set it if you want to do predicting or incremental training in the next step
    - actionType: should be train/predict
    - numBatchInit: we use a mini-batch way when initializing features and network structures on parameter servers. this parameter determines how many batches we uses in this step. 

    **Notes:**
    - The model file, graphsage_cora.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need use ``--files`` to upload the model file.


### Example of DGI

Here we give an example of how to run DGI algorithm beyond Pytorch on Angel.

1. **Generate pytorch sciprt model**
    First, go to directory of python/graph and execute the following command:
    ```$xslt
    python dgi2.py --input_dim 1433 --hidden_dim 128 --output_dim 128 --output_file dgi_cora.pt
    ```
    This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of dgi. After that, you will obtain a model file named "dgi_cora.pt". Here we use the Cora dataset as an example, where the feature dimension for each node is 1433 with 7 different classes.

2. **Preparing input data**
    There are two inputs required for dgi, including the edge table and the feature table.

    The edge table is a file or directory which exists on hdfs. Each line of the file is an edge composed with a source node and a destination node seperated by space/comma/tab. Each node is encoded with a Long type numeric.

    The feature table is a file or directory from hdfs. Each line specifies the feature of one node. The format can be sparse or dense. 

    For sparse format, each line is formated as follows:
    ```
    node\tf1:v1 f2:v2 f3:v3
    ```
    The separator between ``node`` and ``features`` is tab while space is used as separator between different feature indices.

    For dense format, it is:
    ```
    node\tv1 v2 v3
    ```

    Note that, each node contained in the edge table should has a feature line in the feature table file.

2. **Submit model to cluster**
    After obtaining the model file and the inputs, we can submit a task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).
    ```$xslt
    source ./spark-on-angel-env.sh  
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
          --queue $queue \
          --name "dgi-angel" \
          --jars $SONA_SPARK_JARS  \
          --archives angel_libtorch.zip#torch\  #path to c++ library files
          --files dgi_cora.pt \   #path to pytorch script model
          --driver-memory 5g \
          --num-executors 5 \
          --executor-cores 1 \
          --executor-memory 5g \
          --class com.tencent.angel.pytorch.example.unsupervised.cluster.DGIExample \
          ./pytorch-on-angel-${VERSION}.jar \   # jar from Compiling java submodule
          edgePath:$edgePath featurePath:$featurePath\
          torchModelPath:dgi_cora.pt featureDim:1433 stepSize:0.01\
          optimizer:adam numEpoch:10 \
          numPartitions:50 format:sparse samples:10 batchSize:128\
          embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
          actionType:train numBatchInit:5
    ```
    Here we give a short description for the parameters in the submit script. 

    - edgePath: the input path (hdfs) of edge table
    - featurePath: the input path (hdfs) of feature table
    - torchModelPath: the name of the model file, graphsage_cora.pt in this example
    - featureDim: the dimension for the feature for each node, which should be equal with the number when generate the model file
    - stepSize: the learning rate when training
    - optimizer: adam/momentum/sgd/adagrad
    - numEpoch: number of epoches you want to run 
    - numPartitions: partition the data into how many partitions
    - format: should be sparse/dense
    - samples: the number of samples when sampling neighbors in dgi
    - batchSize: batch size for each optimizing step
    - embeddingPath: hdfs path to save the embedding for all nodes in the graph, set it if you need the embedding vectors
    - outputModelPath: hdfs path to save the training model file, which is also a torch model pt file, set it if you want to do predicting or incremental training in the next step
    - actionType: should be train/predict
    - numBatchInit: we use a mini-batch way when initializing features and network structures on parameter servers. this parameter determines how many batches we uses in this step. 

    **Notes:**
    - The model file, dgi_cora.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need use ``--files`` to upload the model file.


### Example of Relation GCN (RGCN)

Relation GCN is semi-supervised graph convolution network which can utilize the types of edges.
The difference between RGCN and GCN is that each edge can has different types.

Here we give an example of using RGCN over pytorch on angel.


1. **Generate pytorch sciprt model**
    First, go to directory of python/graph and execute the following command:
    ```$xslt
    python rgcn.py --input_dim 32 --hidden_dim 16 --n_class 2 --output_file rgcn_mutag.pt --n_relations 46 --n_bases 30
    ```
    This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of rgcn. After that, you will obtain a model file named "rgcn_mutag.pt". Where n_relations is the number of types for edges and n_bases is a parameter of RGCN to avoid overfitting.

2. **Preparing input data**
    There are three inputs required for graphsage, including the edge table, the feature table and the label table.

    RGCN also requires an edge file, a feature file and a label file, similar to graphsage. The difference is that each entry in the edge file contains three elements, including a source node, a destination node and an edge type. For example:
	```
	src dst type
	```
	The src and dst is a Long numeric while the type is an Integer numeric.

    The feature table is a file or directory from hdfs. Each line specifies the feature of one node. The format can be sparse or dense. 

    For sparse format, each line is formated as follows:
    ```
    node\tf1:v1 f2:v2 f3:v3
    ```
    The separator between ``node`` and ``features`` is tab while space is used as separator between different feature indices.

    For dense format, it is:
    ```
    node\tv1 v2 v3
    ```

    The label table contains a set of node-label pairs. Since rgcn is a semi-supervised model, the label table may only contain a small set of node-label pairs. Each line of the label file is a node-label pair where space is used as the separator between node and label.

    Note that, each node contained in the edge table should has a feature line in the feature table file.

2. **Submit model to cluster**
    After obtaining the model file and the inputs, we can submit a task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).
    ```$xslt
    source ./spark-on-angel-env.sh  
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
          --queue $queue \
          --name "rgcn-angel" \
          --jars $SONA_SPARK_JARS  \
          --archives angel_libtorch.zip#torch\  #path to c++ library files
          --files rgcn_mutag.pt \   #path to pytorch script model
          --driver-memory 5g \
          --num-executors 5 \
          --executor-cores 1 \
          --executor-memory 5g \
          --class com.tencent.angel.pytorch.example.supervised.cluster.RGCNExample \
          ./pytorch-on-angel-${VERSION}.jar \   # jar from Compiling java submodule
          edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath\
          torchModelPath:rgcn_mutag.pt featureDim:32 stepSize:0.01\
          optimizer:adam numEpoch:10 testRatio:0.5\
          numPartitions:50 format:sparse samples:10 batchSize:128\
          predictOutputPath:$predictOutputPath embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
          actionType:train numBatchInit:5
    ```
    Here we give a short description for the parameters in the submit script. 

    - edgePath: the input path (hdfs) of edge table, which contains src, dst and type
    - featurePath: the input path (hdfs) of feature table
    - labelPath: the input path (hdfs) of label table
    - torchModelPath: the name of the model file, graphsage_cora.pt in this example
    - featureDim: the dimension for the feature for each node, which should be equal with the number when generate the model file
    - stepSize: the learning rate when training
    - optimizer: adam/momentum/sgd/adagrad
    - numEpoch: number of epoches you want to run 
    - testRatio: use how many nodes from the label file for testing
    - numPartitions: partition the data into how many partitions
    - format: should be sparse/dense
    - samples: the number of samples when sampling neighbors in rgcn
    - batchSize: batch size for each optimizing step
    - predictOutputPath: hdfs path to save the predict label for all nodes in the graph, set it if you need the label
    - embeddingPath: hdfs path to save the embedding for all nodes in the graph, set it if you need the embedding vectors
    - outputModelPath: hdfs path to save the training model file, which is also a torch model pt file, set it if you want to do predicting or incremental training in the next step
    - actionType: should be train/predict
    - numBatchInit: we use a mini-batch way when initializing features and network structures on parameter servers. this parameter determines how many batches we uses in this step. 

    **Notes:**
    - The model file, rgcn_mutag.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need use ``--files`` to upload the model file.



### Example of EdgeProp

[EdgeProp](https://arxiv.org/abs/1906.05546) is an end-to-end Graph Convolution Network (GCN)-based algorithm to learn the embeddings of the nodes and edges of a large-scale time-evolving graph. It consider not only node information and also edge side information. 

Here we give an example of using EdgeProp over pytorch on angel.


1. **Generate pytorch sciprt model**
    First, go to directory of python/graph and execute the following command:
    ```$xslt
    python edgeprop.py --input_dim 23 --edge_input_dim 7 --hidden_dim 128 --output_dim 7 --output_file edgeprop_eth.pt
    ```
    This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of edgeProp. After that, you will obtain a model file named "edgeprop_eth.pt". Where edge_input_dim is the dimension of edge feature, other parameters are same as GraphSAGE.

2. **Preparing input data**
    There are three inputs required for graphsage, including the edge table, the feature table and the label table.

    RGCN also requires an edge file, a feature file and a label file, similar to graphsage. The difference is that each entry in the edge file contains three elements, including a source node, a destination node and an edge feature. For example:
	```
	src\tdst\tedge_feature
	```
	The src and dst is a Long numeric while the edge_feature is a string, which can be dense or libsvm.

    The feature table is a file or directory from hdfs. Each line specifies the feature of one node. The format can be sparse or dense. 

    For sparse format, each line is formated as follows:
    ```
    node\tf1:v1 f2:v2 f3:v3
    ```
    The separator between ``node`` and ``features`` is tab while space is used as separator between different feature indices.

    For dense format, it is:
    ```
    node\tv1 v2 v3
    ```

    The label table contains a set of node-label pairs. Since rgcn is a semi-supervised model, the label table may only contain a small set of node-label pairs. Each line of the label file is a node-label pair where space is used as the separator between node and label.

    Note that, each node contained in the edge table should has a feature line in the feature table file.

2. **Submit model to cluster**
    After obtaining the model file and the inputs, we can submit a task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).
    ```$xslt
    source ./spark-on-angel-env.sh  
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
          --queue $queue \
          --name "rgcn-angel" \
          --jars $SONA_SPARK_JARS  \
          --archives angel_libtorch.zip#torch\  #path to c++ library files
          --files edgeprop_eth.pt \   #path to pytorch script model
          --driver-memory 5g \
          --num-executors 5 \
          --executor-cores 1 \
          --executor-memory 5g \
          --class com.tencent.angel.pytorch.example.supervised.cluster.EdgePropGCNExample \
          ./pytorch-on-angel-${VERSION}.jar \   # jar from Compiling java submodule
          edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath\
          torchModelPath:edgeprop_eth_1.5.pt featureDim:23 edgeFeatureDim:7 stepSize:0.01\
          optimizer:adam numEpoch:10 testRatio:0.5\
          numPartitions:50 format:sparse samples:10 batchSize:128\
          predictOutputPath:$predictOutputPath embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
          actionType:train numBatchInit:5
    ```
    Here we give a short description for the parameters in the submit script. 

    - edgePath: the input path (hdfs) of edge table, which contains src, dst and edge feature
    - featurePath: the input path (hdfs) of feature table
    - labelPath: the input path (hdfs) of label table
    - torchModelPath: the name of the model file, edgeprop_eth.pt in this example
    - featureDim: the dimension for the feature for each node, which should be equal with the number when generate the model file
    - edgeFeatureDim: the dimension for the feature of edge, which should be equal with the number when generate the model file
    - stepSize: the learning rate when training
    - optimizer: adam/momentum/sgd/adagrad
    - numEpoch: number of epoches you want to run 
    - testRatio: use how many nodes from the label file for testing
    - numPartitions: partition the data into how many partitions
    - format: should be sparse/dense
    - samples: the number of samples when sampling neighbors in edgeProp
    - batchSize: batch size for each optimizing step
    - predictOutputPath: hdfs path to save the predict label for all nodes in the graph, set it if you need the label
    - embeddingPath: hdfs path to save the embedding for all nodes in the graph, set it if you need the embedding vectors
    - outputModelPath: hdfs path to save the training model file, which is also a torch model pt file, set it if you want to do predicting or incremental training in the next step
    - actionType: should be train/predict
    - numBatchInit: we use a mini-batch way when initializing features and network structures on parameter servers. this parameter determines how many batches we uses in this step. 

    **Notes:**
    - The model file, rgcn_mutag.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need use ``--files`` to upload the model file.


### Example of GAT

Here we give an example of how to run GAT algorithm beyond Pytorch on Angel.

1. **Generate pytorch sciprt model**
    First, go to directory of python/graph and execute the following command:
    ```$xslt
    python gat.py --input_dim 32 --hidden_dim 128 --output_dim 11 --output_file gat_am.pt
    ```
    This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of GAT. After that, you will obtain a model file named "gat_am.pt". Here we use the am dataset as an example, where the feature dimension for each node is 32 with 11 different classes.

2. **Preparing input data**
    There are three inputs required for graphsage, including the edge table, the feature table and the label table.

    The edge table is a file or directory which exists on hdfs. Each line of the file is an edge composed with a source node and a destination node seperated by space/comma/tab. Each node is encoded with a Long type numeric.

    The feature table is a file or directory from hdfs. Each line specifies the feature of one node. The format can be sparse or dense. 

    For sparse format, each line is formated as follows:
    ```
    node\tf1:v1 f2:v2 f3:v3
    ```
    The separator between ``node`` and ``features`` is tab while space is used as separator between different feature indices.

    For dense format, it is:
    ```
    node\tv1 v2 v3
    ```

    The label table contains a set of node-label pairs. Since graphsage is a semi-supervised model, the label table may only contain a small set of node-label pairs. Each line of the label file is a node-label pair where space is used as the separator between node and label.

    Note that, each node contained in the edge table should has a feature line in the feature table file.

2. **Submit model to cluster**
    After obtaining the model file and the inputs, we can submit a task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).
    ```$xslt
    source ./spark-on-angel-env.sh  
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
          --queue $queue \
          --name "graphsage-angel" \
          --jars $SONA_SPARK_JARS  \
          --archives angel_libtorch.zip#torch\  #path to c++ library files
          --files gat_am.pt \   #path to pytorch script model
          --driver-memory 5g \
          --num-executors 5 \
          --executor-cores 1 \
          --executor-memory 5g \
          --class com.tencent.angel.pytorch.example.supervised.cluster.GATExample \
          ./pytorch-on-angel-${VERSION}.jar \   # jar from Compiling java submodule
          edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath\
          torchModelPath:gat_am.pt featureDim:32 stepSize:0.01\
          optimizer:adam numEpoch:10 testRatio:0.5\
          numPartitions:50 format:sparse samples:10 batchSize:128\
          predictOutputPath:$predictOutputPath embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
          actionType:train numBatchInit:5
    ```
    Here we give a short description for the parameters in the submit script. 

    - edgePath: the input path (hdfs) of edge table
    - featurePath: the input path (hdfs) of feature table
    - labelPath: the input path (hdfs) of label table
    - torchModelPath: the name of the model file, gat_am.pt in this example
    - featureDim: the dimension for the feature for each node, which should be equal with the number when generate the model file
    - stepSize: the learning rate when training
    - optimizer: adam/momentum/sgd/adagrad
    - numEpoch: number of epoches you want to run 
    - testRatio: use how many nodes from the label file for testing
    - numPartitions: partition the data into how many partitions
    - format: should be sparse/dense
    - samples: the number of samples when sampling neighbors in graphsage
    - batchSize: batch size for each optimizing step
    - predictOutputPath: hdfs path to save the predict label for all nodes in the graph, set it if you need the label
    - embeddingPath: hdfs path to save the embedding for all nodes in the graph, set it if you need the embedding vectors
    - outputModelPath: hdfs path to save the training model file, which is also a torch model pt file, set it if you want to do predicting or incremental training in the next step
    - actionType: should be train/predict
    - numBatchInit: we use a mini-batch way when initializing features and network structures on parameter servers. this parameter determines how many batches we uses in this step. 

    **Notes:**
    - The model file, gat_am.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need use ``--files`` to upload the model file.



### Example of HAN 

[HAN](https://arxiv.org/pdf/1903.07293.pdf) is a semi-supervised graph convolution network for heterogeneous graph.  In order to capture the heterogeneous information, HAN defined two different attentions: node-level and semantic level. Here a simplified version of HAN is implemented, which accepts bipartite graph in the form of "user-item", where item nodes could have multiple types. In another words, the input graph has multiple meta-paths in the form of "user-item-user".
HAN classifies user nodes, and outputs their embeddings if needed.


Here we give an example of using HAN over pytorch on angel.


1. **Generate pytorch sciprt model**
    First, go to directory of python/graph and execute the following command:
    ```$xslt
    python semi_han.py --m 64 --input_dim 32 --hidden_dim 16 --output_dim 2 --item_types 5  --output_file han.pt
    ```
    This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of han. After that, you will obtain a model file named "han.pt". 

2. **Preparing input data**
    There are three inputs required for han, including the edge table, the feature table and the label table.

    HAN requires an edge file which contains three columns including the source node column, the destination column and the node type column. The third column indicates the destination nodes' types, each type indicates a meta-path of "A-B-A". For example:
	```
	src dst type
	```
	The src and dst is a Long numeric while the type is an Integer numeric.

    The feature table is a file or directory from hdfs. Each line specifies the feature of one node. The format can be sparse or dense. 

    For sparse format, each line is formated as follows:
    ```
    node\tf1:v1 f2:v2 f3:v3
    ```
    The separator between ``node`` and ``features`` is tab while space is used as separator between different feature indices.

    For dense format, it is:
    ```
    node\tv1 v2 v3
    ```

    The label table contains a set of node-label pairs. Since han is a semi-supervised model, the label table may only contain a small set of node-label pairs. Each line of the label file is a node-label pair where space is used as the separator between node and label.

    Note that, each node contained in the edge table should has a feature line in the feature table file.

2. **Submit model to cluster**
    After obtaining the model file and the inputs, we can submit a task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).
    ```$xslt
    source ./spark-on-angel-env.sh  
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
          --queue $queue \
          --name "rgcn-angel" \
          --jars $SONA_SPARK_JARS  \
          --archives angel_libtorch.zip#torch\  #path to c++ library files
          --files rgcn_mutag.pt \   #path to pytorch script model
          --driver-memory 5g \
          --num-executors 5 \
          --executor-cores 1 \
          --executor-memory 5g \
          --class com.tencent.angel.pytorch.example.supervised.clusterHANExample \
          ./pytorch-on-angel-${VERSION}.jar \   # jar from Compiling java submodule
          edgePath:$edgePath userFeaturePath:$featurePath labelPath:$labelPath\
          torchModelPath:han.pt featureDim:32 userFeatureDim:32 temTypes:5 stepSize:0.01\
          optimizer:adam numEpoch:10 testRatio:0.5\
          numPartitions:50 format:sparse samples:10 batchSize:128\
          predictOutputPath:$predictOutputPath embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
          actionType:train numBatchInit:5
    ```
    Here we give a short description for the parameters in the submit script. 

    - edgePath: the input path (hdfs) of edge table, which contains src, dst and type
    - userFeaturePath: the input path (hdfs) of feature table
    - labelPath: the input path (hdfs) of label table
    - torchModelPath: the name of the model file, graphsage_cora.pt in this example
    - featureDim: the dimension for the feature for each node, which should be equal with the number when generate the model file
    - itemTypes: types of item nodes, which is also the num of meta-paths
    - stepSize: the learning rate when training
    - optimizer: adam/momentum/sgd/adagrad
    - numEpoch: number of epoches you want to run 
    - testRatio: use how many nodes from the label file for testing
    - numPartitions: partition the data into how many partitions
    - format: should be sparse/dense
    - samples: the number of samples when sampling neighbors in rgcn
    - batchSize: batch size for each optimizing step
    - predictOutputPath: hdfs path to save the predict label for all nodes in the graph, set it if you need the label
    - embeddingPath: hdfs path to save the embedding for all nodes in the graph, set it if you need the embedding vectors
    - outputModelPath: hdfs path to save the training model file, which is also a torch model pt file, set it if you want to do predicting or incremental training in the next step
    - actionType: should be train/predict
    - numBatchInit: we use a mini-batch way when initializing features and network structures on parameter servers. this parameter determines how many batches we uses in this step. 

    **Notes:**
    - The model file, rgcn_mutag.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need use ``--files`` to upload the model file.







### FAQ

1. If you want to use GAT or HGAT, pytorch >= v1.5.0.
2. If you found loss is NAN or does not converge, you can decrease the learning rate, such as: 0.001,0.0001 or lower.
3. If you encounter the error `file not found model.json`, please check the version of model.pt and the version of pytorch, whether tem are matched.
4. If you encounter the error `java.lang.UnsupportedOperationException: empty collection`, please check whether the input data is empty.
5. If you encounter the error `ERROR AngelYarnClient: submit application to yarn failed.`, ps did not apply for resources, change another cluster or try later.
6. If you encounter the error `java.lang.UnsatisfiedLinkError: no torch_angel in java.library.path`, please check whether the torch path is correct.