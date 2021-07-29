## GNN Algorithm Instructions

>Pytorch on Angel provides the ability to run graph convolution network algorithm. We follow [Pytorch-Geometric](https://github.com/rusty1s/pytorch_geometric) to define the graph convolution networks while using the parameter server of Angel to store the network structure and features of nodes.


### Introduction
#### How to predict
1. change the input data path and output path
2. change **actionType** to `predict`
3. you can get hdfs://modelPath/xx.pt to local, then use it as training; Or you can use the hdfs path, and set `--files hdfs://modelPath/xx.pt`, in this way the `torchModelPath` can be remove


#### How to train incrementally
1. change the input data path and output path, or you can use the same data to train incrementally
2. set **actionType** as `train` 
3. you can get hdfs://modelPath/xx.pt to local, then use it as training; Or you can use the hdfs path, and set `--files hdfs://modelPath/xx.pt`, in this way the `torchModelPath` can be remove

#### How to calculate the resource

##### Dense data or low dimentition sparse data (dim less than 2000 generally)
- Angel PS resource: in order to ensure that Angel does not hang up, it is necessary to configure memory that is about twice the size of the model. The formula for calculating the size of the graph feature is: node_num * feature_dim * 4Byte, such as: 1kw nodes, feature_dim is 100, the size of graph feature is about 4G, then set ps.instances=3, ps.memory=4G is reasonable(total memory is 2~3 times of data saved in ps.). Of course, this is only a simple case. In many algorithms, there are many types of data saved in ps, such as: edge, node feature, node label etc.
	- **Semi GraphSage/GAT**: edge, node feature, node label, the model size is: (edge\_num * 2 * 8Byte + node\_num * feature\_dim * 4Byte + the num of node with label * 4Byte)
	- **RGCN/HAN**: edge with type(dst type), node feature, node label, the model size is: (edge\_num * 2 * 8Byte + edge\_num * 4Byte + node\_num * feature\_dim * 4Byte + the num of node with label * 4Byte)
	- **DGI/Unsupervised GraphSage**: edge, node feature, the model size is: (edge\_num * 2 * 8Byte + node\_num * feature\_dim * 4Byte)
	- **Semi Bipartite GraphSage**: edge * 2, user feature, item feature, user label, the model size is: (edge\_num * 2 * 8Byte * 2 + user\_num * user\_feature_dim * 4Byte + item\_num * item\_feature\_dim * 4Byte + user\_label\_num * 4Byte)
	- **Semi Heterogeneous GraphSage**: edge * 2, user feature, item feature, user label, item type, the model size is: (edge\_num * 2 * 8Byte * 2 + user\_num * user\_feature\_dim * 4Byte + item_num * item\_feature\_dim * 4Byte + user\_label\_num * 4Byte + edge_num * 4Byte * 2)
	- **Unsupervised GraphSage**: edge, node feature, the model size is: (edge\_num * 2 * 8Byte + node\_num * feature\_dim * 4Byte)
	- **Unsupervised Bipartite GraphSage**: edge * 2, user feature, item feature, the model size is: (edge\_num * 2 * 8Byte * 2 + user_num * user\_feature\_dim * 4Byte + item\_num * item_feature_dim * 4Byte)
	- **IGMC**: edge with type, user feature, item feature, themodel size is: (edge\_num * 2 * 8Byte + edge\_num * 4Byte + user\_num * user\_feature\_dim * 4Byte + item\_num * item_feature\_num * 4Byte)

- Spark Executor Resource: the configuration of Spark resources is mainly considered from the aspect of training data(Edge data is usually saved on Spark Executor), and it is best to save 2 times the input data. If the memory is tight, 1x is acceptable, but it will be relatively slow. For example, a 10 billion edge set is about 200G in size, and a 30G * 20 configuration is sufficient. 

##### High-Sparse data(data has field)
Only support for four algorithms, such as: **Semi GraphSage, Semi Bipartite GraphSage, HGAT and HAN**.
Resources of Angel PS and Spark Executor is similar to Dense data, the only difference is that there is low-dimension embedding matrix for high-sparse data.
low-dimension embedding: input\_embedding\_dim * slots * input\_dim * 4Byte

- `input_embedding_dim`: the dimension of embedding, which is usually low, such as:8; (As for Bipartite GNN or Heterogeneous GNN, the parameter is input\_user\_embedding\_dim or input\_item\_embedding\_dim)
- `slots`: related to the optimizer, the default optimizer adam, slots = 3
- `input_dim`: the dim of input feature; (As for Bipartite GNN or Heterogeneous GNN, the parameter is input\_user\_dim or input\_item\_dim, which is the dim of user's feature or item's feature)
- `input_field_num`: the number of features with value; (As for Bipartite GNN or Heterogeneous GNN, the parameter is input\_user\_field\_num or input\_item\_field\_num, which is the number of user's feature with value or item's feature with value)

- Angel PS Resource:in order to ensure that Angel does not hang up, it is necessary to configure memory that is about twice the size of the model.
	- **Semi GraphSage**: edge, node feature, node label, embedding matrix, the model size is: (edge\_num * 2 * 8Byte + node\_num * `field_num` * 4Byte + the num of node with label * 4Byte + `input_embedding_dim * slots * input_dim * 4Byte`)
	- **Semi Bipartite GraphSage/HGAT**: edge * 2, user feature, item feature, user label, user embedding matrix, item embedding matrix, the model size is: (edge\_num * 2 * 8Byte * 2 + user\_num * `user_field_num` * 4Byte + item\_num * `item_field_num` * 4Byte + user\_label\_num * 4Byte+ `user_embedding_dim * slots * user_feature_dim * 4Byte + item_embedding_dim * slots * item_feature_dim * 4Byte`)
	

- Spark Executor Resource:the configuration of Spark resources is mainly considered from the aspect of training data(Edge data is usually saved on Spark Executor), and it is best to save 2 times the input data. If the memory is tight, 1x is acceptable, but it will be relatively slow. For example, a 10 billion edge set is about 200G in size, and a 30G * 20 configuration is sufficient. 

### Public parameters

Property Name | Default | Meaning
---------------- | --------------- | ---------------
edgePath | "" | the input path (hdfs) of edge table
featurePath | "" | the input path (hdfs) of feature table
labelPath | "" | the input path (hdfs) of label table
torchModelPath | model.pt | the name of the model file, xx.pt in this example
predictOutputPath | "" | hdfs path to save the predict label for all nodes in the graph, set it if you need the label
embeddingPath | "" | hdfs path to save the embedding for all nodes in the graph, set it if you need the embedding vectors
outputModelPath | "" | hdfs path to save the training model file, which is also a torch model pt file, set it if you want to do predicting or incremental training in the next step
userFeaturePath | "" | the input path (hdfs) of user feature table
itemFeaturePath | "" | the input path (hdfs) of item feature table
userEmbeddingPath | "" | hdfs path to save the embedding for all user nodes in the graph, set it if you need the embedding vectors
itemEmbeddingPath | "" | hdfs path to save the embedding for all item nodes in the graph, set it if you need the embedding vectors
featureEmbedInputPath | "" | the embedding matrix for features(contains user, item), set it if you need to increment train only when data is high-sparse

### Example of GraphSage

[GraphSage](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) generates embeddings by sampling and aggregating features from a node’s local neighborhood. Here we give an example of how to run GraphSage algorithm beyond Pytorch on Angel.

1. **Generate pytorch sciprt model**
    First, go to directory of python/graph and execute the following command:  
	**dense/low-sparse data**:  
    ```$xslt
    python graphsage.py --input_dim 1433 --hidden_dim 128 --output_dim 7 --output_file graphsage_cora.pt
    ```  
	**high-sparse data**:  
	```$xslt
    python graphsage.py --input_dim 32 --input_embedding_dim 8 --input_field_num 20 --encode one-hot --hidden_dim 128 --output_dim 7 --output_file graphsage_sparse.pt
    ```  
    This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of graphsage. After that, you will obtain a model file named "graphsage_cora.pt". Here we use the Cora dataset as an example, where the feature dimension for each node is 1433 with 7 different classes.  
	- `field_num`: the number of field num 
	- `input_embedding_dim`: the dimension of embedding
	- `encode`: the encode of feature, the default is `dense`, optional value:dense,one-hot,multi-hot

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
    
    **dense/low-sparse data**:   

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

	**high-sparse data**:   

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
          --files graphsage_sparse.pt \   #path to pytorch script model
          --driver-memory 5g \
          --num-executors 5 \
          --executor-cores 1 \
          --executor-memory 5g \
          --class com.tencent.angel.pytorch.example.supervised.cluster.GraphSageExample \
          ./pytorch-on-angel-${VERSION}.jar \   # jar from Compiling java submodule
          edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath\
          torchModelPath:graphsage_sparse.pt featureDim:1433 stepSize:0.01\
          optimizer:adam numEpoch:10 testRatio:0.5 fieldNum:20 featEmbedDim:8 \
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
    - `featureEmbedInputPath`:the embedding matrix for features(contains user, item), set it if you need to increment train only when data is high-sparse
    - `fieldNum`: the field num of user, the default is `-1`, set it if you need only when data is high-sparse
    - `featEmbedDim`: the dim of user embedding, the default is `-1`, set it if you need only when data is high-sparse
    - `fieldMultiHot`: whether the field is multi-hot(only support the last field is multi-hot), the default is `false`, set it if you need only when data is high-sparse
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
    - numLabels: the num of multi-label classification task if numLabels > 1, `the default is 1` for single-label classification
    - evals: eval method, the default is acc, the optional value: acc,binary_acc,auc,precision,f1,rmse,recall;`multi_auc` for `numLabels` > 1
    - periods: save pt model every few epochs
    - saveCheckpoint: whether checkpoint ps model after init parameters to ps
    - checkpointInterval: save ps model every few epochs
    - decay: the decay of learning ratio, the default is 0
    - validatePeriods: validate model every few epochs
    - useSharedSamples: whether reuse the samples to calculate the train acc to accelerate, the default is false
    
    **Notes:**
    - The model file, graphsage_cora.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need use ``--files`` to upload the model file.


### Example of DGI/Unsupervised GraphSage

Here we give an example of how to run DGI algorithm beyond Pytorch on Angel.

1. **Generate pytorch sciprt model**
    First, go to directory of python/graph and execute the following command:  
	**for DGI pt model**:  
    ```$xslt
    python dgi.py --input_dim 1433 --hidden_dim 128 --output_dim 128 --output_file dgi_cora.pt
    ```  
	**for Unsupervised GraphSage pt model**:  
    ```$xslt
    python unsupervised_graphsage.py --input_dim 1433 --hidden_dim 128 --output_dim 128 --output_file unsupervised_graphsage_cora.pt
    ```  
    This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of dgi. After that, you will obtain a model file named "dgi_cora.pt". Here we use the Cora dataset as an example, where the feature dimension for each node is 1433.

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
	The only difference between **DGI** and Unsupervised **GraphSage** is pt model, the submit scriptis same；
    
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
    - trainRatio: randomly sampling part of samples to train
    - embeddingPath: hdfs path to save the embedding for all nodes in the graph, set it if you need the embedding vectors
    - outputModelPath: hdfs path to save the training model file, which is also a torch model pt file, set it if you want to do predicting or incremental training in the next step
    - actionType: should be train/predict
    - numBatchInit: we use a mini-batch way when initializing features and network structures on parameter servers. this parameter determines how many batches we uses in this step
    - periods: save pt model every few epochs
    - saveCheckpoint: whether checkpoint ps model after init parameters to ps
    - checkpointInterval: save ps model every few epochs
    - decay: the decay of learning ratio, the default is 0. 

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
    This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of rgcn. After that, you will obtain a model file named "rgcn_mutag.pt". Where n_class is the number of classes, n_relations is the number of types for edges and n_bases is a parameter of RGCN to avoid overfitting.

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
    - numLabels: the num of multi-label classification task if numLabels > 1, `the default is 1` for single-label classification
    - evals: eval method, the default is acc, the optional value: acc,binary_acc,auc,precision,f1,rmse,recall;`multi_auc` for `numLabels` > 1
    - periods: save pt model every few epochs
    - saveCheckpoint: whether checkpoint ps model after init parameters to ps
    - checkpointInterval: save ps model every few epochs
    - validatePeriods: validate model every few epochs
    - useSharedSamples: whether reuse the samples to calculate the train acc to accelerate, the default is false

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
    This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of edgeProp. After that, you will obtain a model file named "edgeprop_eth.pt". Where edge\_input\_dim is the dimension of edge feature, other parameters are same as GraphSAGE.

2. **Preparing input data**
    There are three inputs required for graphsage, including the edge table, the feature table and the label table.

    EdgeProp also requires an edge file, a feature file and a label file, similar to graphsage. The difference is that each entry in the edge file contains three elements, including a source node, a destination node and an edge feature. For example:
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
          --name "edgeprop-angel" \
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
          torchModelPath:edgeprop_eth.pt featureDim:23 edgeFeatureDim:7 stepSize:0.01\
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
    - numLabels: the num of multi-label classification task if numLabels > 1, `the default is 1` for single-label classification
    - evals: eval method, the default is acc, the optional value: acc,binary_acc,auc,precision,f1,rmse,recall;`multi_auc` for `numLabels` > 1
    - periods: save pt model every few epochs
    - saveCheckpoint: whether checkpoint ps model after init parameters to ps
    - checkpointInterval: save ps model every few epochs
    - decay: the decay of learning ratio, the default is 0
    - validatePeriods: validate model every few epochs
    - useSharedSamples: whether reuse the samples to calculate the train acc to accelerate, the default is false

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
          --name "gat-angel" \
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
    - numLabels: the num of multi-label classification task if numLabels > 1, `the default is 1` for single-label classification
    - evals: eval method, the default is acc, the optional value: acc,binary_acc,auc,precision,f1,rmse,recall;`multi_auc` for `numLabels` > 1
    - periods: save pt model every few epochs
    - saveCheckpoint: whether checkpoint ps model after init parameters to ps
    - checkpointInterval: save ps model every few epochs
    - decay: the decay of learning ratio, the default is 0
    - validatePeriods: validate model every few epochs
    - useSharedSamples: whether reuse the samples to calculate the train acc to accelerate, the default is false
    
    **Notes:**
    - The model file, gat_am.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need use ``--files`` to upload the model file.



### Example of HAN 

[HAN](https://arxiv.org/pdf/1903.07293.pdf) is a semi-supervised graph convolution network for heterogeneous graph.  In order to capture the heterogeneous information, HAN defined two different attentions: node-level and semantic level. Here a simplified version of HAN is implemented, which accepts bipartite graph in the form of "user-item", where item nodes could have multiple types. In another words, the input graph has multiple meta-paths in the form of "user-item-user".
HAN classifies user nodes, and outputs their embeddings if needed.


Here we give an example of using HAN over pytorch on angel.


1. **Generate pytorch sciprt model**
    First, go to directory of python/graph and execute the following command:  
	**dense/low-sparse data**:  
    ```$xslt
    python semi_han.py --m 64 --input_dim 32 --hidden_dim 16 --output_dim 2 --item_types 5  --output_file han.pt
    ```  
	**high-sparse data**:  
	```$xslt
    python semi_han.py --m 64 --input_dim 32 --input_embedding_dim 8 --input_field_num 20 --encode one-hot --hidden_dim 16 --output_dim 2 --item_types 5 --output_file han_sparse.pt
    ```  
    This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of han. After that, you will obtain a model file named "han.pt".
	- `input_embedding_dim`: the dimension of user embedding
	- `input_field_num`: the number of item field num
	- `encode`: the encode of feature, the default is `dense`, optional value:dense,one-hot,multi-hot 

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
	**dense /low-sparse data**:    
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
          --name "han-angel" \
          --jars $SONA_SPARK_JARS  \
          --archives angel_libtorch.zip#torch\  #path to c++ library files
          --files han.pt \   #path to pytorch script model
          --driver-memory 5g \
          --num-executors 5 \
          --executor-cores 1 \
          --executor-memory 5g \
          --class com.tencent.angel.pytorch.example.supervised.cluster.HANExample \
          ./pytorch-on-angel-${VERSION}.jar \   # jar from Compiling java submodule
          edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath\
          torchModelPath:han.pt featureDim:32 temTypes:5 stepSize:0.01\
          optimizer:adam numEpoch:10 testRatio:0.5\
          numPartitions:50 format:sparse samples:10 batchSize:128\
          predictOutputPath:$predictOutputPath embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
          actionType:train numBatchInit:5
    ```

	**high-sparse data**: 
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
          --name "han-angel" \
          --jars $SONA_SPARK_JARS  \
          --archives angel_libtorch.zip#torch\  #path to c++ library files
          --files han.pt \   #path to pytorch script model
          --driver-memory 5g \
          --num-executors 5 \
          --executor-cores 1 \
          --executor-memory 5g \
          --class com.tencent.angel.pytorch.example.supervised.cluster.HANExample \
          ./pytorch-on-angel-${VERSION}.jar \   # jar from Compiling java submodule
          edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath\
          torchModelPath:han.pt featureDim:32 temTypes:5 stepSize:0.01\
          optimizer:adam numEpoch:10 testRatio:0.5 fieldNum:20 featEmbedDim:8 \
          numPartitions:50 format:sparse samples:10 batchSize:128\
          predictOutputPath:$predictOutputPath embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
          actionType:train numBatchInit:5
    ```
    Here we give a short description for the parameters in the submit script. 

    - edgePath: the input path (hdfs) of edge table, which contains src, dst and type
    - featurePath: the input path (hdfs) of feature table
    - labelPath: the input path (hdfs) of label table
    - torchModelPath: the name of the model file, graphsage_cora.pt in this example
    - featureDim: the dimension for the feature for each user node, which should be equal with the number when generate the model file
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
    - embeddingPath: hdfs path to save the embedding for all user nodes in the graph, set it if you need the embedding vectors
    - outputModelPath: hdfs path to save the training model file, which is also a torch model pt file, set it if you want to do predicting or incremental training in the next step
    - actionType: should be train/predict
    - numBatchInit: we use a mini-batch way when initializing features and network structures on parameter servers. this parameter determines how many batches we uses in this step
    - numLabels: the num of multi-label classification task if numLabels > 1, `the default is 1` for single-label classification
    - evals: eval method, the default is acc, the optional value: acc,binary_acc,auc,precision,f1,rmse,recall;`multi_auc` for `numLabels` > 1
    - periods: save pt model every few epochs
    - saveCheckpoint: whether checkpoint ps model after init parameters to ps
    - checkpointInterval: save ps model every few epochs
    - decay: the decay of learning ratio, the default is 0
    - validatePeriods: validate model every few epochs. 

    **Notes:**
    - The model file, rgcn_mutag.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need use ``--files`` to upload the model file.


### Example of Semi Bipartite GraphSage 

[Semi Bipartite GraphSage](https://conferences.computer.org/icde/2020/pdfs/ICDE2020-5acyuqhpJ6L9P042wmjY1p/290300b677/290300b677.pdf) is a semi-supervised graph convolution network for Bipartite graph.  


Here we give an example of using Semi Bipartite GraphSage over pytorch on angel.


1. **Generate pytorch sciprt model**
    First, go to directory of python/graph and execute the following command:  
	**dense/low-sparse data**:  
    ```$xslt
    python semi_bipartite_graphsage.py --input_user_dim 2 --input_item_dim 19 --hidden_dim 128 --output_dim 2 --output_file semi_bipartite_graphsage.pt --task_type classification
    ```  
    **high-sparse data**:
	```$xslt
    python semi_bipartite_graphsage.py --input_user_dim 10 --input_item_dim 10 --hidden_dim 128 --output_dim 2 --output_file semi_bipartite_graphsage_sparse.pt --task_type classification --input_user_field_num 3 --input_item_field_num 3 --input_user_embedding_dim 8 --input_item_embedding_dim 16
    ```
    This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of Semi Bipartite GraphSage. After that, you will obtain a model file named "semi_bipartite_graphsage.pt". 
	- `input_user_field_num`: the number of user field num 
	- `input_item_field_num`: the number of item field num
	- `input_user_embedding_dim`: the dimension of user embedding
	- `input_item_embedding_dim`: the dimension of item embedding
	- `encode`: the encode of feature, the default is `dense`, optional value:dense,one-hot,multi-hot
	- `task_type`: the type of task, the default value is `classification`, optional value:classification or multi-label-classification(multi labels for one node)

2. **Preparing input data**
    There are three inputs required for han, including the edge table, the user feature table, the item feature table and the label table for user node.

    Semi Bipartite GraphSage requires an edge file which contains two columns including the source node column, the destination column. For example:
	```
	src dst
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
    **dense/low-sparse data**:   

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
          --name "semi_bipartite_graphsage-angel" \
          --jars $SONA_SPARK_JARS  \
          --archives angel_libtorch.zip#torch\  #path to c++ library files
          --files semi_bipartite_graphsage.pt \   #path to pytorch script model
          --driver-memory 5g \
          --num-executors 5 \
          --executor-cores 1 \
          --executor-memory 5g \
          --class com.tencent.angel.pytorch.example.supervised.cluster.BiGCNExample \
          ./pytorch-on-angel-${VERSION}.jar \   # jar from Compiling java submodule
          edgePath:$edgePath userFeaturePath:$featurePath labelPath:$labelPath\
          torchModelPath:semi_bipartite_graphsage.pt userFeatureDim:2 itemFeatureDim:19 stepSize:0.01\
          optimizer:adam numEpoch:10 testRatio:0.5\
          numPartitions:50 format:sparse userNumSamples:10 itemNumSamples:10 batchSize:128\
          predictOutputPath:$predictOutputPath userEmbeddingPath:$userEmbeddingPath outputModelPath:$outputModelPath\
          actionType:train numBatchInit:5
    ```  
	**high-sparse data**:   

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
          --name "semi_bipartite_graphsage-angel" \
          --jars $SONA_SPARK_JARS  \
          --archives angel_libtorch.zip#torch\  #path to c++ library files
          --files semi_bipartite_graphsage_sparse.pt \   #path to pytorch script model
          --driver-memory 5g \
          --num-executors 5 \
          --executor-cores 1 \
          --executor-memory 5g \
          --class com.tencent.angel.pytorch.example.supervised.cluster.BiGCNExample \
          ./pytorch-on-angel-${VERSION}.jar \   # jar from Compiling java submodule
          edgePath:$edgePath userFeaturePath:$featurePath labelPath:$labelPath\
          torchModelPath:semi_bipartite_graphsage_sparse.pt userFeatureDim:10 itemFeatureDim:10 stepSize:0.01\
          optimizer:adam numEpoch:10 testRatio:0.5 userFieldNum:3 itemFieldNum:3 userFeatEmbedDim:8 itemFeatEmbedDim:16\
          numPartitions:50 format:sparse userNumSamples:10 itemNumSamples:10 batchSize:128\
          predictOutputPath:$predictOutputPath userEmbeddingPath:$userEmbeddingPath outputModelPath:$outputModelPath\
          actionType:train numBatchInit:5
    ```
    Here we give a short description for the parameters in the submit script. 

    - edgePath: the input path (hdfs) of edge table, which contains src, dst and type
    - userFeaturePath: the input path (hdfs) of user feature table
    - itemFeaturePath: the input path (hdfs) of item feature table
    - labelPath: the input path (hdfs) of label table
    - testLabelPath: the input path (hdfs) of validate label table
    - torchModelPath: the name of the model file, semi_bipartite_graphsage.pt in this example
    - `userFeatureDim`: the dimension for the feature for each user node, which should be equal with the number when generate the model file
    - `itemFeatureDim`: the dimension for the feature for each item node, which should be equal with the number when generate the model file
    - `featureEmbedInputPath`:the embedding matrix for features(contains user, item), set it if you need to increment train only when data is high-sparse
    - `userFieldNum`: the field num of user, the default is `-1`, set it if you need only when data is high-sparse
    - `itemFieldNum`: the field num of item, the default is `-1`, set it if you need only when data is high-sparse
    - `userFeatEmbedDim`: the dim of user embedding, the default is `-1`, set it if you need only when data is high-sparse
    - `itemFeatEmbedDim`；the dim of item embedding, the default is `-1`, set it if you need only when data is high-sparse
    - `fieldMultiHot`: whether the field is multi-hot(only support the last field is multi-hot), the default is `false`, set it if you need only when data is high-sparse
    - stepSize: the learning rate when training
    - optimizer: adam/momentum/sgd/adagrad
    - numEpoch: number of epoches you want to run 
    - testRatio: use how many nodes from the label file for testing(if `testLabelPath` is set, the testRatio is invalid)
    - numPartitions: partition the data into how many partitions
    - psNumPartition: partition the data into how many partitions on ps
    - format: should be sparse/dense
    - userNumSamples: the number of samples of user neighbors
    - itemNumSamples: the number of samples of item neighbors
    - batchSize: batch size for each optimizing step
    - useBalancePartition:whether use balance partition, the default is `false`
    - predictOutputPath: hdfs path to save the predict label for all nodes in the graph, set it if you need the label
    - userEmbeddingPath: hdfs path to save the embedding for all user nodes in the graph, set it if you need the embedding vectors
    - outputModelPath: hdfs path to save the training model file, which is also a torch model pt file, set it if you want to do predicting or incremental training in the next step
    - actionType: should be train/predict
    - numBatchInit: we use a mini-batch way when initializing features and network structures on parameter servers. this parameter determines how many batches we uses in this step.
    - second: whether use second hop sampling
    - periods: save pt model every few epochs
    - saveCheckpoint: whether checkpoint ps model after init parameters to ps
    - checkpointInterval: save ps model every few epochs
    - decay: the decay of learning ratio, the default is 0
    - evals: eval method, the default is acc, the optional value: acc,binary_acc,auc,precision,f1,rmse,recall;`multi_auc` for `numLabels` > 1
    - validatePeriods: validate model every few epochs
    - useSharedSamples: whether reuse the samples to calculate the train acc to accelerate, the default is false
    - numLabels: the num of multi-label classification task if numLabels > 1, `the default is 1` for single-label classification. 

    **Notes:**
    - The model file, semi\_bipartite\_graphsage.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need use ``--files`` to upload the model file.


### Example of Unsupervised Bipartite GraphSage 

[Unsupervised Bipartite GraphSage](https://conferences.computer.org/icde/2020/pdfs/ICDE2020-5acyuqhpJ6L9P042wmjY1p/290300b677/290300b677.pdf) is a unsupervised graph convolution network for Bipartite graph.  


Here we give an example of using Unsupervised Bipartite GraphSage over pytorch on angel.


1. **Generate pytorch sciprt model**
    First, go to directory of python/graph and execute the following command:
    ```$xslt
    python unsupervised_bipartite_graphsage.py --input_user_dim 2 --input_item_dim 19 --hidden_dim 128 --output_dim 128 --output_file un_bipartite_graphsage.pt
    ```
    This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of Unsupervised Bipartite GraphSage. After that, you will obtain a model file named "un_bipartite_graphsage.pt". 

2. **Preparing input data**
    There are three inputs required for Unsupervised Bipartite GraphSage, including the edge table, the user feature table，and item feature table.

    Unsupervised Bipartite GraphSage requires an edge file which contains two columns including the source node column, the destination column. For example:
	```
	src dst
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
          --name "semi_bipartite_graphsage-angel" \
          --jars $SONA_SPARK_JARS  \
          --archives angel_libtorch.zip#torch\  #path to c++ library files
          --files unsupervised_bipartite_graphsage.pt \   #path to pytorch script model
          --driver-memory 5g \
          --num-executors 5 \
          --executor-cores 1 \
          --executor-memory 5g \
          --class com.tencent.angel.pytorch.example.unsupervised.cluster.BiGraphSageExample \
          ./pytorch-on-angel-${VERSION}.jar \   # jar from Compiling java submodule
          edgePath:$edgePath userFeaturePath:$featurePath labelPath:$labelPath\
          torchModelPath:unsupervised_bipartite_graphsage.pt userFeatureDim:2 itemFeatureDim:19 stepSize:0.01\
          optimizer:adam numEpoch:10\
          numPartitions:50 format:sparse userNumSamples:10 itemNumSamples:10 batchSize:128\
          predictOutputPath:$predictOutputPath userEmbeddingPath:$userEmbeddingPath itemEmbeddingPath:$itemEmbeddingPath outputModelPath:$outputModelPath\
          actionType:train numBatchInit:5
    ```
    Here we give a short description for the parameters in the submit script. 

    - edgePath: the input path (hdfs) of edge table, which contains src, dst and type
    - userFeaturePath: the input path (hdfs) of user feature table
    - itemFeaturePath: the input path (hdfs) of item feature table
    - torchModelPath: the name of the model file, unsupervised_bipartite_graphsage.pt in this example
    - userFeatureDim: the dimension for the feature for each user node, which should be equal with the number when generate the model file
    - itemFeatureDim: the dimension for the feature for each item node, which should be equal with the number when generate the model file
    - stepSize: the learning rate when training
    - optimizer: adam/momentum/sgd/adagrad
    - numEpoch: number of epoches you want to run 
    - trainRatio: randomly sampling part of samples to train
    - numPartitions: partition the data into how many partitions
    - psNumPartition: partition the data into how many partitions on ps
    - format: should be sparse/dense
    - userNumSamples: the number of samples of user neighbors
    - itemNumSamples: the number of samples of item neighbors
    - batchSize: batch size for each optimizing step
    - useBalancePartition:whether use balance partition, the default is `false`
    - predictOutputPath: hdfs path to save the predict label for all nodes in the graph, set it if you need the label
    - userEmbeddingPath: hdfs path to save the embedding for all user nodes in the graph, set it if you need the embedding vectors
    - itemEmbeddingPath: hdfs path to save the embedding for all item nodes in the graph, set it if you need the embedding vectors
    - outputModelPath: hdfs path to save the training model file, which is also a torch model pt file, set it if you want to do predicting or incremental training in the next step
    - actionType: should be train/predict
    - numBatchInit: we use a mini-batch way when initializing features and network structures on parameter servers. this parameter determines how many batches we uses in this step.
    - second: whether use second hop sampling
    - periods: save pt model every few epochs
    - saveCheckpoint: whether checkpoint ps model after init parameters to ps
    - checkpointInterval: save ps model every few epochs
    - decay: the decay of learning ratio, the default is 0
    - useSharedSamples: whether reuse the samples to calculate the train acc to accelerate, the default is false

    **Notes:**
    - The model file, unsupervised\_bipartite\_graphsage.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need use ``--files`` to upload the model file.


### Example of Unsupervised Heterogeneous Graph Attention Network(HGAT) 

[HGAT](https://arxiv.org/pdf/1903.07293.pdf) is a unsupervised graph attention convolution network for Bipartite graph.  


Here we give an example of using HGAT over pytorch on angel.


1. **Generate pytorch sciprt model**
    First, go to directory of python/graph and execute the following command:
    **dense/low-sparse data**:

    ```$xslt
    python unsupervised_heterogeneous_gat.py --input_user_dim 64 --input_item_dim 64 --hidden_dim 64 --output_dim 64 --output_file hgat_dense.pt --negative_size 32 --heads 2
    ```
	**high-sparse data**:
	
	```$xslt
    python unsupervised_heterogeneous_gat.py --input_user_dim 32 --input_item_dim 32 --hidden_dim 8 --output_dim 64 --output_file hgat_sparse.pt --input_user_field_num 4 --input_item_field_num 2 --input_user_embedding_dim 8 --input_item_embedding_dim 16 --negative_size 32 --heads 2 --encode multi-hot
    ```
    This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of Unsupervised Bipartite GraphSage. After that, you will obtain a model file named "hgat_dense.pt or hgat_sparse.pt". 
	- `negative_size`:multiples of positive samples, the default is 1
	- `heads`:the number of attention heads, the default is 1
	- `input_user_field_num`: the number of user field num 
	- `input_item_field_num`: the number of item field num
	- `input_user_embedding_dim`: the dimension of user embedding
	- `input_item_embedding_dim`: the dimension of item embedding
	- `encode`: the encode of feature, the default is `dense`, optional value:dense,one-hot,multi-hot

2. **Preparing input data**
    There are three inputs required for HGAT, including the edge table, the user feature table，and item feature table.

    HGAT requires an edge file which contains two columns including the source node column, the destination column. For example:
	```
	src dst
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

    Note that, each node contained in the edge table should has a feature line in the feature table file.

2. **Submit model to cluster**
    After obtaining the model file and the inputs, we can submit a task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).
	**dense/low-sparse data submit script**:
 
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
          --name "HGAT-angel" \
          --jars $SONA_SPARK_JARS  \
          --archives angel_libtorch.zip#torch\  #path to c++ library files
          --files hgat_dense.pt \   #path to pytorch script model
          --driver-memory 5g \
          --num-executors 5 \
          --executor-cores 1 \
          --executor-memory 5g \
          --class com.tencent.angel.pytorch.example.unsupervised.cluster.HGATExample \
          ./pytorch-on-angel-${VERSION}.jar \   # jar from Compiling java submodule
          edgePath:$edgePath userFeaturePath:$featurePath labelPath:$labelPath\
          torchModelPath:hgat_dense.pt userFeatureDim:64 itemFeatureDim:64 stepSize:0.0001 decay:0.001\
          optimizer:adam numEpoch:10 testRatio:0.5 \
          numPartitions:50 format:dense userNumSamples:5 itemNumSamples:5 batchSize:128\
          predictOutputPath:$predictOutputPath userEmbeddingPath:$userEmbeddingPath itemEmbeddingPath:$itemEmbeddingPath outputModelPath:$outputModelPath\
          actionType:train numBatchInit:5
    ```


	**high-sparse data submit script**:
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
          --name "HGAT-angel" \
          --jars $SONA_SPARK_JARS  \
          --archives angel_libtorch.zip#torch\  #path to c++ library files
          --files hgat_sparse.pt \   #path to pytorch script model
          --driver-memory 5g \
          --num-executors 5 \
          --executor-cores 1 \
          --executor-memory 5g \
          --class com.tencent.angel.pytorch.example.unsupervised.cluster.HGATExample \
          ./pytorch-on-angel-${VERSION}.jar \   # jar from Compiling java submodule
          edgePath:$edgePath userFeaturePath:$featurePath labelPath:$labelPath\
          torchModelPath:hgat_sparse.pt userFeatureDim:25000000 itemFeatureDim:80000 stepSize:0.0001 decay:0.001 fieldMultiHot:true \
          optimizer:adam numEpoch:10 testRatio:0.5 userFieldNum:4 itemFieldNum:2 userFeatEmbedDim:8 itemFeatEmbedDim:16\
          numPartitions:50 format:sparse userNumSamples:5 itemNumSamples:5 batchSize:128\
          predictOutputPath:$predictOutputPath userEmbeddingPath:$userEmbeddingPath itemEmbeddingPath:$itemEmbeddingPath outputModelPath:$outputModelPath\
          actionType:train numBatchInit:5
    ```
    Here we give a short description for the parameters in the submit script. 

    - edgePath: the input path (hdfs) of edge table, which contains src, dst and type
    - userFeaturePath: the input path (hdfs) of user feature table
    - itemFeaturePath: the input path (hdfs) of item feature table
    - torchModelPath: the name of the model file, unsupervised_bipartite_graphsage.pt in this example
    - `userFeatureDim`: the dimension for the feature for each user node, which should be equal with the number when generate the model file
    - `itemFeatureDim`: the dimension for the feature for each item node, which should be equal with the number when generate the model file
    - `featureEmbedInputPath`:the embedding matrix for features(contains user, item), set it if you need to increment train only when data is high-sparse
    - `userFieldNum`: the field num of user, the default is `-1`, set it if you need only when data is high-sparse
    - `itemFieldNum`: the field num of item, the default is `-1`, set it if you need only when data is high-sparse
    - `userFeatEmbedDim`: the dim of user embedding, the default is `-1`, set it if you need only when data is high-sparse
    - `itemFeatEmbedDim`；the dim of item embedding, the default is `-1`, set it if you need only when data is high-sparse
    - `fieldMultiHot`: whether the field is multi-hot(only support the last field is multi-hot), the default is `false`, set it if you need only when data is high-sparse
    - stepSize: the learning rate when training
    - optimizer: adam/momentum/sgd/adagrad
    - numEpoch: number of epoches you want to run 
    - trainRatio: randomly sampling part of samples to train
    - numPartitions: partition the data into how many partitions
    - psNumPartition: partition the data into how many partitions on ps
    - format: should be sparse/dense
    - userNumSamples: the number of samples of user neighbors
    - itemNumSamples: the number of samples of item neighbors
    - batchSize: batch size for each optimizing step
    - useBalancePartition:whether use balance partition, the default is `false`
    - predictOutputPath: hdfs path to save the predict label for all nodes in the graph, set it if you need the label
    - userEmbeddingPath: hdfs path to save the embedding for all user nodes in the graph, set it if you need the embedding vectors
    - itemEmbeddingPath: hdfs path to save the embedding for all item nodes in the graph, set it if you need the embedding vectors
    - outputModelPath: hdfs path to save the training model file, which is also a torch model pt file, set it if you want to do predicting or incremental training in the next step
    - actionType: should be train/predict
    - numBatchInit: we use a mini-batch way when initializing features and network structures on parameter servers. this parameter determines how many batches we uses in this step
    - periods: save pt model every few epochs
    - saveCheckpoint: whether checkpoint ps model after init parameters to ps
    - checkpointInterval: save ps model every few epochs
    - decay: the decay of learning ratio, the default is 0
    - useSharedSamples: whether reuse the samples to calculate the train acc to accelerate, the default is false

    **Notes:**
    - The model file, hgat_sparse.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need use ``--files`` to upload the model file.


### Example of INDUCTIVE MATRIX COMPLETION BASED ON GRAPH NEURAL NETWORKS(IGMC)

[IGMC](https://arxiv.org/pdf/1904.12058.pdf) IGMC trains a graph neural network (GNN) based purely on 1-hop subgraphs around (user, item) pairs generated from the rating matrix and maps these subgraphs to their corresponding ratings 

Here we give an example of using IGMC over pytorch on angel.


1. **Generate pytorch sciprt model**
    First, go to directory of python/graph and execute the following command:  
    **classification**:  

    ```$xslt
    supervised_igmc.py --input_user_dim 23 --input_item_dim 18 --hidden_dim 32 --edge_types 5 --output_dim 5 --output_file igmc_ml_class.pt
    ```  
	**regression**:  
	
	```$xslt
    python supervised_igmc.py --input_user_dim 23 --input_item_dim 18 --hidden_dim 32 --edge_types 5 --output_dim 5 --method regression --output_file igmc_ml_reg.pt
    ```  
    This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of Unsupervised Bipartite GraphSage. After that, you will obtain a model file named "igmc_ml_class.pt or igmc_ml_reg.pt". 
	- `edge_types`:the category of rating
	- `method`: the encode of feature, the default is `classification`, optional value:classification,regression.

2. **Preparing input data**
    There are three inputs required for IGMC, including the edge table(with rating), the feature table.

    IGMC requires an edge file which contains three columns including the source node column, the destination column and rating column. For example:
	```
	src dst rating
	```
	The src and dst is a Long numeric while the rating is an Integer numeric.

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

	**The only difference between classification job and regression job is pt model, the submit scriptis same**:
	 	
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
          --name "IGMC-angel" \
          --jars $SONA_SPARK_JARS  \
          --archives angel_libtorch.zip#torch\  #path to c++ library files
          --files igmc_ml_class.pt \   #path to pytorch script model
          --driver-memory 5g \
          --num-executors 5 \
          --executor-cores 1 \
          --executor-memory 5g \
          --class com.tencent.angel.pytorch.example.supervised.cluster.IGMCExample \
          ./pytorch-on-angel-${VERSION}.jar \   # jar from Compiling java submodule
          edgePath:$edgePath userFeaturePath:$userFeaturePath itemFeaturePath:$itemFeaturePath\
          torchModelPath:igmc_ml_class.pt userFeatureDim:23 itemFeatureDim:18 stepSize:0.0001 decay:0.001\
          optimizer:adam numEpoch:10 testRatio:0.5 \
          numPartitions:50 format:dense batchSize:128\
          predictOutputPath:$predictOutputPath outputModelPath:$outputModelPath\
          actionType:train numBatchInit:5
    ```  
    Here we give a short description for the parameters in the submit script. 

    - edgePath: the input path (hdfs) of edge table, which contains src, dst and type
    - userFeaturePath: the input path (hdfs) of user feature table
    - itemFeaturePath: the input path (hdfs) of item feature table
    - torchModelPath: the name of the model file, unsupervised_bipartite_graphsage.pt in this example
    - `userFeatureDim`: the dimension for the feature for each user node, which should be equal with the number when generate the model file
    - `itemFeatureDim`: the dimension for the feature for each item node, which should be equal with the number when generate the model file
    - stepSize: the learning rate when training
    - optimizer: adam/momentum/sgd/adagrad
    - numEpoch: number of epoches you want to run 
    - testRatio: use how many nodes from the label file for testing(if `testLabelPath` is set, the testRatio is invalid)
    - numPartitions: partition the data into how many partitions
    - psNumPartition: partition the data into how many partitions on ps
    - format: should be sparse/dense
    - numSamples: the number of samples of neighbors
    - batchSize: batch size for each optimizing step
    - useBalancePartition:whether use balance partition, the default is `false`
    - predictOutputPath: hdfs path to save the predict label for all nodes in the graph, set it if you need the label
    - outputModelPath: hdfs path to save the training model file, which is also a torch model pt file, set it if you want to do predicting or incremental training in the next step
    - actionType: should be train/predict
    - numBatchInit: we use a mini-batch way when initializing features and network structures on parameter servers. this parameter determines how many batches we uses in this step.
    - periods: save pt model every few epochs
    - saveCheckpoint: whether checkpoint ps model after init parameters to ps
    - checkpointInterval: save ps model every few epochs
    - decay: the decay of learning ratio, the default is 0
    - evals: eval method, the default is acc, the optional value: acc,binary_acc,auc,precision,f1,rmse,recall
    - validatePeriods: validate model every few epochs
    - useSharedSamples: whether reuse the samples to calculate the train acc to accelerate, the default is false

    **Notes:**
    - The model file, igmc\_ml\_class.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need use ``--files`` to upload the model file.


### FAQ

1. If you want to use GAT or HGAT, pytorch >= v1.5.0.
2. If you found loss is NAN or does not converge, you can decrease the learning rate, such as: 0.001,0.0001 or lower.
3. If you encounter the error `file not found model.json`, please check the version of model.pt and the version of pytorch, whether tem are matched.
4. If you encounter the error `java.lang.UnsupportedOperationException: empty collection`, please check whether the input data is empty.
5. If you encounter the error `ERROR AngelYarnClient: submit application to yarn failed.`, ps did not apply for resources, change another cluster or try later.
6. If you encounter the error `java.lang.UnsatisfiedLinkError: no torch_angel in java.library.path`, please check whether the torch path is correct.