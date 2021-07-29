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

	Algo Name | Data saved on ps | Resource
	---------------- | --------------- | ---------------
	**Semi GraphSage/GAT** | edge, node feature, node label | (edge\_num * 2 * 8Byte + node\_num * feature\_dim * 4Byte + the num of node with label * 4Byte)
	**RGCN/HAN** | edge with type(dst type), node feature, node label | (edge\_num * 2 * 8Byte + edge\_num * 4Byte + node\_num * feature\_dim * 4Byte + the num of node with label * 4Byte)
	**RGCN/HAN** | edge with type(dst type), node feature, node label | (edge\_num * 2 * 8Byte + edge\_num * 4Byte + node\_num * feature\_dim * 4Byte + the num of node with label * 4Byte)
	**DGI/Unsupervised GraphSage** | edge, node feature | (edge\_num * 2 * 8Byte + node\_num * feature\_dim * 4Byte)
	**Semi Bipartite GraphSage** | edge * 2, user feature, item feature, user label | (edge\_num * 2 * 8Byte * 2 + user\_num * user\_feature_dim * 4Byte + item\_num * item\_feature\_dim * 4Byte + user\_label\_num * 4Byte)
	**Unsupervised Bipartite GraphSage** | edge * 2, user feature, item feature | (edge\_num * 2 * 8Byte * 2 + user_num * user\_feature\_dim * 4Byte + item\_num * item_feature_dim * 4Byte)
	**IGMC** | edge with type, user feature, item feature | (edge\_num * 2 * 8Byte + edge\_num * 4Byte + user\_num * user\_feature\_dim * 4Byte + item\_num * item_feature\_num * 4Byte)

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

	Algo Name | Data saved on ps | Resource
	---------------- | --------------- | ---------------
	**Semi GraphSage** | edge, node feature, node label, embedding matrix | (edge\_num * 2 * 8Byte + node\_num * `field_num` * 4Byte + the num of node with label * 4Byte + `input_embedding_dim * slots * input_dim * 4Byte`)
	**Semi Bipartite GraphSage** | edge * 2, user feature, item feature, user label, user embedding matrix, item embedding matrix | (edge\_num * 2 * 8Byte * 2 + user\_num * `user_field_num` * 4Byte + item\_num * `item_field_num` * 4Byte + user\_label\_num * 4Byte + `user_embedding_dim * slots * user_feature_dim * 4Byte + item_embedding_dim * slots * item_feature_dim * 4Byte`)
	**HGAT** | edge * 2, user feature, item feature, user embedding matrix, item embedding matrix | (edge\_num * 2 * 8Byte * 2 + user\_num * `user_field_num` * 4Byte + item\_num * `item_field_num` * 4Byte + `user_embedding_dim * slots * user_feature_dim * 4Byte + item_embedding_dim * slots * item_feature_dim * 4Byte`)
	

- Spark Executor Resource:the configuration of Spark resources is mainly considered from the aspect of training data(Edge data is usually saved on Spark Executor), and it is best to save 2 times the input data. If the memory is tight, 1x is acceptable, but it will be relatively slow. For example, a 10 billion edge set is about 200G in size, and a 30G * 20 configuration is sufficient. 


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
    There are three inputs required for graphsage, including the edge table, the node feature table and the node label table.

    The detail info see [Data Format](./data_format_gnn.md)

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

    - `featureEmbedInputPath`:the embedding matrix for features(contains user, item), set it if you need to increment train only when data is high-sparse
    - `fieldNum`: the field num of user, the default is `-1`, set it if you need only when data is high-sparse
    - `featEmbedDim`: the dim of user embedding, the default is `-1`, set it if you need only when data is high-sparse
    - `fieldMultiHot`: whether the field is multi-hot(only support the last field is multi-hot), the default is `false`, set it if you need only when data is high-sparse

	Other parameters see [details](./public_parameters_gnn.md)
    
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
    There are two inputs required for dgi, including the edge table and the node feature table.

    The detail info see [Data Format](./data_format_gnn.md)

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
    Here we give a short description for the parameters in the submit script. Detail parameters see [details](./public_parameters_gnn.md)

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
    There are three inputs required for graphsage, including the edge table with type, the node feature table and the node label table.

    The detail info see [Data Format](./data_format_gnn.md)

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
    Here we give a short description for the parameters in the submit script. Detail parameters see [details](./public_parameters_gnn.md)

    - edgePath: the input path (hdfs) of edge table, which contains src, dst and type

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
    There are three inputs required for graphsage, including the edge table with edge feature, the node feature table and the node label table.

    The detail info see [Data Format](./data_format_gnn.md)

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
    There are three inputs required for graphsage, including the edge table, the node feature table and the node label table.

    The detail info see [Data Format](./data_format_gnn.md)

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
    Here we give a short description for the parameters in the submit script. Detail parameters see [details](./public_parameters_gnn.md)
    
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
    There are three inputs required for han, including the edge table with type, the node feature table and the node label table.

    HAN requires an edge file which contains three columns including the source node column, the destination column and the node type column. The third column indicates the destination nodes' types, each type indicates a meta-path of "A-B-A".  

	The detail info see [Data Format](./data_format_gnn.md)

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
    Here we give a short description for the parameters in the submit script. Detail parameters see [details](./public_parameters_gnn.md)

    - edgePath: the input path (hdfs) of edge table, which contains src, dst and type
	- `featureEmbedInputPath`:the embedding matrix for features(contains user, item), set it if you need to increment train only when data is high-sparse
    - `fieldNum`: the field num of user, the default is `-1`, set it if you need only when data is high-sparse
    - `featEmbedDim`: the dim of user embedding, the default is `-1`, set it if you need only when data is high-sparse
    - `fieldMultiHot`: whether the field is multi-hot(only support the last field is multi-hot), the default is `false`, set it if you need only when data is high-sparse
    
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
    There are three inputs required for han, including the edge table, the user node feature table, the item node feature table and the label table for user node.

    The detail info see [Data Format](./data_format_gnn.md)

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
    Here we give a short description for the parameters in the submit script. Detail parameters see [details](./public_parameters_gnn.md)

    - `userFeatureDim`: the dimension for the feature for each user node, which should be equal with the number when generate the model file
    - `itemFeatureDim`: the dimension for the feature for each item node, which should be equal with the number when generate the model file
    - `featureEmbedInputPath`:the embedding matrix for features(contains user, item), set it if you need to increment train only when data is high-sparse
    - `userFieldNum`: the field num of user, the default is `-1`, set it if you need only when data is high-sparse
    - `itemFieldNum`: the field num of item, the default is `-1`, set it if you need only when data is high-sparse
    - `userFeatEmbedDim`: the dim of user embedding, the default is `-1`, set it if you need only when data is high-sparse
    - `itemFeatEmbedDim`；the dim of item embedding, the default is `-1`, set it if you need only when data is high-sparse
    - `fieldMultiHot`: whether the field is multi-hot(only support the last field is multi-hot), the default is `false`, set it if you need only when data is high-sparse

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
    There are three inputs required for Unsupervised Bipartite GraphSage, including the edge table, the user node feature table，and item node feature table.

    The detail info see [Data Format](./data_format_gnn.md)

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
    Here we give a short description for the parameters in the submit script. Detail parameters see [details](./public_parameters_gnn.md)

    - edgePath: the input path (hdfs) of edge table, which contains src, dst
    - userFeatureDim: the dimension for the feature for each user node, which should be equal with the number when generate the model file
    - itemFeatureDim: the dimension for the feature for each item node, which should be equal with the number when generate the model file

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
    There are three inputs required for HGAT, including the edge table, the user feature node table，and item node feature table.

    The detail info see [Data Format](./data_format_gnn.md)

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
    Here we give a short description for the parameters in the submit script. Detail parameters see [details](./public_parameters_gnn.md)

    - `userFeatureDim`: the dimension for the feature for each user node, which should be equal with the number when generate the model file
    - `itemFeatureDim`: the dimension for the feature for each item node, which should be equal with the number when generate the model file
    - `featureEmbedInputPath`:the embedding matrix for features(contains user, item), set it if you need to increment train only when data is high-sparse
    - `userFieldNum`: the field num of user, the default is `-1`, set it if you need only when data is high-sparse
    - `itemFieldNum`: the field num of item, the default is `-1`, set it if you need only when data is high-sparse
    - `userFeatEmbedDim`: the dim of user embedding, the default is `-1`, set it if you need only when data is high-sparse
    - `itemFeatEmbedDim`；the dim of item embedding, the default is `-1`, set it if you need only when data is high-sparse
    - `fieldMultiHot`: whether the field is multi-hot(only support the last field is multi-hot), the default is `false`, set it if you need only when data is high-sparse

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
    There are three inputs required for IGMC, including the edge table(with rating), the node feature table.

    The detail info see [Data Format](./data_format_gnn.md)

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
    Here we give a short description for the parameters in the submit script. Detail parameters see [details](./public_parameters_gnn.md)

    - edgePath: the input path (hdfs) of edge table, which contains src, dst and type
    - userFeatureDim: the dimension for the feature for each user node, which should be equal with the number when generate the model file
    - itemFeatureDim: the dimension for the feature for each item node, which should be equal with the number when generate the model file

    **Notes:**
    - The model file, igmc\_ml\_class.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need use ``--files`` to upload the model file.


### FAQ

1. If you want to use GAT or HGAT, pytorch >= v1.5.0.
2. If you found loss is NAN or does not converge, you can decrease the learning rate, such as: 0.001,0.0001 or lower.
3. If you encounter the error `file not found model.json`, please check the version of model.pt and the version of pytorch, whether tem are matched.
4. If you encounter the error `java.lang.UnsupportedOperationException: empty collection`, please check whether the input data is empty.
5. If you encounter the error `ERROR AngelYarnClient: submit application to yarn failed.`, ps did not apply for resources, change another cluster or try later.
6. If you encounter the error `java.lang.UnsatisfiedLinkError: no torch_angel in java.library.path`, please check whether the torch path is correct.