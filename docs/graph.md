## GNN Algorithm Instructions

> Pytorch on Angel provides the ability to run graph convolution network algorithm. We follow [Pytorch-Geometric](https://github.com/rusty1s/pytorch_geometric) to define the graph convolution networks while using the parameter server of Angel to store the network structure and features of nodes.


### Introduction
#### 1. How to predict
1. change the input data path and output path
2. change **actionType** to `predict`
3. you can get hdfs://modelPath/xx.pt to local, then use it as training; Or you can use the hdfs path, and set `--files hdfs://modelPath/xx.pt`, in this way the `torchModelPath` can be remove


#### 2. How to train incrementally
1. change the input data path and output path, or you can use the same data to train incrementally
2. set **actionType** as `train` 
3. you can get hdfs://modelPath/xx.pt to local, then use it as training; Or you can use the hdfs path, and set `--files hdfs://modelPath/xx.pt`, in this way the `torchModelPath` can be remove

#### 3. How to calculate the resource
In order to know how to set the resources, you should figure out where the data saved firstly, and then calculate how much data storage space, finally set 2~3 times of data storage. The detail calculation method refer to [Resource Calculation Method](./resource_calculation_method.md)


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
	Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

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
          torchModelPath:graphsage_sparse.pt featureDim:32 stepSize:0.01\
          optimizer:adam numEpoch:10 testRatio:0.5 fieldNum:20 featEmbedDim:8 \
          numPartitions:50 format:sparse samples:10 batchSize:128\
          predictOutputPath:$predictOutputPath embeddingPath:$embeddingPath outputModelPath:$outputModelPath\
          actionType:train numBatchInit:5
    ```

    Here we give a short description for the parameters in the submit script. Detailed parameters and the output result see [details](./public_parameters_gnn.md)
    
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
	Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

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
    Here we give a short description for the parameters in the submit script. Detailed parameters and the output result see [details](./public_parameters_gnn.md)

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
	Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

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
    Here we give a short description for the parameters in the submit script. Detailed parameters and the output result see [details](./public_parameters_gnn.md)

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
	Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

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
	Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

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
    Here we give a short description for the parameters in the submit script. Detailed parameters and the output result see [details](./public_parameters_gnn.md)
    
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
	Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

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
    Here we give a short description for the parameters in the submit script. Detailed parameters and the output result see [details](./public_parameters_gnn.md)

    - edgePath: the input path (hdfs) of edge table, which contains src, dst and type
    
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
	Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

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
    Here we give a short description for the parameters in the submit script. Detailed parameters and the output result see [details](./public_parameters_gnn.md)

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
	Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

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
    Here we give a short description for the parameters in the submit script. Detailed parameters and the output result see [details](./public_parameters_gnn.md)
    
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
	Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

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
    Here we give a short description for the parameters in the submit script. Detailed parameters and the output result see [details](./public_parameters_gnn.md)

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
	Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

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
    Here we give a short description for the parameters in the submit script. Detailed parameters and the output result see [details](./public_parameters_gnn.md)

    - edgePath: the input path (hdfs) of edge table, which contains src, dst and type

    **Notes:**
    - The model file, igmc\_ml\_class.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need use ``--files`` to upload the model file.


### Example of GAMLP

[GAMLP](https://arxiv.org/pdf/2108.10097.pdf) Following the routine of decoupled GNNs, the feature propagation in GAMLP is executed during pre-computation, which helps it maintain high scalability.

GAMLP contains two independent modules:

    1.GAMLP-aggregator, feature propagation aggregation module. This module only needs to be run once in the preprocessing stage as the feature input for subsequent GNN model training; the module can also be run as an independent component as a feature propagation aggregation component common to other GNN algorithms
    2.GAMLP-training, the GNN model training module, loads the features of the aggregation module for training. In this training stage, it is no longer necessary to do node sampling, pulling node feature aggregation and other communication overhead operations, which greatly improves the model training efficiency.

Here we give an example of using GAMLP over pytorch on angel.

#### GAMLP-aggregator
1. **Generate pytorch sciprt model**
   First, go to directory of python/graph and execute the following command:
    ```$xslt
    python aggregator.py --aggregation_type mean --output_file aggregator.pt
    ```
   This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of GAMLP-aggregator. After that, you will obtain a model file named "aggregator.pt". Here we use the Cora dataset as an example, where the feature dimension for each node is 1433 with 7 different classes.  
   Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

2. **Preparing input data**
   There are two inputs required for GAMLP-aggregator, including the edge table and the node feature table.

   The detail info see [Data Format](./data_format_gnn.md)

2. **Submit model to cluster**
   After obtaining the model file and the inputs, we can submit a task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

   **dense/low-sparse data**:

   ```$xslt
   source ./spark-on-angel-env.sh  
    $SPARK_HOME/bin/spark-submit \
    --master yarn-cluster\
    --conf spark.hadoop.hadoop.job.ugi=tdwadmin,supergroup\
    --conf spark.tdw.authentication=usp:usp@all2012\
    --conf spark.yarn.allocation.am.maxMemory=55g \
    --conf spark.yarn.allocation.executor.maxMemory=55g \
    --conf spark.driver.maxResultSize=20g \
    --conf spark.kryoserializer.buffer.max=2000m\
    --conf spark.submitter=lucytjia \
    --conf spark.ps.instances=20 \
    --conf spark.ps.cores=20 \
    --conf spark.ps.jars=$SONA_ANGEL_JARS \
    --conf spark.ps.memory=10g \
    --conf spark.ps.log.level=INFO \
    --conf spark.hadoop.angel.ps.jvm.direct.factor.use.direct.buff=0.20 \
    --conf spark.hadoop.angel.ps.backup.interval.ms=2000000000 \
    --conf spark.hadoop.angel.netty.matrixtransfer.max.message.size=209715200 \
    --conf spark.hadoop.angel.matrixtransfer.request.timeout.ms=240000 \
    --conf spark.hadoop.angel.ps.request.resource.use.minibatch=true \
    --conf spark.hadoop.angel.ps.router.type=range \
    --conf spark.executor.extraLibraryPath=./torch/torch-lib \
    --conf spark.driver.extraLibraryPath=./torch/torch-lib\
    --conf spark.executorEnv.OMP_NUM_THREADS=2 \
    --conf spark.executorEnv.MKL_NUM_THREADS=2 \
    --queue $queue \
    --name "haggregator" \
    --jars $SONA_SPARK_JARS  \
    --archives $torch#torch\
    --files aggregator.pt\
    --driver-memory 45g \
    --num-executors 50 \
    --executor-cores 20 \
    --executor-memory 10g \
    --class com.tencent.angel.pytorch.examples.supervised.cluster.AggregatorExample \
    ./pytorch-on-angel-${VERSION}.jar \ # jar from Compiling java submodule
    edgePath:$edgePath featurePath:$featurePath hops:4 sampleMethod:aliasTable \
    upload_torchModelPath:aggregator.pt featureDim:1433 sep:tab \
    numPartitions:100 format:dense samples:10 batchSize:128 numBatchInit:128 \
    predictOutputPath:$predictOutputPath periods:10 \
    checkpointInterval:10 psNumPartition:100 useBalancePartition:false
   ```  
#### GAMLP-training
1. **Generate pytorch sciprt model**
   First, go to directory of python/graph and execute the following command:
    ```$xslt
    python gamlp.py --input_dim 1433 --hidden_dim 128 --output_dim 7 --hops 4 --output_file gamlp.pt
    ```
   Note that the --hops must be same with the GAMLP-aggregator module.
   This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of GAMLP-aggregator. After that, you will obtain a model file named "aggregator.pt". Here we use the Cora dataset as an example, where the feature dimension for each node is 1433 with 7 different classes.  
   Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

2. **Preparing input data**
   There are three inputs required for GAMLP-training, including the edge table, the node feature table and the node label talbe.
   Note that the feature comes from the predictOutputPath of the previous GAMLP-aggregator module.
   The detail info see [Data Format](./data_format_gnn.md)

2. **Submit model to cluster**
   After obtaining the model file and the inputs, we can submit a task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

   **dense/low-sparse data**:

   ```$xslt
    --master yarn-cluster \
    --conf spark.hadoop.hadoop.job.ugi=tdwadmin,supergroup\
    --conf spark.tdw.authentication=usp:usp@all2012\
    --conf spark.yarn.allocation.am.maxMemory=55g \
    --conf spark.yarn.allocation.executor.maxMemory=55g \
    --conf spark.driver.maxResultSize=20g \
    --conf spark.kryoserializer.buffer.max=2000m\
    --conf spark.submitter=lucytjia \
    --conf spark.ps.instances=2 \
    --conf spark.ps.cores=2 \
    --conf spark.ps.jars=$SONA_ANGEL_JARS \
    --conf spark.ps.memory=10g \
    --conf spark.ps.log.level=INFO \
    --conf spark.hadoop.angel.ps.jvm.direct.factor.use.direct.buff=0.20 \
    --conf spark.hadoop.angel.ps.backup.interval.ms=2000000000 \
    --conf spark.hadoop.angel.netty.matrixtransfer.max.message.size=209715200 \
    --conf spark.hadoop.angel.matrixtransfer.request.timeout.ms=240000 \
    --conf spark.hadoop.angel.ps.request.resource.use.minibatch=true \
    --conf spark.hadoop.angel.ps.router.type=hash \
    --conf spark.executor.extraLibraryPath=./torch/torch-lib \
    --conf spark.driver.extraLibraryPath=./torch/torch-lib\
    --conf spark.executorEnv.OMP_NUM_THREADS=2 \
    --conf spark.executorEnv.MKL_NUM_THREADS=2 \
    --queue $queue \
    --name "gamlp" \
    --jars $SONA_SPARK_JARS  \
    --archives $torch#torch\
    --files gamlp.pt\
    --driver-memory 45g \
    --num-executors 2 \
    --executor-cores 2 \
    --executor-memory 10g \
    --class com.tencent.angel.pytorch.examples.supervised.cluster.GAMLPExample \
    ./pytorch-on-angel-${VERSION}.jar \
    edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath hops:4\
    featureDim:1433 sep:tab labelsep:tab stepSize:0.001\
    optimizer:adam numEpoch:100 testRatio:0.4 validatePeriods:1 evals:acc,f1 \
    numPartitions:10 format:dense samples:10 batchSize:128 numBatchInit:128 \
    embeddingPath:$output periods:10 outputModelPath:$outputModelPath \
    checkpointInterval:10 predictOutputPath:$predictOutputPath psNumPartition:10 useBalancePartition:false
   ``` 
   Here we give a short description for the parameters in the submit script. Detailed parameters and the output result see [details](./public_parameters_gnn.md)

   **Notes:**
    - The model file, aggregator.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need use ``--files`` to upload the model file.
    - The --hops must be same in all parameter settings.



### Example of HGAMLP

HGAMLP is an extension of GAMLP for heterogeneous graphs, it contains two independent modules:

    1.HGAMLP-aggregator, Feature propagation aggregation module for heterogeneous graphs. According to the metapaths passed in by the user, aggregate different types of nodes, and output the aggregated features of each type of nodes.
    2.HGAMLP-training, the GNN model training module, which is same as GAMLP-training.

Here we give an example of using HGAMLP over pytorch on angel.

#### GAMLP-aggregator
1. **Generate pytorch sciprt model**
   First, go to directory of python/graph and execute the following command:
    ```$xslt
    python aggregator.py --aggregation_type mean --output_file aggregator.pt
    ```
   This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of GAMLP-aggregator. After that, you will obtain a model file named "aggregator.pt". Here we use the Cora dataset as an example, where the feature dimension for each node is 1433 with 7 different classes.  
   Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

2. **Preparing input data**
   There are two inputs required for HGAMLP-aggregator, including the edge table and the node feature table.

   The detail info see [Data Format](./data_format_gnn.md)

2. **Submit model to cluster**
   After obtaining the model file and the inputs, we can submit a task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

   **dense/low-sparse data**:

   ```$xslt
   source ./spark-on-angel-env.sh  
    $SPARK_HOME/bin/spark-submit \
    --master yarn-cluster\
    --conf spark.hadoop.hadoop.job.ugi=tdwadmin,supergroup\
    --conf spark.tdw.authentication=usp:usp@all2012\
    --conf spark.yarn.allocation.am.maxMemory=55g \
    --conf spark.yarn.allocation.executor.maxMemory=55g \
    --conf spark.driver.maxResultSize=20g \
    --conf spark.kryoserializer.buffer.max=2000m\
    --conf spark.submitter=lucytjia \
    --conf spark.ps.instances=20 \
    --conf spark.ps.cores=20 \
    --conf spark.ps.jars=$SONA_ANGEL_JARS \
    --conf spark.ps.memory=10g \
    --conf spark.ps.log.level=INFO \
    --conf spark.hadoop.angel.ps.jvm.direct.factor.use.direct.buff=0.20 \
    --conf spark.hadoop.angel.ps.backup.interval.ms=2000000000 \
    --conf spark.hadoop.angel.netty.matrixtransfer.max.message.size=209715200 \
    --conf spark.hadoop.angel.matrixtransfer.request.timeout.ms=240000 \
    --conf spark.hadoop.angel.ps.request.resource.use.minibatch=true \
    --conf spark.hadoop.angel.ps.router.type=range \
    --conf spark.executor.extraLibraryPath=./torch/torch-lib \
    --conf spark.driver.extraLibraryPath=./torch/torch-lib\
    --conf spark.executorEnv.OMP_NUM_THREADS=2 \
    --conf spark.executorEnv.MKL_NUM_THREADS=2 \
    --queue $queue \
    --name "haggregator" \
    --jars $SONA_SPARK_JARS  \
    --archives $torch#torch\
    --files aggregator.pt\
    --driver-memory 45g \
    --num-executors 50 \
    --executor-cores 20 \
    --executor-memory 10g \
    --class com.tencent.angel.pytorch.examples.supervised.cluster.HeteAggregatorExample  \
    ./pytorch-on-angel-${VERSION}.jar \ # jar from Compiling java submodule
    edgePaths:$edgePaths featurePaths:$featurePaths metapaths:$metapaths isWeighted:false \
    upload_torchModelPath:aggregator.pt sep:tab featureSep:tab featureDims:$featureDims aggregator_in_scala:true \
    numPartitions:2 format:dense samples:10 batchSize:128 numBatchInit:128 useWeightedAggregate:true \
    embeddingOutputPaths:$embeddingOutputPaths periods:10 sampleMethod:randm \
    checkpointInterval:10 psNumPartition:2 useBalancePartition:false
   ```  

#### HGAMLP-training
1. **Generate pytorch sciprt model**
   First, go to directory of python/graph and execute the following command:
    ```$xslt
    python gamlp.py --input_dim 1433 --hidden_dim 128 --output_dim 7 --hops 4 --output_file gamlp.pt
    ```
   Note that the --hops must be same with the HGAMLP-aggregator metapaths length.
   This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of GAMLP-aggregator. After that, you will obtain a model file named "aggregator.pt". Here we use the Cora dataset as an example, where the feature dimension for each node is 1433 with 7 different classes.  
   Detailed parameter introduction reference [Python Model Parameters](./public_parameters_gnn.md)

2. **Preparing input data**
   There are three inputs required for GAMLP-training, including the edge table, the node feature table and the node label talbe.
   Note that the feature comes from the predictOutputPath of the previous GAMLP-aggregator module.
   The detail info see [Data Format](./data_format_gnn.md)

2. **Submit model to cluster**
   After obtaining the model file and the inputs, we can submit a task through [Spark on Angel](https://github.com/Angel-ML/angel/blob/master/docs/tutorials/spark_on_angel_quick_start_en.md).

   **dense/low-sparse data**:

   ```$xslt
    --master yarn-cluster \
    --conf spark.hadoop.hadoop.job.ugi=tdwadmin,supergroup\
    --conf spark.tdw.authentication=usp:usp@all2012\
    --conf spark.yarn.allocation.am.maxMemory=55g \
    --conf spark.yarn.allocation.executor.maxMemory=55g \
    --conf spark.driver.maxResultSize=20g \
    --conf spark.kryoserializer.buffer.max=2000m\
    --conf spark.submitter=lucytjia \
    --conf spark.ps.instances=2 \
    --conf spark.ps.cores=2 \
    --conf spark.ps.jars=$SONA_ANGEL_JARS \
    --conf spark.ps.memory=10g \
    --conf spark.ps.log.level=INFO \
    --conf spark.hadoop.angel.ps.jvm.direct.factor.use.direct.buff=0.20 \
    --conf spark.hadoop.angel.ps.backup.interval.ms=2000000000 \
    --conf spark.hadoop.angel.netty.matrixtransfer.max.message.size=209715200 \
    --conf spark.hadoop.angel.matrixtransfer.request.timeout.ms=240000 \
    --conf spark.hadoop.angel.ps.request.resource.use.minibatch=true \
    --conf spark.hadoop.angel.ps.router.type=hash \
    --conf spark.executor.extraLibraryPath=./torch/torch-lib \
    --conf spark.driver.extraLibraryPath=./torch/torch-lib\
    --conf spark.executorEnv.OMP_NUM_THREADS=2 \
    --conf spark.executorEnv.MKL_NUM_THREADS=2 \
    --queue $queue \
    --name "gamlp" \
    --jars $SONA_SPARK_JARS  \
    --archives $torch#torch\
    --files gamlp.pt\
    --driver-memory 45g \
    --num-executors 2 \
    --executor-cores 2 \
    --executor-memory 10g \
    --class com.tencent.angel.pytorch.examples.supervised.GAMLPExample \
    ./pytorch-on-angel-${VERSION}.jar \
    edgePath:$edgePath featurePath:$featurePath labelPath:$labelPath hops:4\
    featureDim:1433 sep:tab labelsep:tab stepSize:0.001\
    optimizer:adam numEpoch:100 testRatio:0.4 validatePeriods:1 evals:acc,f1 \
    numPartitions:10 format:dense samples:10 batchSize:128 numBatchInit:128 \
    embeddingPath:$output periods:10 outputModelPath:$outputModelPath \
    checkpointInterval:10 predictOutputPath:$predictOutputPath psNumPartition:10 useBalancePartition:false
   ``` 
   Here we give a short description for the parameters in the submit script. Detailed parameters and the output result see [details](./public_parameters_gnn.md)

   **Notes:**
    - The model file, aggregator.pt and gamlp.pt, should be uploaded to Spark Driver and each Executor. Therefore, we need use ``--files`` to upload the model file.
    - The --hops must be same in all parameter settings.

### FAQ

1. If you want to use GAT or HGAT, pytorch >= v1.5.0.
2. If you found loss is NAN or does not converge, you can decrease the learning rate, such as: 0.001,0.0001 or lower.
3. If you encounter the error `file not found model.json`, please check the version of model.pt and the version of pytorch, whether them are matched.
4. If you encounter the error `java.lang.UnsupportedOperationException: empty collection`, please check whether the input data is empty.
5. If you encounter the error `ERROR AngelYarnClient: submit application to yarn failed.`, ps did not apply for resources, change another cluster or try later.
6. If you encounter the error `java.lang.UnsatisfiedLinkError: no torch_angel in java.library.path`, please check whether the torch path is correct.
