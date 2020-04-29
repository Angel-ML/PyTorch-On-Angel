## GraphSage/GCN/DGI

Pytorch on Angel provides the ability to run graph convolution network algorithm. We follow [Pytorch-Geometric](https://github.com/rusty1s/pytorch_geometric) to define the graph convolution networks while using the parameter server of Angel to store the network structure and features of nodes.

### Example of GraphSage

Here we give an example of how to run GraphSage algorithm beyond Pytorch on Angel.

1. **Generate pytorch sciprt model**
    First, go to directory of python/graph and execute the following command:
    ```$xslt
    python graphsage.py --input_dim 1433 --hidden_dim 128 --output_dim 7 --output_file graphsage_cora.pt
    ```
    This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of graphsage. After that, you will obtain a model file named "graphsage_cora.pt". Here we use the Cora dataset as an example, where the feature dimention for each node is 1433 with 7 different classes.

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
          --class com.tencent.angel.pytorch.example.supervised.GCNExample \
          ./pytorch-on-angel-1.0-SNAPSHOT.jar \   # jar from Compiling java submodule
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
    This script utilizes [TorchScript](https://pytorch.org/docs/stable/jit.html) to generate a model file which contains the dataflow graph of dgi. After that, you will obtain a model file named "dgi_cora.pt". Here we use the Cora dataset as an example, where the feature dimention for each node is 1433 with 7 different classes.

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
          --class com.tencent.angel.pytorch.example.unsupervised.DGIExample \
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
          --class com.tencent.angel.pytorch.example.supervised.RGCNExample \
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