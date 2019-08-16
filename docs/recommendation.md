### Recommendation Algorithm

Currently, Pytorch on angel supports a series of recommendation algorithms.

In detail, the following methods are currently implemented:

* **[FM](../python/recommendation/fm.py)** from Steffen Rendle : [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
* **[DeepFM](../python/recommendation/deepfm.py)** from Huifeng Guo et al: [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf) 
* **[AttentionFM](../python/recommendation/attention_fm.py)** from Jun Xiao et al: [Attentional Factorization Machines:
Learning the Weight of Feature Interactions via Attention Networks](https://arxiv.org/pdf/1708.04617.pdf)
* **[DCN](../python/recommendation/dcn.py)** from Ruoxi Wang et al: [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/pdf/1708.05123.pdf)
* **[DeepAndWide](../python/recommendation/deepandwide.py)** from Heng-Tze Cheng et al: [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)
* **[PNN](../python/recommendation/pnn.py)** from Yanru Qu et al: [Product-based Neural Networks for User Response Prediction](https://arxiv.org/pdf/1611.00144.pdf)
* **[XDeepFM](../python/recommendation/xdeepfm.py)** from Jianxun Lian et al: [xDeepFM: Combining Explicit and Implicit Feature Interactions
for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)

We use DeepFM as an example to illustrate the details process of running an algorithm.
The methods are similar for other algorithms.

### Example of DeepFM

1. ** Generate pytorch script model**
    First, go to directory of python/recommendation and execute the following command:
    ```$xslt
    python deepfm.py --input_dim 148 --n_fields 13 --embedding_dim 10 --fc_dims 10 5 1
    ```

    Some explanations for the parameters.
    - input_dim: the feature dimension for the data
    - n_fields: number of fields for data
    - embedding_dim: dimension for embedding layer
    - fc_dims: the dimensions for fc layers in deepfm. "10 5 1" indicates a two-layers mlp composed with one 10x5 layer and one 5x1 layer.

    This python script will generate a TorchScript model with the structure of dataflow graph for deepfm. This file is named ``deepfm.pt``.

2. ** Preparing the input data**
    The input data of DeepFM should be libffm format. Each line of the input data represents one data sample.
    ```
    label field1:feature1:value1 field2:feature2:value2
    ```
    In Pytorch on angel, multi-hot field is allowed, which means some field can be appeared multi-times in one data example.

3. ** Training model**
    After obtaining the model file (deepfm.pt) and the input data, we can submit a task through Spark on Angel to train the model. The command is:
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
          --name "deepfm on angel" \
          --jars $SONA_SPARK_JARS  \
          --archives angel_libtorch.zip#torch\  #path to c++ library files
          --files deepfm.pt \   #path to pytorch script model
          --driver-memory 5g \
          --num-executors 5 \
          --executor-cores 1 \
          --executor-memory 5g \
          --class com.tencent.angel.pytorch.examples.ClusterExample \
          ./pytorch-on-angel-1.0-SNAPSHOT.jar \   # jar from Compiling java submodule
          input:$input batchSize:128 torchModelPath:deepfm.pt \
          stepSize:0.001 numEpoch:10 validateRatio:0.2 \
          modelPath:$output \
    ```

    Description for the parameters:

    - input: the input path (hdfs) for training data
    - batchSize: batch size for each optimizing step
    - torchModelPath: the name of the generated torch model
    - stepSize: learning rate
    - numEpoch: how many epoches for the training process
    - validateRatio: how many training examples are used for testing
    - modelPath: the output path (hdfs) for the training model
