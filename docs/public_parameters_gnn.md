## Public Parameters for GNN Algorithms

### Input/Oupput Path

Property Name | Default | Meaning
---------------- | --------------- | ---------------
edgePath | "" | the input path (hdfs) of edge table
featurePath | "" | the input path (hdfs) of feature table
labelPath | "" | the input path (hdfs) of label table
testLabelPath | "" | the input path (hdfs) of validate label table
torchModelPath | model.pt | the name of the model file, xx.pt in this example
predictOutputPath | "" | hdfs path to save the predict label for all nodes in the graph, set it if you need the label
embeddingPath | "" | hdfs path to save the embedding for all nodes in the graph, set it if you need the embedding vectors
outputModelPath | "" | hdfs path to save the training model file, which is also a torch model pt file, set it if you want to do predicting or incremental training in the next step
userFeaturePath | "" | the input path (hdfs) of user feature table
itemFeaturePath | "" | the input path (hdfs) of item feature table
userEmbeddingPath | "" | hdfs path to save the embedding for all user nodes in the graph, set it if you need the embedding vectors
itemEmbeddingPath | "" | hdfs path to save the embedding for all item nodes in the graph, set it if you need the embedding vectors
featureEmbedInputPath | "" | the embedding matrix for features(contains user, item), set it if you need to increment train only when data is high-sparse

### Data Parameters

Property Name | Default | Meaning
---------------- | --------------- | ---------------
featureDim | -1 | the dimension for the feature for each node, which should be equal with the number when generate the model file
edgeFeatureDim | -1 | the dimension for the feature of edge, which should be equal with the number when generate the model file
itemTypes | 266 | types of item nodes, which is also the num of meta-paths, for HAN
userFeatureDim | -1 | the dimension for the feature for each user node, which should be equal with the number when generate the model file
itemFeatureDim | -1 | the dimension for the feature for each item node, which should be equal with the number when generate the model file
fieldNum | -1 | the field num of user, the default is `-1`, set it if you need only when data is high-sparse
featEmbedDim | -1 | the dim of user embedding, the default is `-1`, set it if you need only when data is high-sparse
userFieldNum | -1 | the field num of user, the default is `-1`, set it if you need only when data is high-sparse
itemFieldNum | -1 | the field num of item, the default is `-1`, set it if you need only when data is high-sparse
userFeatEmbedDim | -1 | the dim of user embedding, the default is `-1`, set it if you need only when data is high-sparse
itemFeatEmbedDim | -1 | the dim of item embedding, the default is `-1`, set it if you need only when data is high-sparse
fieldMultiHot | false | whether the field is multi-hot(only support the last field is multi-hot), the default is `false`, set it if you need only when data is high-sparse
format | sparse | should be sparse/dense
numLabels | 1 | the num of multi-label classification task if numLabels > 1, `the default is 1` for single-label classification

### Algorithm Parameters

Property Name | Default | Meaning
---------------- | --------------- | ---------------
stepSize | 0.01 | the learning rate when training
decay | 0.001 | the decay of learning ratio, the default is 0
optimizer | adma | adam/momentum/sgd/adagrad
numEpoch | 10 | number of epoches you want to run 
testRatio | 0.5 | use how many nodes from the label file for testing(if `testLabelPath` is set, the testRatio is invalid)
trainRatio | 0.5 | randomly sampling part of samples to train; for unsupervised gnn algo
samples | 5 | the number of samples when sampling neighbors
userNumSamples | 5 | the number of samples of user neighbors training in the next step
itemNumSamples | 5 | the number of samples of item neighbors
second | true | whether use second hop sampling
batchSize | 100 | batch size for each optimizing step
actionType | train | hshould be train/predict
numBatchInit | 5 | we use a mini-batch way when initializing features and network structures on parameter servers. this parameter determines how many batches we uses in this step.
evals | acc | eval method, the default is acc, the optional value: acc,binary_acc,auc,precision,f1,rmse,recall;`multi_auc` for `numLabels` > 1
periods | 1000 | save pt model every few epochs
saveCheckpoint | false | whether checkpoint ps model after init parameters to ps
checkpointInterval | 0 | save ps model every few epochs
validatePeriods | 5 | validate model every few epochs
useSharedSamples | false | whether reuse the samples to calculate the train acc to accelerate, the default is false
numPartitions | 10 | partition the data into how many partitions
psNumPartition | 10 | partition the data into how many partitions on ps
batchSizeMultiple | 10 | the multiple of batchsize used to accelerate when predict prediction or embedding