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


### Algorithm Parameters for Pytorch Model

Property Name | Default | Meaning
---------------- | --------------- | ---------------
input_dim | -1 | input dimention of node features
hidden_dim | -1 | hidden dimension of convolution layer
output_dim | -1 | the number of classes for supervised algo, or the dimension of output embedding for unsupervised algo  
output_file | "model_name.pt" | output file name 
input_user_dim | -1 | input dimention of user node features
input_item_dim | -1 | input dimention of item node features
input_user_field_num | -1 | the number of user field num, for high-sparse
input_item_field_num | -1 | the number of item field num, for high-sparse
input_user_embedding_dim | -1 | embedding dim of user node features, for high-sparse
input_item_embedding_dim | -1 | embedding dim of item node features, for high-sparse
field_num | -1 | field num of node features, for high-sparse
input_embedding_dim | -1 | embedding dim of node features, for high-sparse
encode | dense | the encode of feature, optional value:dense, one-hot, multi-hot
edge_input_dim | -1 | the dimension of edge feature
item_types | -1 | the num of item_types
negative_size | 1 | multiples of positive samples
heads | 1 | the number of attention heads, the default is 1
dropout | 0 | the dropout ratio
edge_types | false | the dimension of edge feature
method | classification | the encode of feature, the default is `classification`, optional value:classification,regression for IGMC
task_type | classification | the type of task, the default value is `classification`, optional value:classification or multi-label-classification(multi labels for one node)
class_weights | "" | class weights for supervised GNN, in order to balance class, such as: 0.1,0.9



### Result for Algorithm

Property Name | Result for predictOutputPath | Result for EmbeddingPath
---------------- | --------------- | ---------------
**Semi GraphSage** | node label softmax | node embedding
**DGI/Unsupervised** | - | node embedding
**RGCN** | node label softmax | node embedding
**EdgeProp** | node label softmax embedding | - 
**GAT** | node label softmax embedding | - 
**HAN** | user-node label softmax embedding | - 
**Semi Bipartite GraphSage** | user-node label softmax embedding | - 
**Unsupervised Bipartite GraphSage** | - | node embedding
**HGAT** | - | node embedding
**IGMC** | src dst label | -
