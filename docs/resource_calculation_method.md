## How to calculate the resource

### Dense data or low dimentition sparse data (dim less than 2000 generally)
- Angel PS resource: in order to ensure that Angel does not hang up, it is necessary to configure memory that is about twice the size of the model. The formula for calculating the size of the graph feature is: node_num * feature_dim * 4Byte, such as: 1kw nodes, feature_dim is 100, the size of graph feature is about 4G, then set ps.instances=3, ps.memory=4G is reasonable(**total memory is 2~3 times of data saved in ps**). Of course, this is only a simple case. In many algorithms, there are many types of data saved in ps, such as: edge, node feature, node label etc.

	Algo Name | Data saved on ps | Resource calculation Method (**total memory is 2~3 times of data saved in ps**)
	---------------- | --------------- | ---------------
	**Semi GraphSage/GAT** | edge, node feature, node label | (edge\_num * 2 * 8Byte + node\_num * feature\_dim * 4Byte + the num of node with label * 4Byte) * 2 or 3
	**RGCN/HAN** | edge with type(dst type), node feature, node label | (edge\_num * 2 * 8Byte + edge\_num * 4Byte + node\_num * feature\_dim * 4Byte + the num of node with label * 4Byte) * 2 or 3
	**RGCN/HAN** | edge with type(dst type), node feature, node label | (edge\_num * 2 * 8Byte + edge\_num * 4Byte + node\_num * feature\_dim * 4Byte + the num of node with label * 4Byte) * 2 or 3
	**DGI/Unsupervised GraphSage** | edge, node feature | (edge\_num * 2 * 8Byte + node\_num * feature\_dim * 4Byte) * 2 or 3
	**Semi Bipartite GraphSage** | edge * 2, user feature, item feature, user label | (edge\_num * 2 * 8Byte * 2 + user\_num * user\_feature_dim * 4Byte + item\_num * item\_feature\_dim * 4Byte + user\_label\_num * 4Byte) * 2 or 3
	**Unsupervised Bipartite GraphSage** | edge * 2, user feature, item feature | (edge\_num * 2 * 8Byte * 2 + user_num * user\_feature\_dim * 4Byte + item\_num * item_feature_dim * 4Byte) * 2 or 3
	**IGMC** | edge with type, user feature, item feature | (edge\_num * 2 * 8Byte + edge\_num * 4Byte + user\_num * user\_feature\_dim * 4Byte + item\_num * item_feature\_num * 4Byte) * 2 or 3

- Spark Executor Resource: the configuration of Spark resources is mainly considered from the aspect of training data(Edge data is usually saved on Spark Executor), and it is best to save **2~3 times** the input data. If the memory is tight, 1x is acceptable, but it will be relatively slow. For example, a 10 billion edge set is about 200G in size, and a 30G * 20 configuration is sufficient. 

### High-Sparse data(data has field)
Only support for four algorithms, such as: **Semi GraphSage, Semi Bipartite GraphSage, HGAT and HAN**.
Resources of Angel PS and Spark Executor is similar to Dense data, the only difference is that there is low-dimension embedding matrix for high-sparse data.
low-dimension embedding: input\_embedding\_dim * slots * input\_dim * 4Byte

- `input_embedding_dim`: the dimension of embedding, which is usually low, such as:8; (As for Bipartite GNN or Heterogeneous GNN, the parameter is input\_user\_embedding\_dim or input\_item\_embedding\_dim)
- `slots`: related to the optimizer, the default optimizer adam, slots = 3
- `input_dim`: the dim of input feature; (As for Bipartite GNN or Heterogeneous GNN, the parameter is input\_user\_dim or input\_item\_dim, which is the dim of user's feature or item's feature)
- `input_field_num`: the number of features with value; (As for Bipartite GNN or Heterogeneous GNN, the parameter is input\_user\_field\_num or input\_item\_field\_num, which is the number of user's feature with value or item's feature with value)

- Angel PS Resource:in order to ensure that Angel does not hang up, it is necessary to configure memory that is about twice the size of the model.

	Algo Name | Data saved on ps | Resource calculation Method (**total memory is 2~3 times of data saved in ps**)
	---------------- | --------------- | ---------------
	**Semi GraphSage** | edge, node feature, node label, embedding matrix | (edge\_num * 2 * 8Byte + node\_num * `field_num` * 4Byte + the num of node with label * 4Byte + `input_embedding_dim * slots * input_dim * 4Byte`) * 2 or 3
	**Semi Bipartite GraphSage** | edge * 2, user feature, item feature, user label, user embedding matrix, item embedding matrix | (edge\_num * 2 * 8Byte * 2 + user\_num * `user_field_num` * 4Byte + item\_num * `item_field_num` * 4Byte + user\_label\_num * 4Byte + `user_embedding_dim * slots * user_feature_dim * 4Byte + item_embedding_dim * slots * item_feature_dim * 4Byte`) * 2 or 3
	**HGAT** | edge * 2, user feature, item feature, user embedding matrix, item embedding matrix | (edge\_num * 2 * 8Byte * 2 + user\_num * `user_field_num` * 4Byte + item\_num * `item_field_num` * 4Byte + `user_embedding_dim * slots * user_feature_dim * 4Byte + item_embedding_dim * slots * item_feature_dim * 4Byte`) * 2 or 3
	**HAN** | edge * 2, user feature, user embedding matrix | (edge\_num * 2 * 8Byte * 2 + user\_num * `user_field_num` * 4Byte + `user_embedding_dim * slots * user_feature_dim * 4Byte`) * 2 or 3
	

- Spark Executor Resource:the configuration of Spark resources is mainly considered from the aspect of training data(Edge data is usually saved on Spark Executor), and it is best to save **2~3 times** the input data. If the memory is tight, 1x is acceptable, but it will be relatively slow. For example, a 10 billion edge set is about 200G in size, and a 30G * 20 configuration is sufficient. 