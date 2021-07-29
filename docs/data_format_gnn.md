## Data Format for GNN Algorithm
> There are two format for GNN Algorithm, such as: dense and sparse

### Data Format
    
Contents | Data Format | Format Demo | Data Demo 
---------------- | --------------- | --------------- | ---------------
edge data table | - | ```src dst```, seperated by space/comma/tab | ```0 1```
edge data table with type or rating | - | ```src dst type/rating``` | ```0 1 0```
edge data table with edge feature | dense | ```src\tdst\tv1 v2 v3``` | ```0	1	0.3 0.5 1 2```
edge data table with edge feature | sparse | ```src\tdst\tf1:v1 f2:v2 f3:v3``` | ```0	1	1:1 2:1 5:2.1```
node feature data table | dense | ```node\tv1 v2 v3``` | ```0	0.1 0.3 1.3```
node feature data table | sparse | ```node\tf1:v1 f2:v2 f3:v3``` | ```0	1:1 3:2 5:1.2```
node label data table | - | ```node label```,the label table may only contain a small set of node-label pairs. Each line of the label file is a node-label pair where space is used as the separator between node and label. | ```0 1```

**Note**:  
- Data format should be same in one job, if there are more than one feature data or edge data with feature.
- Note that, each node contained in the edge table should has a feature line in the feature table file.
- High-Sparse data, the format is same to sparse, the only difference is that there is field(s) in high-sparse data and each field must has value.