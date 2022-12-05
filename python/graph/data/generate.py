import os
import shutil


read_path = "cora"
save_path = "generate"
cora_cites = os.path.join(read_path, "cora.cites")
cora_content = os.path.join(read_path, "cora.content")
EdgeTable = os.path.join(save_path, "EdgeTable")
FeatureTable = os.path.join(save_path, "FeatureTable")
LabelTable = os.path.join(save_path, "LabelTable")

# mapping
# ======================================================================
'''
    mapping [origin paper index] to [ascend num index]
    35 - 0
    40 - 1
      ...
'''


def compare_int(string):
    return int(string)


maps = {}
paperIndex = []
with open(cora_content, "r") as fin:
    lines = fin.readlines()
    for line in lines:
        cols = line.split('\t')
        idx1 = cols[0]
        if idx1 in paperIndex:
            pass
        else:
            paperIndex.append(idx1)
            # read all paper index, set map[paper index]=1 as init
    paperIndex.sort(key=compare_int)
    # paper index is sort as ['35', '40', '114', '117', '128', ...]
    index = 0
    for paper in paperIndex:
        maps[paper] = index
        index += 1
    # {'35':0 , '40':1 , ... }


# generate EdgeTable FeatureTable LabelTable
# ======================================================================
# EdgeTable
'''
    35\t40 --> 0\t1
           ...
'''
edges = []
with open(cora_cites, "r") as fin:
    lines = fin.readlines()
    for line in lines:
        cols = line.split('\t')
        idx1 = cols[0]
        idx2 = cols[1].split('\n')[0]
        edges.append(str(maps[idx1]) + '\t' + str(maps[idx2]))

with open(EdgeTable, "w") as fout:
    for i in range(len(edges)):
        print(edges[i], file=fout)
    fout.close()

# check EdgeTable
# with open(EdgeTable, "r") as fcheck:
#     data = fcheck.read()
#     print(data)

# # ======================================================================
# # LabelTable
'''
    1\tGenetic_Algorithms
    2\tGenetic_Algorithms
    3\tReinforcement_Learning
    4\tReinforcement_Learning
'''


def take_first(string):
    return int(string.split('\t')[0])


labels = []
with open(cora_content, "r") as fin:
    lines = fin.readlines()
    for line in lines:
        cols = line.split('\t')
        paper = cols[0]
        label = cols[-1].split('\n')[0]
        labels.append(str(maps[paper])+'\t'+label)
        # print(str(maps[paper])+'\t'+label)

labels.sort(key=take_first)

f = open(LabelTable, "w")
for i in range(len(labels)):
    print(labels[i], file=f)
f.close()

# check LabelTable
# with open(LabelTable, "r") as f:
#     data = f.read()
#     print(data)

# ======================================================================
# FeatureTable
'''
    1\t0 0 0 ...
    2\t0 0 0 ...
    3\t1 0 0 ...
'''
features = []
with open(cora_content, "r") as f:
    lines = f.readlines()
    for line in lines:
        cols = line.split('\t')
        index = 0
        nolabel = ""
        for i in cols:
            if index == 0:
                nolabel = str(maps[i]) + '\t'
            elif index < len(cols) - 2:
                nolabel += i + ' '
            elif index == len(cols) - 2:
                # 最后不加空格' '
                nolabel += i
            else:
                # 不加最后一列
                pass
            index += 1
        features.append(nolabel)
        # print(nolabel)

features.sort(key=take_first)

f = open(FeatureTable, "w")
for i in range(len(features)):
    print(features[i], file=f)
f.close()

# check FeatureTable
# with open(FeatureTable, "r") as f:
#     data = f.read()
#     print(data)
