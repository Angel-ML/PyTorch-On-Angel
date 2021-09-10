import os
import shutil


read_path = "cora"
save_path = "generate"
cora_cites = os.path.join(read_path, "cora.cites")
cora_content = os.path.join(read_path, "cora.content")
EdgeTable = os.path.join(save_path, "EdgeTable")
FeatureTable = os.path.join(save_path, "FeatureTable")
LabelTable = os.path.join(save_path, "LabelTable")

# generate EdgeTable FeatureTable LabelTable
# ======================================================================
# EdgeTable
shutil.copyfile(cora_cites, EdgeTable)

# ======================================================================
# LabelTable
txt = []
with open(cora_content, "r") as fin:
    lines = fin.readlines()
    for line in lines:
        cols = line.split('\t')
        paper = cols[0]
        label = cols[-1].split('\n')[0]
        txt.append(paper+'\t'+label)
        # print(paper+'\t'+label)

f = open(LabelTable, "w")
for i in range(len(txt)):
    print(txt[i], file=f)
f.close()

# # check LabelTable
# with open(LabelTable, "r") as f:
#     data = f.read()
#     print(data)

# ======================================================================
# FeatureTable
txt = []
with open(cora_content, "r") as f:
    lines = f.readlines()
    for line in lines:
        cols = line.split('\t')
        index = 0
        nolabel = ""
        for i in cols:
            if index < len(cols) - 2:
                nolabel += i + '\t'
            elif index == len(cols) - 2:
                # 不加\t
                nolabel += i
            else:
                # 不加最后一列
                pass
            index += 1
        txt.append(nolabel)
        # print(nolabel)

f = open(FeatureTable, "w")
for i in range(len(txt)):
    print(txt[i], file=f)
f.close()

# # check FeatureTable
# with open(FeatureTable, "r") as f:
#     data = f.read()
#     print(data)
