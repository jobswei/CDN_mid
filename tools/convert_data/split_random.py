
# 分割数据集
import random
import shutil 
import os
import os.path as osp
import json
import tqdm

def init_dir(name):
    if osp.exists(name):
        shutil.rmtree(name)
    os.makedirs(name)

data_root="/home/wzy/CDN/data/demo_video_fps2/hico"

with open(osp.join(data_root,"all_hico.json"),"r") as fp:
    lis=json.load(fp)

split_ratio = 0.8  
split_point = int(len(lis) * split_ratio)
random.shuffle(lis)

first_part = lis[:split_point]
second_part = lis[split_point:]
with open(osp.join(data_root,"annotations/trainval_hico.json"),"w") as fp:
    json.dump(first_part,fp)
with open(osp.join(data_root,"annotations/test_hico.json"),"w") as fp:
    json.dump(second_part,fp)


with open(osp.join(data_root,"annotations/trainval_hico.json")) as fp:
    origin_lis=json.load(fp)
init_dir(osp.join(data_root,"images/train2015"))
name_lis=set()
for i in origin_lis:
    name_lis.add(i["file_name"])
for name in tqdm.tqdm(name_lis):
    shutil.copyfile(osp.join(data_root,"images",name),osp.join(data_root,"images/train2015",name))


with open(osp.join(data_root,"annotations/test_hico.json")) as fp:
    origin_lis=json.load(fp)
init_dir(osp.join(data_root,"images/test2015"))
name_lis=set()
for i in origin_lis:
    name_lis.add(i["file_name"])
for name in tqdm.tqdm(name_lis):
    shutil.copyfile(osp.join(data_root,"images",name),osp.join(data_root,"images/test2015",name))
