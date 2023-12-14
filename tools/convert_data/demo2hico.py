import os
import os.path as osp
import json
import tqdm

def filter_bbox(ann:dict):
    new_ann={}
    new_ann["id"]=ann["id"]
    try:
        new_ann["class"]=ann["value"]["rectanglelabels"][0]
    except:
        new_ann["class"]="unknown"
        print(ann["id"])
        print("a unknown")
    val:dict=ann["value"]
    del val["rotation"]
    del val["rectanglelabels"]
    new_ann["bbox"]=val
    return new_ann
def filter_hoi(ann:dict):
    new_ann={}
    new_ann["subject_id"]=ann["from_id"]
    new_ann["object_id"]=ann["to_id"]
    new_ann["category_id"]=-1
    return new_ann

def convert_bbox(ann:dict):
    class_map={"person":0, "luggage":1,"securitybox":2,"unknown":-1}
    new_ann={}
    new_ann["hoi_id"]=ann["id"]
    new_ann["bbox"]=list(map(lambda x:x/100,list(ann["bbox"].values())))
    new_ann["category_id"]=class_map[ann["class"]]
    return new_ann
def convert_hoi(ann:dict):
    new_ann={}
    new_ann["subject_id"]=ann["subject_id"]
    new_ann["object_id"]=ann["object_id"]
    new_ann["category_id"]=0
    return new_ann

def extract_useful_info1(img_information):
    useful_info={}
    useful_info["id"]=img_information["id"]
    useful_info["annotations"]=img_information["annotations"][0]["result"]
    useful_info["data"]=img_information["data"]
    return useful_info

def extract_useful_info2(img_information):
    useful_info={}
    useful_info["id"]=img_information["id"]
    useful_info["img"]=osp.basename(img_information["data"]["image"])
    useful_info["heigth"]=1080
    useful_info["width"]=1920
    useful_info["bboxes"]=[]
    useful_info["hois"]=[]
    for ann in img_information["annotations"]:
        if ann["type"]=="rectanglelabels":
            ann=filter_bbox(ann)
            useful_info["bboxes"].append(ann)
        elif ann["type"]=="relation":
            ann=filter_hoi(ann)
            useful_info["hois"].append(ann)
        else:
            raise ValueError("Unknown Type!!")
    return useful_info

def convert_data_format(img_information):
    useful_info={}
    useful_info["filename"]=img_information["img"]
    # useful_info["heigth"]=1080
    # useful_info["width"]=1920
    useful_info["hoi_annotation"]=[]
    useful_info["annotations"]=[]
    for ann in img_information["bboxes"]:
        ann=convert_bbox(ann)
        useful_info["annotations"].append(ann)
    for ann in img_information["hois"]:
        ann=convert_hoi(ann)
        useful_info["hoi_annotation"].append(ann)
    return useful_info

def extract_bbox_reid(img_information):
    useful_info={}
    useful_info["file_name"]=img_information["filename"]
    useful_info["hoi_annotation"]=[]
    useful_info["annotations"]=[]
    hoi_dic={}
    bbox_dic={}
    id_map={}
    for bbox in img_information["annotations"]:
        id=bbox["hoi_id"]
        del bbox["hoi_id"]
        bbox_dic[id]=bbox
    for hoi in img_information["hoi_annotation"]:
        sub_id,obj_id=hoi["subject_id"],hoi["object_id"]
        if sub_id not in id_map:
            useful_info["annotations"].append(bbox_dic[sub_id])
            id_map[sub_id]=len(useful_info["annotations"])-1
        if obj_id not in id_map:
            useful_info["annotations"].append(bbox_dic[obj_id])
            id_map[obj_id]=len(useful_info["annotations"])-1
        hoi["subject_id"]=id_map[sub_id]
        hoi["object_id"]=id_map[obj_id]
        useful_info["hoi_annotation"].append(hoi)
    return useful_info

def align(img_information):
    origin_width=1920
    origin_hight=1080
    for bbox in img_information["annotations"]:
        x,y,w,h=tuple(bbox["bbox"])
        x1,x2=x,x+w
        y1,y2=y,y+h
        bbox["bbox"]=list(map(int,[x1*origin_width,y1*origin_hight,x2*origin_width,y2*origin_hight]))
        bbox["category_id"]+=1
    for bbox in img_information["hoi_annotation"]:
        bbox["category_id"]+=1
    return img_information

data_root="/home/wzy/CDN/data/demo_video_fps2/hico"
with open(osp.join(data_root,"../project-19-full.json")) as fp:
    origin_lis=json.load(fp)

lis=[]
for img_information in tqdm.tqdm(origin_lis):
    # Extract the useful information, 1st step
    info=extract_useful_info1(img_information)
    # Extract the useful information, 2nd step
    # filter bboxes and hois
    info=extract_useful_info2(info)
    """
    cls: 0-person, 1-luggage, 2-securitybox
    hoi: all hoi is 0
    bbox: [x,y,w,h]: List[int]
    """
    info=convert_data_format(info)
    # Delete redundant bbox and ReId
    info=extract_bbox_reid(info)
    info=align(info)
    lis.append(info)
with open(osp.join(data_root,"all_hico.json"),"w") as fp:
    json.dump(lis,fp)
