import json
import cv2
import os.path as osp


def filter_res_score(preds,thre=0.02):
    hoi_new=[]
    preds_new=[]
    id_map=dict()
    num=0
    for hoi in preds["hoi_prediction"]:
        if hoi["score"]>thre:
            if hoi["subject_id"] not in id_map:
                preds_new.append(preds["predictions"][hoi["subject_id"]])
                id_map[hoi["subject_id"]]=num
                num+=1
            if hoi["object_id"] not in id_map:
                preds_new.append(preds["predictions"][hoi["object_id"]])
                id_map[hoi["object_id"]]=num
                num+=1
            hoi["subject_id"]=id_map[hoi["subject_id"]]
            hoi["object_id"]=id_map[hoi["object_id"]]
            hoi_new.append(hoi)

    preds["predictions"]=preds_new
    preds["hoi_prediction"]=hoi_new
    return preds

def filter_res_num(preds,num=5,score=0):
    hois=preds["hoi_prediction"]
    hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
    hois=preds["hoi_prediction"]=hois[:num]
    # import ipdb;ipdb.set_trace()
    preds=filter_res_score(preds,score)
    return preds
    
def display(preds):
    print("filename:")
    print(preds["filename"])
    print(f"object detection: {len(preds['predictions'])} objs")
    print(*preds["predictions"],sep="\n")
    print(f"hoi pred: {len(preds['hoi_prediction'])} hois")
    print(*preds["hoi_prediction"],sep="\n")
def show_hoi(preds, obj_path, verb_path):
    def getLis(filename):
        with open(filename,"r") as fp:
            lis=fp.read().strip().split("\n")
        lis=[i.strip().split(" ")[-1] for i in lis]
        return lis
    obj_lis=getLis(obj_path)
    verb_lis=getLis(verb_path)
    pairs=[]
    hois=preds["hoi_prediction"]
    bboxes=preds["predictions"]
    for hoi in hois:
        sub,obj=hoi["subject_id"],hoi["object_id"]
        pair=[]
        pair.append(obj_lis[bboxes[sub]["category_id"]]+str(sub))
        pair.append(obj_lis[bboxes[obj]["category_id"]]+str(obj))
        pair.append(verb_lis[hoi["category_id"]])
        pairs.append(pair)
    print(*pairs,sep="\n")
    return pairs


if __name__=="__main__":
    with open("./results.json","r") as fp:
        file=eval(json.load(fp))

    preds_all=file["preds"]

    for preds in preds_all:
        preds=filter_res_num(preds,num=1)
        display(preds)
        print(preds["filename"])
        pairs=show_hoi(preds)
        print()

        file_path=osp.join("/home/wzy/CDN/data/hico_20160224_det/images/test2015/",preds["filename"])
        image=cv2.imread(file_path)

        for num,obj in enumerate(preds["predictions"]):
            (x1, y1, x2, y2),category=tuple(map(int,obj["bbox"])),int(obj["category_id"])
            color=(0, 255, 0) if category==0 else (255,0,0)
            cv2.rectangle(image, (x1, y1), (x2, y2),color , 2)
            cv2.putText(image, str(num) , (int((x1+x2)/2),y1),cv2.FONT_HERSHEY_SIMPLEX ,1, color, 2)
        cv2.imwrite(osp.join(f"./outputs/",f"{preds['filename']}"), image)
