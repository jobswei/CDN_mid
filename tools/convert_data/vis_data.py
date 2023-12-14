import cv2
import json
import os
import os.path as osp

data_root="./data/demo_video_fps2/hico"
with open(osp.join("./data/demo_video_fps2/hico/all_hico.json")) as fp:
    lis=json.load(fp)

toVis=lis[0]
for i in lis:
    if i["file_name"]=="1_10211004591245.jpg":
        toVis=i
        break
file_path=osp.join(osp.join(data_root,"images"),toVis["file_name"])
image=cv2.imread(file_path)

for num,obj in enumerate(toVis["annotations"]):
    (x1, y1, x2, y2),category=tuple(map(int,obj["bbox"])),int(obj["category_id"])

    color=(0, 255, 0) if category==1 else (255,0,0)
    cv2.rectangle(image, (x1, y1), (x2, y2),color , 2)
    cv2.putText(image, str(num) , (int((x1+x2)/2),y1),cv2.FONT_HERSHEY_SIMPLEX ,1, color, 2)
cv2.imwrite(osp.join(f"./outputs",f"{toVis['file_name']}"), image)