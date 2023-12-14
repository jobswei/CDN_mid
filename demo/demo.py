import os
import cv2
import sys
sys.path.append("/home/wzy/CDN_mod")
import time
import json
import torch
import base64
import random
import datetime
import argparse
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from pathlib import Path
from models import build_model
import datasets.transforms as T
from torch.utils.data import DataLoader, DistributedSampler



def make_hico_transforms():

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return T.Compose([
        T.RandomResize([800], max_size=1333),
        normalize,
    ])

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr_drop', default=60, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers_hopd', default=3, type=int,
                        help="Number of hopd decoding layers in the transformer")
    parser.add_argument('--dec_layers_interaction', default=3, type=int,
                        help="Number of interaction decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # HOI
    parser.add_argument('--num_obj_classes', type=int, default=80,
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes', type=int, default=117,
                        help="Number of verb classes")
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--subject_category_id', default=0, type=int)
    parser.add_argument('--verb_loss_type', type=str, default='focal',
                        help='Loss type for the verb classification')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--use_matching', action='store_true',
                        help="Use obj/sub matching 2class loss in first decoder, default not use")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=2.5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=1, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_obj_class', default=1, type=float,
                        help="Object class coefficient in the matching cost")
    parser.add_argument('--set_cost_verb_class', default=1, type=float,
                        help="Verb class coefficient in the matching cost")
    parser.add_argument('--set_cost_matching', default=1, type=float,
                        help="Sub and obj box matching coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=2.5, type=float)
    parser.add_argument('--giou_loss_coef', default=1, type=float)
    parser.add_argument('--obj_loss_coef', default=1, type=float)
    parser.add_argument('--verb_loss_coef', default=2, type=float)
    parser.add_argument('--alpha', default=0.5, type=float, help='focal loss alpha')
    parser.add_argument('--matching_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--hoi_path', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda:2',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # decoupling training parameters
    parser.add_argument('--freeze_mode', default=0, type=int)
    parser.add_argument('--obj_reweight', action='store_true')
    parser.add_argument('--verb_reweight', action='store_true')
    parser.add_argument('--use_static_weights', action='store_true', 
                        help='use static weights or dynamic weights, default use dynamic')
    parser.add_argument('--queue_size', default=4704*1.0, type=float,
                        help='Maxsize of queue for obj and verb reweighting, default 1 epoch')
    parser.add_argument('--p_obj', default=0.7, type=float,
                        help='Reweighting parameter for obj')
    parser.add_argument('--p_verb', default=0.7, type=float,
                        help='Reweighting parameter for verb')

    # hoi eval parameters
    parser.add_argument('--use_nms_filter', action='store_true', help='Use pair nms filter, default not use')
    parser.add_argument('--thres_nms', default=0.7, type=float)
    parser.add_argument('--nms_alpha', default=1.0, type=float)
    parser.add_argument('--nms_beta', default=0.5, type=float)
    parser.add_argument('--json_file', default='results.json', type=str)

    return parser

class Demo:
    def __init__(self,predictor,output_dir,obj_path,verb_path):
        self.predictor=predictor
        self.obj_path=obj_path
        self.verb_path=verb_path
        self.out_dir=output_dir

    def filter_res_score(self,preds,thre=0.02):
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

    def filter_res_num(self,preds, num=5, score=0):
        hois = preds["hoi_prediction"]
        hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        hois = preds["hoi_prediction"] = hois[:num]
        preds = self.filter_res_score(preds, score)
        return preds

    def show_hoi(self,preds, obj_path, verb_path):

        def getLis(filename):
            with open(filename, "r") as fp:
                lis = fp.read().strip().split("\n")
            lis = [i.strip().split(" ")[-1] for i in lis]
            return lis

        obj_lis = getLis(obj_path)
        verb_lis = getLis(verb_path)
        pairs = []
        hois = preds["hoi_prediction"]
        bboxes = preds["predictions"]
        for hoi in hois:
            sub, obj = hoi["subject_id"], hoi["object_id"]
            pair = []
            pair.append(obj_lis[bboxes[sub]["category_id"]] + str(sub))
            pair.append(obj_lis[bboxes[obj]["category_id"]] + str(obj))
            pair.append(verb_lis[hoi["category_id"]])
            pairs.append(pair)
        # print(*pairs, sep="\n")
        return pairs

    def _demo(self,root=None,source=None,is_base64=False,savename=None,savejson=True,saveimage=False,num=5,score=0.1):
        """
        root:指图片所在文件夹，当is_base64为False时生效
        source:在is_base64为False时指图片名称，在is_base64为True时指base64字符串
        is_base64:指明传入的时base64字符串还是图片路径
        save_name:在is_base64为True时有效，为想要保存的文件名，也即json的文件名
        savejson:是否在out_dir下保存json文件
        """
        if is_base64:
            savename=savename
        else:
            savename=source
        preds=self.predictor.predict(root=root,source=source,is_base64=is_base64,savename=savename)[0]
        # import ipdb;ipdb.set_trace()
        preds=self.filter_res_num(preds,num=num,score=score)
        # print(preds)
        if savejson:
            with open(os.path.join(self.out_dir,savename.rstrip(".jpg")+".json"),"w") as f:
                f.write(str(preds))
        # display(preds)
        # print(preds["filename"])
        if saveimage:
            pairs=self.show_hoi(preds,self.obj_path,self.verb_path)
            if is_base64:
                decoded_bytes = base64.b64decode(source)
                img = Image.open(BytesIO(decoded_bytes))
                img = np.array(img)
                image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                file_path=os.path.join(root,source)
                image = cv2.imread(file_path)

            for num,obj in enumerate(preds["predictions"]):
                (x1, y1, x2, y2),category=tuple(map(int,obj["bbox"])),int(obj["category_id"])
                color=(0, 255, 0) if category==0 else (255,0,0)
                cv2.rectangle(image, (x1, y1), (x2, y2),color , 2)
                cv2.putText(image, str(num) , (int((x1+x2)/2),y1),cv2.FONT_HERSHEY_SIMPLEX ,1, color, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            x, y = 800, 30  # 留出一些边距
            font_scale = 1
            color = (0, 255, 255)  # BGR
            thickness = 2
            for pair in pairs:
                text=f"{pair[0]}  {pair[1]}  {pair[2]}"
                cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
                # 更新 y 坐标以便下一行文字不会覆盖前一行
                y += 25
            # print(self.out_dir,f"{preds['filename']}")
            cv2.imwrite(os.path.join(self.out_dir,f"{preds['filename']}"), image)
        return preds

    def __call__(self,root=None,source=None,is_base64=False,savename=None,savejson=True,saveimage=False,num=5,score=0.1):
        if type(source)==list:
            for i,name in tqdm(enumerate(source)):
                jsonfile=self._demo(root=root,source=name,is_base64=is_base64,savename=str(i)+savename,savejson=savejson,saveimage=saveimage,num=num,score=score)
        elif type(source)==str:
            jsonfile=self._demo(root=root,source=source,is_base64=is_base64,savename=savename,savejson=savejson,saveimage=saveimage,num=num,score=score)
        return jsonfile

class Predictor:
    def __init__(self, model, postprocessors, args, hoi_thre=0.02) -> None:
        self.device = args.device
        self.thres_nms = args.thres_nms
        self.nms_alpha = args.nms_alpha
        self.nms_beta = args.nms_beta
        # self.correct_mat = np.load(correct_mat_path)
        self.model = model
        self.postprocessors = postprocessors
        self.hoi_thre = hoi_thre
        self.model.eval()

    def getLis(self, filename):
        with open(filename, "r") as fp:
            lis = fp.read().strip().split("\n")
        lis = [i.strip().split(" ")[-1] for i in lis]
        return lis

    def triplet_nms_filter(self, preds):
        preds_filtered = []
        for img_preds in preds:
            pred_bboxes = img_preds['predictions']
            pred_hois = img_preds['hoi_prediction']
            all_triplets = {}
            for index, pred_hoi in enumerate(pred_hois):
                triplet = str(pred_bboxes[pred_hoi['subject_id']]['category_id']) + '_' + \
                          str(pred_bboxes[pred_hoi['object_id']]['category_id']) + '_' + str(pred_hoi['category_id'])

                if triplet not in all_triplets:
                    all_triplets[triplet] = {'subs': [], 'objs': [], 'scores': [], 'indexes': []}
                all_triplets[triplet]['subs'].append(pred_bboxes[pred_hoi['subject_id']]['bbox'])
                all_triplets[triplet]['objs'].append(pred_bboxes[pred_hoi['object_id']]['bbox'])
                all_triplets[triplet]['scores'].append(pred_hoi['score'])
                all_triplets[triplet]['indexes'].append(index)

            all_keep_inds = []
            for triplet, values in all_triplets.items():
                subs, objs, scores = values['subs'], values['objs'], values['scores']
                keep_inds = self.pairwise_nms(np.array(subs), np.array(objs), np.array(scores))

                keep_inds = list(np.array(values['indexes'])[keep_inds])
                all_keep_inds.extend(keep_inds)

            preds_filtered.append({
                'filename': img_preds['filename'],
                'predictions': pred_bboxes,
                'hoi_prediction': list(np.array(img_preds['hoi_prediction'])[all_keep_inds])
            })

        return preds_filtered

    def pairwise_nms(self, subs, objs, scores):
        sx1, sy1, sx2, sy2 = subs[:, 0], subs[:, 1], subs[:, 2], subs[:, 3]
        ox1, oy1, ox2, oy2 = objs[:, 0], objs[:, 1], objs[:, 2], objs[:, 3]

        sub_areas = (sx2 - sx1 + 1) * (sy2 - sy1 + 1)
        obj_areas = (ox2 - ox1 + 1) * (oy2 - oy1 + 1)

        order = scores.argsort()[::-1]

        keep_inds = []
        while order.size > 0:
            i = order[0]
            keep_inds.append(i)

            sxx1 = np.maximum(sx1[i], sx1[order[1:]])
            syy1 = np.maximum(sy1[i], sy1[order[1:]])
            sxx2 = np.minimum(sx2[i], sx2[order[1:]])
            syy2 = np.minimum(sy2[i], sy2[order[1:]])

            sw = np.maximum(0.0, sxx2 - sxx1 + 1)
            sh = np.maximum(0.0, syy2 - syy1 + 1)
            sub_inter = sw * sh
            sub_union = sub_areas[i] + sub_areas[order[1:]] - sub_inter

            oxx1 = np.maximum(ox1[i], ox1[order[1:]])
            oyy1 = np.maximum(oy1[i], oy1[order[1:]])
            oxx2 = np.minimum(ox2[i], ox2[order[1:]])
            oyy2 = np.minimum(oy2[i], oy2[order[1:]])

            ow = np.maximum(0.0, oxx2 - oxx1 + 1)
            oh = np.maximum(0.0, oyy2 - oyy1 + 1)
            obj_inter = ow * oh
            obj_union = obj_areas[i] + obj_areas[order[1:]] - obj_inter

            ovr = np.power(sub_inter / sub_union, self.nms_alpha) * np.power(obj_inter / obj_union, self.nms_beta)
            inds = np.where(ovr <= self.thres_nms)[0]

            order = order[inds + 1]
        return keep_inds

    def predict(self, root=None, source=None,is_base64=False,savename=None):
        transform = make_hico_transforms()

        if is_base64:
            decoded_bytes = base64.b64decode(source)
            img = Image.open(BytesIO(decoded_bytes))
        else:
            filepath = os.path.join(root, source)
            img = Image.open(filepath)
        samples, _ = transform(img, None)
        samples = samples.to(self.device)
        samples = samples.unsqueeze(dim=0)
        outputs = self.model(samples)
        img_size = list(img.size)
        img_size.reverse()
        orig_target_sizes = torch.tensor([img_size])
        results = self.postprocessors['hoi'](outputs, orig_target_sizes)

        # correct_mat = self.correct_mat
        # import ipdb;ipdb.set_trace()
        preds = []
        for index, img_preds in enumerate(results):
            img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items()}
            bboxes = [{'bbox': list(bbox), 'category_id': label} for bbox, label in zip(img_preds['boxes'], img_preds['labels'])]
            hoi_scores = img_preds['verb_scores']
            verb_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
            subject_ids = np.tile(img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
            object_ids = np.tile(img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T

            hoi_scores = hoi_scores.ravel()
            verb_labels = verb_labels.ravel()
            subject_ids = subject_ids.ravel()
            object_ids = object_ids.ravel()

            if len(subject_ids) > 0:
                # object_labels = np.array([bboxes[object_id]['category_id'] for object_id in object_ids])
                # import ipdb;ipdb.set_trace()
                # masks = correct_mat[verb_labels, object_labels]
                # hoi_scores *= masks

                hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                        subject_id, object_id, category_id, score in zip(subject_ids, object_ids, verb_labels, hoi_scores)]
                hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
                hois = hois[:100]
            else:
                hois = []
            if is_base64:
                preds.append({
                    'filename': savename,
                    'predictions': bboxes,
                    'hoi_prediction': hois
                })
            else:
                preds.append({
                    'filename': source,
                    'predictions': bboxes,
                    'hoi_prediction': hois
                })

        preds = self.triplet_nms_filter(preds)
        preds[0]["hoi_prediction"].sort(key=lambda k: (k.get('score', 0)), reverse=True)
        return preds


def main(pretrained=None,device=None,out_dir=None,root=None,source=None,is_base64=True,savename=None,savejson=True,saveimage=False,score=0.1,num=5):
    """
    root:指图片所在文件夹，当is_base64为False时生效
    source:在is_base64为False时指图片名称，在is_base64为True时指base64字符串
    is_base64:指明传入的时base64字符串还是图片路径
    save_name:在is_base64为True时有效，为想要保存的文件名，也即json的文件名
    savejson:是否在out_dir下保存json文件
    """
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # args.pretrained="work_dir/1205_cdn_l_lr_2e-5/checkpoint_best.pth"
    args.pretrained = pretrained
    args.dataset_file = "hico"
    # args.hoi_path = "./data/demo_video_fps2/hico2"
    # args.hoi_path="/home/wzy/CDN_mod/data/hico_20160224_det"
    args.num_obj_classes = 2
    args.num_verb_classes = 1
    args.backbone = "resnet50"
    args.num_queries = 64
    args.dec_layers_hopd = 3
    args.dec_layers_interaction = 0
    args.use_nms_filter = True
    args.batch_size = 2
    args.device = device
    args.use_matching = True

    # args=dict()
    # print(type(args))
    device = torch.device(args.device)
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    checkpoint = torch.load(args.pretrained, map_location='cpu')
    # print(checkpoint['model'].keys())
    # del checkpoint['model']['matching_embed.weight']
    # del checkpoint['model']['matching_embed.bias']
    model.load_state_dict(checkpoint['model'])
    predictor=Predictor(model,postprocessors,
            args=args)
    if saveimage:
        demor=Demo(predictor,
                   out_dir,
                   "./data/demo_video_fps2/hico/annotations/objs.txt",
                   "./data/demo_video_fps2/hico/annotations/verb.txt")
    else:
        demor=Demo(predictor,
                   out_dir,
                   None,
                   None)

    jsonfile=demor(root=root,
                   source=source,
                   is_base64=is_base64,
                   savename=savename,
                   savejson=savejson,
                   saveimage=saveimage, 
                   score=score, 
                   num=num)

    return jsonfile






if __name__ == '__main__':


    path = "./demo/test.jpg"
    # encoded_strings=[]
    # for i in tqdm(range(100)):
    #     with open(path, "rb") as image_file:
    #         # 将图片文件的内容进行Base64编码
    #         encoded_string = base64.b64encode(image_file.read()).decode()
    #         encoded_strings.append(encoded_string)
    with open(path, "rb") as image_file:
        # 将图片文件的内容进行Base64编码
        encoded_string = base64.b64encode(image_file.read()).decode()


    jsonfile=main(pretrained="/home/airport/CDN_mod/work_dir/1206_train_oneobj/checkpoint_best.pth",
                  device="cuda:4",
                  out_dir="./demo/out",
                  root="./demo",
                  source=encoded_string,
                  is_base64=True,
                  savename="test.jpg",
                  savejson=False,
                  saveimage=False, 
                  score=0.01, 
                  num=5)
    print(jsonfile)
