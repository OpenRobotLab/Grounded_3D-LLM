import argparse
import json
from collections import defaultdict
import re
import os
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from copy import deepcopy
from collections import OrderedDict

import re
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from benchmark.evaluate_semantic_instance import evaluate as mask3d_det_evaluation
import glob
import pickle

try:
    from models.metrics.multi3drefer_evaluator import parse_prediction,Multi3DReferEvaluator
except:
    from multi3drefer_evaluator import parse_prediction,Multi3DReferEvaluator
import copy

template = OrderedDict([
    ("detection"            , []),
    ("scanrefer:unique"     , []),
    ("scanrefer:multiple"   , []),
    ("scanrefer:overall"    , []),
    ("m3dref:overall"       , []),
    ("donteval"             , []),
    ("scanqa:text_only"     , []),
    ("scanqa:with_grounding", [])])


def extract_numbers_and_string(string):
    match = re.search(r'(.*?)\[(.*?)\](.*?)', string)
    if match:
        numbers_str = match.group(2)
        numbers_list = [int(num) for num in numbers_str.split(',')]
        return [numbers_list]
    else:
        return None

def get_batch_aabb_pair_ious(batch_boxes_1_bound, batch_boxes_2_bound):
    box_1_x_min, box_1_y_min, box_1_z_min = torch.tensor_split(batch_boxes_1_bound[:, 0], 3, dim=1)
    box_1_x_max, box_1_y_max, box_1_z_max = torch.tensor_split(batch_boxes_1_bound[:, 1], 3, dim=1)
    box_2_x_min, box_2_y_min, box_2_z_min = torch.tensor_split(batch_boxes_2_bound[:, 0], 3, dim=1)
    box_2_x_max, box_2_y_max, box_2_z_max = torch.tensor_split(batch_boxes_2_bound[:, 1], 3, dim=1)
    x_a = torch.maximum(box_1_x_min, box_2_x_min)
    y_a = torch.maximum(box_1_y_min, box_2_y_min)
    z_a = torch.maximum(box_1_z_min, box_2_z_min)
    x_b = torch.minimum(box_1_x_max, box_2_x_max)
    y_b = torch.minimum(box_1_y_max, box_2_y_max)
    z_b = torch.minimum(box_1_z_max, box_2_z_max)
    zero_tensor = torch.zeros_like(x_a)
    intersection_volume = torch.maximum((x_b - x_a), zero_tensor) * torch.maximum((y_b - y_a), zero_tensor) * \
                        torch.maximum((z_b - z_a), zero_tensor)
    box_1_volume = (box_1_x_max - box_1_x_min) * (box_1_y_max - box_1_y_min) * (box_1_z_max - box_1_z_min)
    box_2_volume = (box_2_x_max - box_2_x_min) * (box_2_y_max - box_2_y_min) * (box_2_z_max - box_2_z_min)
    iou = intersection_volume / (box_1_volume + box_2_volume - intersection_volume + torch.finfo(torch.float32).eps)
    return iou.flatten()


def eval_llm_iou_score(pred,mask):
    from datasets.language_info import lang_info_data
    score = deepcopy(template)
    bbox_score_25 = deepcopy(template)
    bbox_score_50 = deepcopy(template)
    mask_score_25 = deepcopy(template)
    mask_score_50 = deepcopy(template)
    m3dref_bbox_result  = []
    assert len(mask['target_full'])==1
    for instance,gt_label in zip(pred, mask["batch_gt_inst_ids"]):
        gt_label = gt_label[1]
        
        gt_label:lang_info_data
        instance["gt_inst_ids"] = {
            "inst_ids_answer":gt_label.inst_ids_answer,
            "query_ids_answer":gt_label.query_ids_answer,
            "inst_ids_question":gt_label.inst_ids_question,
            "query_ids_question":gt_label.query_ids_question,
        }
        if "text_only" in instance['type']:
            instance["iou"] = None
            instance["bbox_iou"] = None
        else:
            # (0, ([(0, 13), (62, 78)], [[13], [27]],[[query ids]])),  
            gt_label = gt_label.inst_ids_answer
            obj_list = instance["grounding_result"]
            if gt_label is not None:
                for item in gt_label:
                    if isinstance(item, list):
                        gt_label = [num for sublist in gt_label for num in sublist]
                        break
            if isinstance(gt_label,list):
                gt_label = [element for element in gt_label if element is not None]
            if obj_list is not None:
                for item in obj_list:
                    if isinstance(item, list):
                        try:
                            obj_list = [num for sublist in obj_list for num in sublist]
                        except:
                            obj_list = None
                        break
            if isinstance(obj_list,list):
                obj_list = [element for element in obj_list if element is not None]
            predict_score =  instance["score"]
            if predict_score is not None:
                for item in predict_score:
                    if isinstance(item, list):
                        try:
                            predict_score = [num for sublist in predict_score for num in sublist]
                        except:
                            predict_score = None
                        break
            if isinstance(predict_score,list):
                predict_score = [element for element in predict_score if element is not None]
            instance["iou"] = None
            iou = None
            bbox_iou_25_f1_score = None
            bbox_iou_50_f1_score = None
            mask_iou_25_f1_score = None
            mask_iou_50_f1_score = None
            top_1_bbox_iou = None
            top_1_mask_iou = None
            save_gt_bbox = []
            save_pred_bbox = []
            if obj_list == [None]: obj_list = None
            if obj_list: # model predicts something 
                try:
                    if isinstance(gt_label,np.ndarray):
                        if gt_label.size == 0:
                            gt_label = None
                    elif gt_label and isinstance(gt_label[0],np.ndarray):
                        gt_label = list(gt_label[0])
                        if gt_label == [] or gt_label == [[]] or gt_label == [None]:
                            gt_label = None
                    elif gt_label == [] or gt_label == [[]] or gt_label == [None]:
                        gt_label = None
                except:
                    from IPython import embed;embed()
                if gt_label is None: # no gt
                    iou = 0
                    bbox_iou_25_f1_score = 0
                    bbox_iou_50_f1_score = 0
                    mask_iou_25_f1_score = 0
                    mask_iou_50_f1_score = 0
                    top_1_bbox_iou = 0
                    top_1_mask_iou = 0
                    all_pred_bbox = []
                    all_pred_mask = []
                    # for every pred mask
                    for predict_mask_id in obj_list:
                        pred_mask =mask['pred_inst_masks'][0][:,predict_mask_id].squeeze().bool()
                        pred_points = mask["original_coordinates"][0][pred_mask]
                        if pred_points.size:
                            min_vals,max_vals  = pred_points.min(axis=0),pred_points.max(axis=0)
                            pred_bbox = np.vstack((min_vals, max_vals))
                            all_pred_bbox.append(pred_bbox)
                            all_pred_mask.append(pred_mask)
                    save_pred_bbox = all_pred_bbox
                else:
                    # calculate top1
                    top1_score = -1
                    top1_obj_id = None
                    for one_score,obj in zip(predict_score,obj_list):
                        try:
                            if one_score>top1_score:
                                top1_score = one_score
                                top1_obj_id = obj
                                pred_top1_mask = mask['pred_inst_masks'][0][:,top1_obj_id].squeeze().bool() # TODO: mapping
                        except:
                            print(" ============= eval iou error=============== ")
                            print(predict_score,obj_list)
                            pred_top1_mask = torch.zeros(mask['pred_inst_masks'][0][:,0].shape()).bool() # TODO: mapping
                    # calculate mask iou
                    if mask['pred_inst_masks'][0][:,obj_list].shape[-1]!=1:
                        pred_mask = torch.any(mask['pred_inst_masks'][0][:,obj_list].squeeze(),dim=1) # TODO: mapping
                    else:
                        pred_mask = mask['pred_inst_masks'][0][:,obj_list].squeeze()
                    if mask['target_full'][0]['masks'][gt_label,:].shape[0] !=1:
                        gt_mask = torch.any(mask['target_full'][0]['masks'][gt_label,:].squeeze(),dim=0)
                    else:
                        gt_mask = mask['target_full'][0]['masks'][gt_label,:].squeeze()
                    pred_mask = pred_mask.bool()
                    gt_mask = gt_mask.bool()
                    inter = (pred_mask & gt_mask).sum()
                    outer = (pred_mask | gt_mask).sum()
                    iou = inter / (outer + 1e-8)
                    # top1 mask iou
                    inter = (pred_top1_mask & gt_mask).sum()
                    outer = (pred_top1_mask | gt_mask).sum()
                    top_1_mask_iou =  inter / (outer + 1e-8)
                    # top1 bbox iou
                    top_1_pred_points = mask["original_coordinates"][0][pred_top1_mask]

                    if top_1_pred_points.size:
                        min_vals,max_vals  = top_1_pred_points.min(axis=0),top_1_pred_points.max(axis=0)
                        pred_bbox = np.vstack((min_vals, max_vals))
                        gt_points = mask["original_coordinates"][0][gt_mask]
                        min_vals,max_vals  = gt_points.min(axis=0),gt_points.max(axis=0)
                        gt_bbox = np.vstack((min_vals, max_vals))
                        all_pred_bbox = torch.tensor(np.stack([pred_bbox],axis=0))
                        all_gt_bbox = torch.tensor(np.stack([gt_bbox],axis=0))
                        top_1_bbox_iou = get_batch_aabb_pair_ious(all_pred_bbox , all_gt_bbox).tolist()[0]
                    elif top_1_pred_points.size:
                        top_1_bbox_iou = 0.0
                        print("no top 1 bbox iou since no gt points or pred points")

                    # calculate bbox iou
                    all_pred_bbox = []
                    all_pred_mask = []
                    # for every pred mask
                    for predict_mask_id in obj_list:
                        pred_mask =mask['pred_inst_masks'][0][:,predict_mask_id].squeeze().bool()
                        pred_points = mask["original_coordinates"][0][pred_mask]
                        if pred_points.size:
                            min_vals,max_vals  = pred_points.min(axis=0),pred_points.max(axis=0)
                            pred_bbox = np.vstack((min_vals, max_vals))
                            all_pred_bbox.append(pred_bbox)
                            all_pred_mask.append(pred_mask)
                    save_pred_bbox = all_pred_bbox
                    all_gt_bbox = []
                    all_gt_mask = []
                    # for every gt mask
                    for gt_mask_id in gt_label:
                        gt_mask = mask['target_full'][0]['masks'][gt_mask_id,:].squeeze().bool()
                        gt_points = mask["original_coordinates"][0][gt_mask]
                        min_vals,max_vals  = gt_points.min(axis=0),gt_points.max(axis=0)
                        gt_bbox = np.vstack((min_vals, max_vals))
                        all_gt_bbox.append(gt_bbox)
                        all_gt_mask.append(gt_mask)
                    save_gt_bbox = all_gt_bbox
                    if all_pred_bbox != []: # have prediction and is not empty
                        all_pred_bbox = torch.tensor(np.stack(all_pred_bbox,axis=0))
                        all_gt_bbox = torch.tensor(np.stack(all_gt_bbox,axis=0))
                        all_pred_mask = torch.tensor(np.stack(all_pred_mask))
                        all_gt_mask = torch.tensor(np.stack(all_gt_mask))
                        # bbox iou
                        M, N = all_pred_bbox.size(0), all_gt_bbox.size(0)
                        all_pred_bbox_expanded = all_pred_bbox.unsqueeze(1).expand(M, N, 2, 3)
                        all_pred_bbox_repeated = all_pred_bbox_expanded.reshape(M*N, 2, 3)
                        all_gt_bbox_expanded = all_gt_bbox.unsqueeze(0).expand(M, N, 2, 3)
                        all_gt_bbox_repeated = all_gt_bbox_expanded.reshape(M*N, 2, 3)
                        bbox_iou = np.array(get_batch_aabb_pair_ious(all_pred_bbox_repeated, all_gt_bbox_repeated).view(M, N))
                        max_dim = max(M, N)
                        padded_array = np.zeros((max_dim, max_dim))
                        padded_array[:M, :N] = bbox_iou
                        bbox_iou = padded_array
                        row_idx, col_idx = linear_sum_assignment(bbox_iou * -1)
                        iou_25_tp = 0
                        iou_50_tp = 0
                        # iterate matched pairs, check ious
                        for i in range(M):
                            i_iou = bbox_iou[row_idx[i], col_idx[i]]
                            # calculate true positives
                            if i_iou >= 0.25:
                                iou_25_tp += 1
                            if i_iou >= 0.5:
                                iou_50_tp += 1
                        # calculate precision, recall and f1-score for the current scene
                        bbox_iou_25_f1_score = 2 * iou_25_tp / (N+M)
                        bbox_iou_50_f1_score = 2 * iou_50_tp / (N+M)
                        # mask iou
                        L = all_gt_mask.size(1)
                        all_pred_mask_expanded = all_pred_mask.unsqueeze(1).expand(M,N,L)
                        all_pred_mask_repeated = all_pred_mask_expanded.reshape(M*N,L)
                        all_gt_mask_expanded = all_gt_mask.unsqueeze(0).expand(M, N, L)
                        all_gt_mask_repeated = all_gt_mask_expanded.reshape(M*N,L)
                        intersection = (all_pred_mask_repeated & all_gt_mask_repeated).sum(dim=1)
                        union = (all_pred_mask_repeated | all_gt_mask_repeated).sum(dim=1)
                        mask_iou = intersection / (union + 1e-8)
                        mask_iou = mask_iou.view(M,N)
                        padded_array = np.zeros((max_dim, max_dim))
                        padded_array[:M, :N] = mask_iou
                        mask_iou = padded_array
                        row_idx, col_idx = linear_sum_assignment(bbox_iou * -1)
                        iou_25_tp = 0
                        iou_50_tp = 0
                        # iterate matched pairs, check ious
                        for i in range(M):
                            i_iou = bbox_iou[row_idx[i], col_idx[i]]
                            # calculate true positives
                            if i_iou >= 0.25:
                                iou_25_tp += 1
                            if i_iou >= 0.5:
                                iou_50_tp += 1
                        # calculate precision, recall and f1-score for the current scene
                        mask_iou_25_f1_score = 2 * iou_25_tp / (N+M)
                        mask_iou_50_f1_score = 2 * iou_50_tp / (N+M)
                    else: # have prediction but is empty
                        bbox_iou_25_f1_score = 0.0
                        bbox_iou_50_f1_score = 0.0
                        mask_iou_25_f1_score = 0
                        mask_iou_50_f1_score = 0
            else:  # no prediction
                if isinstance(gt_label,np.ndarray):
                    if gt_label.size == 0:
                        gt_label = None
                elif gt_label and isinstance(gt_label[0],np.ndarray):
                    gt_label = list(gt_label[0])
                    if gt_label == [] or gt_label == [[]] or gt_label == [None]:
                        gt_label = None
                elif gt_label == [] or gt_label == [[]] or gt_label == [None]:
                    gt_label = None
                if gt_label is None: # no gt
                    iou = 1
                    mask_iou_25_f1_score = 1
                    mask_iou_50_f1_score = 1
                    bbox_iou_25_f1_score = 1
                    bbox_iou_50_f1_score = 1
                    top_1_bbox_iou = 1
                    top_1_mask_iou = 1
                else:
                    all_gt_bbox = []
                    for gt_mask_id in gt_label:
                        gt_mask = mask['target_full'][0]['masks'][gt_mask_id,:].squeeze().bool()
                        gt_points = mask["original_coordinates"][0][gt_mask]
                        min_vals,max_vals  = gt_points.min(axis=0),gt_points.max(axis=0)
                        gt_bbox = np.vstack((min_vals, max_vals))
                        all_gt_bbox.append(gt_bbox)
                    save_gt_bbox = all_gt_bbox
                    iou = 0
                    mask_iou_25_f1_score = 0
                    mask_iou_50_f1_score = 0
                    bbox_iou_25_f1_score = 0
                    bbox_iou_50_f1_score = 0
                    top_1_bbox_iou = 0
                    top_1_mask_iou = 0
            if "m3dref" in instance['type']:
                ann_id = instance['type'].split(":")[-2]
                m3dref_bbox_result.append({
                    "ann_id": ann_id,
                    "pred_aabb_bound": save_pred_bbox,
                    "gt_aabb_bound": save_gt_bbox,
                    "eval_type":instance['type'].split(":")[1]
                })
            if iou is not None:
                instance["iou"] = float(iou)
                instance["bbox_iou_25_f1_score"] = float(bbox_iou_25_f1_score)
                instance["bbox_iou_50_f1_score"] = float(bbox_iou_50_f1_score)
                instance["mask_iou_25_f1_score"] = float(mask_iou_25_f1_score)
                instance["mask_iou_50_f1_score"] = float(mask_iou_50_f1_score)
                if top_1_bbox_iou is not None:
                    instance["top_1_bbox_iou"] = float(top_1_bbox_iou)
                else:
                    instance["top_1_bbox_iou"] = top_1_bbox_iou
                instance["top_1_mask_iou"] = float(top_1_mask_iou)
                if "m3dref" in instance['type']:
                    score["m3dref:overall"].append(float(iou))
                    bbox_score_25["m3dref:overall"].append(float(bbox_iou_25_f1_score))
                    bbox_score_50["m3dref:overall"].append(float(bbox_iou_50_f1_score))
                    mask_score_25["m3dref:overall"].append(float(mask_iou_25_f1_score))
                    mask_score_50["m3dref:overall"].append(float(mask_iou_50_f1_score))
                elif "scanqa:with_grounding" in instance['type']:
                    score["scanqa:with_grounding"].append(float(iou))
                    bbox_score_25["scanqa:with_grounding"].append(float(bbox_iou_25_f1_score))
                    bbox_score_50["scanqa:with_grounding"].append(float(bbox_iou_50_f1_score))
                    mask_score_25["scanqa:with_grounding"].append(float(mask_iou_25_f1_score))
                    mask_score_50["scanqa:with_grounding"].append(float(mask_iou_50_f1_score))
                elif "detection" in instance['type']:
                    score["detection"].append(float(iou))
                    bbox_score_25["detection"].append(float(bbox_iou_25_f1_score))
                    bbox_score_50["detection"].append(float(bbox_iou_50_f1_score))
                    mask_score_25["detection"].append(float(mask_iou_25_f1_score))
                    mask_score_50["detection"].append(float(mask_iou_50_f1_score))
                elif "scanrefer" in instance['type']:
                    data_type = instance['type'].replace(":with_grounding","")
                    score[data_type].append(float(iou))
                    bbox_score_25[data_type].append(float(bbox_iou_25_f1_score))
                    bbox_score_50[data_type].append(float(bbox_iou_50_f1_score))
                    mask_score_25[data_type].append(float(mask_iou_25_f1_score))
                    mask_score_50[data_type].append(float(mask_iou_50_f1_score))
    return pred, score, bbox_score_25, bbox_score_50, mask_score_25, mask_score_50,m3dref_bbox_result

class StopwordFilter(object):

    def __init__(self, filename):
        self.pats = []
        if os.path.exists(filename):
            for ln in open(filename, 'r').readlines():
                ww = ln.split()
                if len(ww) == 1:
                    self.pats.append((re.compile(r'^' + ww[0] + r'$'), ''))
                elif len(ww) == 2:
                    self.pats.append((re.compile(r'^' + ww[0] + r'$'), ww[1]))

    def _filter(self, input_words):
        output_words = []
        for w in input_words:
            target = w
            for p in self.pats:
                v = p[0].sub(p[1], w)
                if v != w:
                    target = v
                    break
            if target != '':
                output_words.append(target)
        return output_words

    def __call__(self, input_words):
        if isinstance(input_words, str):
            return ' '.join(self._filter(input_words.split()))
        elif isinstance(input_words, list):
            return self._filter(input_words)
        else:
            return None


class Evaluator():
    def __init__(self,directory_path,
                 statistics=True,
                 test_scanrefer=True,
                 test_m3drefer=True,
                 test_language=True,
                 test_detection=True,
                 zt_threshold=0.15) -> None:
        self.directory_path = directory_path
        self.statistics = statistics
        self.test_scanrefer = test_scanrefer
        self.test_m3drefer = test_m3drefer
        self.test_language = test_language
        self.test_detection = test_detection
        self.zt_threshold = zt_threshold

    @staticmethod
    def to_coco(kvs, keys):
        res = defaultdict(list)
        for k in keys:
            if k in kvs:
                caps = kvs[k]
                for c in caps:
                    res[k].append({'caption': c})
            else:
                res[k].append({'caption': ''})
        return res

    def evaluate(self,ground_truths,prediction,verbose = True,iou = None):
        if iou is None:
            iou_25 = np.ones(len(ground_truths))
            iou_50 = np.ones(len(ground_truths))
        else:
            iou = np.array([iou[k] for k in ground_truths.keys()])
            iou_25 = iou>0.25
            print(f"iou 25: {np.sum(iou_25)/len(iou_25)}")
            iou_50 = iou>0.5
            print(f"iou 50: {np.sum(iou_50)/len(iou_50)}")
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE"),
        ]
        tokenizer = PTBTokenizer()
        ref_sent = ground_truths
        hypo_sent = prediction
        final_scores = {}
        ref_coco = tokenizer.tokenize(self.to_coco(ref_sent, ref_sent.keys()))
        hypo_coco = tokenizer.tokenize(self.to_coco(hypo_sent, ref_sent.keys()))
        for scorer, method in scorers:
            if verbose:
                print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(ref_coco, hypo_coco)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores

    @staticmethod
    def clean_answer(data):
        data = data.lower()
        data = re.sub('[ ]+$' ,'', data)
        data = re.sub('^[ ]+' ,'', data)
        data = re.sub(' {2,}', ' ', data)

        data = re.sub('\.[ ]{2,}', '. ', data)
        data = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', data)
        data = re.sub('ç' ,'c', data)
        data = re.sub('’' ,'\'', data)
        data = re.sub(r'\bletf\b' ,'left', data)
        data = re.sub(r'\blet\b' ,'left', data)
        data = re.sub(r'\btehre\b' ,'there', data)
        data = re.sub(r'\brigth\b' ,'right', data)
        data = re.sub(r'\brght\b' ,'right', data)
        data = re.sub(r'\bbehine\b', 'behind', data)
        data = re.sub(r'\btv\b' ,'TV', data)
        data = re.sub(r'\bchai\b' ,'chair', data)
        data = re.sub(r'\bwasing\b' ,'washing', data)
        data = re.sub(r'\bwaslked\b' ,'walked', data)
        data = re.sub(r'\boclock\b' ,'o\'clock', data)
        data = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data)

        # digit to word, only for answer
        data = re.sub(r'\b0\b', 'zero', data)
        data = re.sub(r'\bnone\b', 'zero', data)
        data = re.sub(r'\b1\b', 'one', data)
        data = re.sub(r'\b2\b', 'two', data)
        data = re.sub(r'\b3\b', 'three', data)
        data = re.sub(r'\b4\b', 'four', data)
        data = re.sub(r'\b5\b', 'five', data)
        data = re.sub(r'\b6\b', 'six', data)
        data = re.sub(r'\b7\b', 'seven', data)
        data = re.sub(r'\b8\b', 'eight', data)
        data = re.sub(r'\b9\b', 'nine', data)
        data = re.sub(r'\b10\b', 'ten', data)
        data = re.sub(r'\b11\b', 'eleven', data)
        data = re.sub(r'\b12\b', 'twelve', data)
        data = re.sub(r'\b13\b', 'thirteen', data)
        data = re.sub(r'\b14\b', 'fourteen', data)
        data = re.sub(r'\b15\b', 'fifteen', data)
        data = re.sub(r'\b16\b', 'sixteen', data)
        data = re.sub(r'\b17\b', 'seventeen', data)
        data = re.sub(r'\b18\b', 'eighteen', data)
        data = re.sub(r'\b19\b', 'nineteen', data)
        data = re.sub(r'\b20\b', 'twenty', data)
        data = re.sub(r'\b23\b', 'twenty-three', data)

        # misc
        # no1, mat2, etc
        data = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data)
        data = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data)
        data = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data)
        data = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data)

        data = re.sub(r'\bbackwards\b', 'backward', data)

        return data
    
    def special_token_filter(self,lan,clean = True,truncation = True,max_length = 256):
        """
        Usage:
            clean the language, remove stop words and special tokens
        Args:
            lan: List[str], language to be cleaned
            clean: bool, if apply LEO clean strategy
            truncation: to avoid crash pycocoevalcap
        """
        replacements = {
        "ASSISTANT:": "",
        "ASSISTANT: ": "",
        "\n": "",
        "<s>": "",
        "</s>": "",
        "<unk>": "",
        "<p>": "",
        "</p>": "",
        "<ref>": "",
        "<|endoftext|>": ""  # for GPT2
        }
        for old, new in replacements.items():
            lan = lan.replace(old, new)
        lan = lan.strip()
        lan = re.sub(r'\s{2,}', ' ', lan)
        if truncation:
            if len(lan)>max_length:
                lan = lan[:max_length]
        if clean:
            lan = self.clean_answer(lan)
        return lan
    
    @staticmethod
    def replace_none_with_zero(pred):
        for i,val in enumerate(pred):
                if val is None:
                    pred[i] = 0.0
        return pred

    @staticmethod
    def show_iou_score(pred,threshold,print_result=True):
        pred_th = np.array(pred)>threshold
        score = sum(pred_th)/len(pred_th)
        if print_result:
            print(score)
        return score

    @staticmethod
    def refined_EM(data,gt,set_zero_as_error=True,not_refine=False):
        EM = []
        _data = copy.deepcopy(data)
        if not_refine:
            for ins in _data:
                    pred  = _data[ins][0]
                    if pred in gt[ins]:
                        EM.append(1)
                    else:
                        EM.append(0)
        else:
            for ins in _data:
                to_append = 0
                pred  = _data[ins][0]
                if set_zero_as_error:
                    if pred in [" ",""]:
                        pred = "@@@@@@@@@@@@@@@@@@@@"
                for _gt in gt[ins]:
                    if pred == _gt:
                        to_append = 1
                        continue
                    elif "".join(pred.split()) in "".join(_gt.split()):
                        to_append = 1
                        continue
                    elif "".join(_gt.split()) in "".join(pred.split()):
                        to_append = 1
                        continue
                EM.append(to_append)
        return EM

    @staticmethod
    def print_dict(lan):
        for key in lan:
            print(f"{key}:      {lan[key]}")

    @staticmethod
    def average_two_score(iou50,iou25,print_score=True):
        avg_50,avg_25 = sum(iou50)/len(iou50),sum(iou25)/len(iou25)
        if print_score:
            print(f"iou 50 {avg_50}")
            print(f"iou 25 {avg_25}")
        return avg_50,avg_25

    def evaluate_scanrefer(self,
                           scanrefer_multi_bbox_score,
                           scanrefer_unique_bbox_score,
                           scanrefer_multi_mask_score,
                           scanrefer_unique_mask_score,
                           ):
        """
        Args: 
        scanrefer_multi_bbox_score: List[float]
            list of iou
        scanrefer_unique_bbox_score: List[float]
            list of iou
        """
        scanrefer_multi_bbox_score = self.replace_none_with_zero(scanrefer_multi_bbox_score)
        scanrefer_unique_bbox_score = self.replace_none_with_zero(scanrefer_unique_bbox_score)

        print(" ======================== scanrefer =========================")
        print(" ==== scanrefer unique bbox top1 0.5 ====")
        self.show_iou_score(scanrefer_unique_bbox_score,0.5)
        print(" ==== scanrefer unique bbox top1 0.25 ====")
        self.show_iou_score(scanrefer_unique_bbox_score,0.25)
        print(" ==== scanrefer multi bbox top1 0.5 ====")
        self.show_iou_score(scanrefer_multi_bbox_score,0.5)
        print(" ==== scanrefer multi bbox top1 0.25 ====")
        self.show_iou_score(scanrefer_multi_bbox_score,0.25)
        print(" ==== scanrefer overall bbox top1 0.5 ====")
        self.show_iou_score(scanrefer_multi_bbox_score+scanrefer_unique_bbox_score,0.5)
        print(" ==== scanrefer overall bbox top1 0.25 ====")
        self.show_iou_score(scanrefer_multi_bbox_score+scanrefer_unique_bbox_score,0.25)

        print(" ==== scanrefer unique mask top1 0.5 ====")
        self.show_iou_score(scanrefer_unique_mask_score,0.5)
        print(" ==== scanrefer unique mask top1 0.25 ====")
        self.show_iou_score(scanrefer_unique_mask_score,0.25)
        print(" ==== scanrefer multi mask top1 0.5 ====")
        self.show_iou_score(scanrefer_multi_mask_score,0.5)
        print(" ==== scanrefer multi mask top1 0.25 ====")
        self.show_iou_score(scanrefer_multi_mask_score,0.25)
        print(" ==== scanrefer overall mask top1 0.5 ====")
        self.show_iou_score(scanrefer_multi_mask_score+scanrefer_unique_mask_score,0.5)
        print(" ==== scanrefer overall mask top1 0.25 ====")
        self.show_iou_score(scanrefer_multi_mask_score+scanrefer_unique_mask_score,0.25)
        print(" =============================================================")

    def evaluate_m3drefer(self,
                          m3dref_st_wo_d_f1_50,
                          m3dref_st_wo_d_f1_25,
                          m3dref_st_w_d_f1_50,
                          m3dref_st_w_d_f1_25,
                          m3dref_mt_f1_50,
                          m3dref_mt_f1_25,
                          m3dref_zt_wo_d_f1_50,
                          m3dref_zt_wo_d_f1_25,
                          m3dref_zt_w_d_f1_50,
                          m3dref_zt_w_d_f1_25,
                          zt_threshold=None,
                          m3dref_zt_wo_d_f1_50_threshold=None,
                          m3dref_zt_wo_d_f1_25_threshold=None,
                          m3dref_zt_w_d_f1_50_threshold=None,
                          m3dref_zt_w_d_f1_25_threshold=None,
                        ):
        print(" ========================= m3drefer ==========================")
        print(f" each type amount: st_wo_d {len(m3dref_st_wo_d_f1_50)} st_w_d {len(m3dref_st_w_d_f1_50)} mt {len(m3dref_mt_f1_50)} zt_wo_d {len(m3dref_zt_wo_d_f1_50)} zt_w_d {len(m3dref_zt_w_d_f1_50)}")
        len_st_w = len(m3dref_st_w_d_f1_50)
        if len_st_w<5328:
            m3dref_st_w_d_f1_25+=[0]*(5328-len_st_w)
            m3dref_st_w_d_f1_50+=[0]*(5328-len_st_w)
        len_st_wo = len(m3dref_st_wo_d_f1_50)
        if len_st_wo<2099:
            m3dref_st_wo_d_f1_25+=[0]*(2099 - len_st_wo)
            m3dref_st_wo_d_f1_50+=[0]*(2099 - len_st_wo)
        len_zt_w = len(m3dref_zt_w_d_f1_50)
        if len_zt_w<378:
            m3dref_zt_w_d_f1_25+=[0]*(378 - len_zt_w)
            m3dref_zt_w_d_f1_50+=[0]*(378 - len_zt_w)
        len_zt_wo_d = len(m3dref_zt_wo_d_f1_25)
        if len_zt_wo_d<528:
            m3dref_zt_wo_d_f1_25+=[0]*(528 - len_zt_wo_d)
            m3dref_zt_wo_d_f1_50+=[0]*(528 - len_zt_wo_d)
        len_mt = len(m3dref_mt_f1_50)
        if len_mt<2757:
            m3dref_mt_f1_25+=[0]*(2757 - len_mt)
            m3dref_mt_f1_50+=[0]*(2757 - len_mt)
        
        print( "==== m3dref st_wo_d_50/25 ====")
        self.average_two_score(m3dref_st_wo_d_f1_50,m3dref_st_wo_d_f1_25)
        print( "==== m3dref st_w_d_50/25 ====")
        self.average_two_score(m3dref_st_w_d_f1_50,m3dref_st_w_d_f1_25)
        print( "==== m3dref mt_50/25 ====")
        self.average_two_score(m3dref_mt_f1_50,m3dref_mt_f1_25)
            
        print( "==== m3dref zt_wo_d_50/25 ====")
        self.average_two_score(m3dref_zt_wo_d_f1_50,m3dref_zt_wo_d_f1_25)
        if zt_threshold is not None:
            print(f" threshold {zt_threshold}")
            self.average_two_score(m3dref_zt_wo_d_f1_50_threshold,m3dref_zt_wo_d_f1_25_threshold)
        print( "==== m3dref zt_w_d_50/25 ====")
        self.average_two_score(m3dref_zt_w_d_f1_50,m3dref_zt_w_d_f1_25)
        if zt_threshold is not None:
            print(f" threshold {zt_threshold}")
            self.average_two_score(m3dref_zt_w_d_f1_50_threshold,m3dref_zt_w_d_f1_25_threshold)
        print( "==== m3dref all ====")
        self.average_two_score(m3dref_st_wo_d_f1_50+m3dref_st_w_d_f1_50+m3dref_mt_f1_50+m3dref_zt_wo_d_f1_50+m3dref_zt_w_d_f1_50,
                        m3dref_st_wo_d_f1_25+m3dref_st_w_d_f1_25+m3dref_mt_f1_25+m3dref_zt_wo_d_f1_25+m3dref_zt_w_d_f1_25)
        print(" =============================================================")

    def evaluate_detection(self):
        all_detection_files = glob.glob(f"{self.directory_path}/*.npz")
        all_det_result = {}
        for path in all_detection_files:
            data = np.load(path)
            npz_name = path.split("/")[-1]
            scene_name = npz_name.split(".")[0]
            all_det_result.update({scene_name:data})
        mask3d_det_evaluation(preds=all_det_result,
                              gt_path="./data/processed/scannet200/instance_gt/validation",
                              output_file=f"{self.directory_path}/detection_result.csv",
                              dataset="scannet200")


    def load_data_and_eval(self):
        if test_m3drefer:
            try:
                print(" ================= m3drefer official evaluator ================= ")
                lang_input_path = "data/multi3drefer/multi3drefer_val.json"
                pred_data, gt_data = parse_prediction(f"{self.directory_path}/m3drefer")
                evaluator = Multi3DReferEvaluator("m3dv",verbose=True)
                evaluator.set_ground_truths(gt_data)
                _ = evaluator.evaluate(pred_data)
                print(" =============================================================== ")
            except Exception as e:
                print("no folder storing bbox info, skip m3drefer official evaluator")
                print(e)

        combined_data = []
        combined_score = []
        combined_score_bbox_score_25 = []
        combined_score_bbox_score_50 = []
        combined_score_bbox_top1 = []
        scanrefer_multi_bbox_score = []
        scanrefer_unique_bbox_score = []
        scanrefer_multi_mask_score = []
        scanrefer_unique_mask_score = []

        m3dref_st_wo_d_f1_50 = []
        m3dref_st_wo_d_f1_25 = []

        m3dref_st_w_d_f1_50 = []
        m3dref_st_w_d_f1_25 = []

        m3dref_mt_f1_50=[]
        m3dref_mt_f1_25=[]

        m3dref_zt_wo_d_f1_50 = []
        m3dref_zt_wo_d_f1_25 = []

        m3dref_zt_w_d_f1_50 = []
        m3dref_zt_w_d_f1_25 = []

        m3dref_zt_wo_d_f1_50_threshold = []
        m3dref_zt_wo_d_f1_25_threshold = []

        m3dref_zt_w_d_f1_50_threshold = []
        m3dref_zt_w_d_f1_25_threshold = []
        zt_threshold = self.zt_threshold
        if self.test_detection:
            self.evaluate_detection()

        for filename in tqdm(os.listdir(self.directory_path)):
            if filename.endswith('.json'):
                file_path = os.path.join(self.directory_path, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    for line in data["prediction"]:
                        line["scene_id"] = (file_path.split("_"))[-2]
                        if "scanrefer:multiple" in line["type"]:
                            scanrefer_multi_bbox_score.append(line["top_1_bbox_iou"])
                            scanrefer_multi_mask_score.append(line["iou"])

                        elif "scanrefer:unique" in line["type"]:
                            scanrefer_unique_bbox_score.append(line["top_1_bbox_iou"])
                            scanrefer_unique_mask_score.append(line["iou"])

                        elif "m3dref:st_wo_d" in line["type"]:
                            m3dref_st_wo_d_f1_25.append(line["bbox_iou_25_f1_score"])
                            m3dref_st_wo_d_f1_50.append(line["bbox_iou_50_f1_score"])
                            
                        elif "m3dref:st_w_d" in line["type"]:
                            m3dref_st_w_d_f1_25.append(line["bbox_iou_25_f1_score"])
                            m3dref_st_w_d_f1_50.append(line["bbox_iou_50_f1_score"])

                        elif "m3dref:zt_wo_d" in line["type"]:
                            m3dref_zt_wo_d_f1_25.append(line["bbox_iou_25_f1_score"])
                            m3dref_zt_wo_d_f1_50.append(line["bbox_iou_50_f1_score"])
                            if line['score'] and line["score"][0][0]<zt_threshold:
                                m3dref_zt_wo_d_f1_25_threshold.append(1)
                                m3dref_zt_wo_d_f1_50_threshold.append(1)
                            else:
                                m3dref_zt_wo_d_f1_25_threshold.append(line["bbox_iou_25_f1_score"])
                                m3dref_zt_wo_d_f1_50_threshold.append(line["bbox_iou_50_f1_score"])

                        elif "m3dref:zt_w_d" in line["type"]:
                            m3dref_zt_w_d_f1_25.append(line["bbox_iou_25_f1_score"])
                            m3dref_zt_w_d_f1_50.append(line["bbox_iou_50_f1_score"])
                            if line['score'] and line["score"][0][0]<zt_threshold:
                                m3dref_zt_w_d_f1_25_threshold.append(1)
                                m3dref_zt_w_d_f1_50_threshold.append(1)
                            else:
                                m3dref_zt_w_d_f1_25_threshold.append(line["bbox_iou_25_f1_score"])
                                m3dref_zt_w_d_f1_50_threshold.append(line["bbox_iou_50_f1_score"])

                        elif "m3dref:mt" in line["type"]:
                            m3dref_mt_f1_25.append(line["bbox_iou_25_f1_score"])
                            m3dref_mt_f1_50.append(line["bbox_iou_50_f1_score"])

                    combined_data+=data["prediction"]
                    combined_score+=[data["score"]]
                    combined_score_bbox_score_25 += [data["bbox_score_25"]]
                    combined_score_bbox_score_50 += [data["bbox_score_50"]]

        if self.test_scanrefer:
            self.evaluate_scanrefer(scanrefer_multi_bbox_score,
                                scanrefer_unique_bbox_score,
                                scanrefer_multi_mask_score,
                                scanrefer_unique_mask_score,
                                )

        if self.test_m3drefer:
            self.evaluate_m3drefer(m3dref_st_wo_d_f1_50,
                                m3dref_st_wo_d_f1_25,
                                m3dref_st_w_d_f1_50,
                                m3dref_st_w_d_f1_25,
                                m3dref_mt_f1_50,
                                m3dref_mt_f1_25,
                                m3dref_zt_wo_d_f1_50,
                                m3dref_zt_wo_d_f1_25,
                                m3dref_zt_w_d_f1_50,
                                m3dref_zt_w_d_f1_25,
                                zt_threshold,
                                m3dref_zt_wo_d_f1_50_threshold,
                                m3dref_zt_wo_d_f1_25_threshold,
                                m3dref_zt_w_d_f1_50_threshold,
                                m3dref_zt_w_d_f1_25_threshold,
                            )

        data = combined_data
        scanqa_lan = {}
        scanqa_gt = {}

        scan2cap_lan = {}
        scan2cap_gt = {}
        scan2cap_iou = {}

        objdesc_lan = {}
        objdesc_gt = {}

        dialog_lan = {}
        dialog_gt = {}

        planning_lan = {}
        planning_gt = {}

        if test_language:
            for idxx ,line in enumerate(data):
                if "text_only" in line["type"] and "scanqa" in line["type"]:
                    idx = line["type"].split(":")[1]
                    if idx in scanqa_gt:
                        scanqa_gt[idx].append(self.special_token_filter(line["gt"]))
                        if len(scanqa_lan[idx][0])<3 or len(scanqa_lan[idx][0])>20:
                            scanqa_lan[idx]= [self.special_token_filter(line["output_language"])]
                    else: # follow LEO, we clean answers and predictions
                        scanqa_lan[idx]= [self.special_token_filter(line["output_language"])]
                        scanqa_gt[idx] = [self.special_token_filter(line["gt"])]
                # following scan2cap evaluation settings, we manually add "bos" and "eos" to the start and end of both gt and pred sentences
                elif "scan2cap" in line["type"]:
                    try:
                        t = line["gt_inst_ids"]["inst_ids_question"][0][0]
                    except: #old eval
                        t = eval(line["gt_inst_ids"])[1][1][0][0]
                    idxx = f"{line['scene_id']}_{t}"
                    if idxx not in scan2cap_lan:
                        scan2cap_lan[idxx]= ["sos "+ self.special_token_filter(line["output_language"],clean=False)+ " eos"]
                    if idxx not in scan2cap_gt:
                        scan2cap_gt[idxx] = ["sos "+self.special_token_filter(line["gt"],clean=False)+" eos"]
                    else:
                        scan2cap_gt[idxx].append("sos "+self.special_token_filter(line["gt"],clean=False)+" eos")
                    if idxx not in scan2cap_iou:
                        scan2cap_iou[idxx] = line["gt_predicted_iou"]
                    if line["gt_predicted_iou"] > scan2cap_iou[idxx]:
                        scan2cap_iou[idxx] = line["gt_predicted_iou"]
                        scan2cap_lan[idxx]= ["sos "+ self.special_token_filter(line["output_language"],clean=False)+" eos"]
                elif "objdesc" in line["type"]:
                    objdesc_lan[idxx]= [self.special_token_filter(line["output_language"],clean=False)]
                    objdesc_gt[idxx] = [self.special_token_filter(line["gt"],clean=False)]
                elif "dialog" in line["type"]:
                    dialog_lan[idxx]= [self.special_token_filter(line["output_language"],clean=False)]
                    dialog_gt[idxx] = [self.special_token_filter(line["gt"],clean=False)]
                elif "plan" in line["type"]:
                    planning_lan[idxx]= [self.special_token_filter(line["output_language"],clean=False)]
                    planning_gt[idxx] = [self.special_token_filter(line["gt"],clean=False)]

        if self.statistics:
            print(" ============================= statistics ==============================")

            total_grounding_detect = 0
            total_grounding_correct = 0
            total_grounding_truth = 0

            average_score_bbox_score_25 = deepcopy(template)
            average_score_bbox_score_50 = deepcopy(template)

            for score in combined_score_bbox_score_25:
                for key in average_score_bbox_score_25:
                    average_score_bbox_score_25[key] = average_score_bbox_score_25[key]+score[key]
                    if key == "scanrefer:overall":
                        average_score_bbox_score_25[key] = average_score_bbox_score_25[key]+score["scanrefer:unique"]+score["scanrefer:multiple"]
            
            for score in combined_score_bbox_score_50:
                for key in average_score_bbox_score_50:
                    average_score_bbox_score_50[key] = average_score_bbox_score_50[key]+score[key]
                    if key == "scanrefer:overall":
                        average_score_bbox_score_50[key] = average_score_bbox_score_50[key]+score["scanrefer:unique"]+score["scanrefer:multiple"]

            print(" ==== bbox f1 0.25 ====")
            for key in average_score_bbox_score_25:
                if len(average_score_bbox_score_25[key]):
                    print(f"length of query:{len(average_score_bbox_score_25[key])}")
                    average_score_bbox_score_25[key] = sum(average_score_bbox_score_25[key])/len(average_score_bbox_score_25[key])
                    print(f"{key}: {average_score_bbox_score_25[key]}")

            print(" ==== bbox f1 0.50 ====")
            for key in average_score_bbox_score_50:
                if len(average_score_bbox_score_50[key]):
                    print(f"length of query:{len(average_score_bbox_score_50[key])}")
                    average_score_bbox_score_50[key] = sum(average_score_bbox_score_50[key])/len(average_score_bbox_score_50[key])
                    print(f"{key}: {average_score_bbox_score_50[key]}")

            tp = 0
            total_det = 0
            for item in data:
                if "det" in item["type"]:
                    total_det+=1
                    if "no" not in item["gt"].lower() and "no" not in item["output_language"].lower():
                        total_grounding_truth+=1
                        total_grounding_correct+=1
                        total_grounding_detect+=1
                        tp+=1
                    elif "no" not in item["gt"].lower() and "no"in item["output_language"].lower():
                        total_grounding_truth+=1
                    elif "no" in item["gt"].lower() and "no" not in item["output_language"].lower():
                        total_grounding_detect+=1
                    else:
                        tp+=1

            precision_if_grounding = total_grounding_correct/total_grounding_detect if total_grounding_detect > 0 else 0
            recall_if_grounding = total_grounding_correct/total_grounding_truth if total_grounding_truth > 0 else 0

            print(f"precision if grounding: {precision_if_grounding}, recall if grounding: {recall_if_grounding}")
            print(f"gt probability of det {total_grounding_truth / total_det}")
            print(f"pred probability of det {total_grounding_detect / total_det}")
            print(f"f1 {tp/total_det}")
            print(" =========================================================== ")

        if self.test_language:
            print(" ============================================================= ")
            print(" ========================= scan 2 cap ======================== ")
            print(f"scan2cap test length: {len(scan2cap_lan)}")
            if len(scan2cap_gt):
                scan2cap_25 = deepcopy(scan2cap_lan)
                scan2cap_50 = deepcopy(scan2cap_lan)
                for key in scan2cap_iou:
                    if scan2cap_iou[key]<0.25:
                        scan2cap_25[key] = ["sos eos"]
                    if scan2cap_iou[key]<0.50:
                        scan2cap_50[key] = ["sos eos"]

                try:
                    print(" ================= scan2cap 0.25 ====================== ")
                    final_scores = self.evaluate(scan2cap_gt,scan2cap_25,verbose=False)
                    self.print_dict(final_scores)
                    print(" ================= scan2cap 0.50 ====================== ")
                    final_scores = self.evaluate(scan2cap_gt,scan2cap_50,verbose=False)
                    self.print_dict(final_scores)
                except Exception as e:
                    print('Error in evaluating Scan2Cap.')
            else:
                print(" can not find data to evaluate, skip scan 2 cap")

            print(" ========================= scan qa ===========================")
            print(f"scanqa val length: {len(scanqa_gt)}")
            if len(scanqa_gt):
                final_scores = self.evaluate(scanqa_gt, scanqa_lan,verbose=False)
                self.print_dict(final_scores)
                EM = self.refined_EM(scanqa_lan,scanqa_gt,not_refine=True)
                print(f"EM:         { sum(EM)/len(EM)}")
                EM_type = self.refined_EM(scanqa_lan,scanqa_gt,False)
                print(f"refined EM: { sum(EM_type)/len(EM_type)}")
            else:
                print(" can not find data to evaluate, skip scan qa")

            print(" ============================================================= ")
            print(" ======================= obj description ===================== ")
            print(f"objdesc test length: {len(objdesc_lan)}")
            if len(objdesc_gt):
                final_scores = self.evaluate(objdesc_gt,objdesc_lan,verbose=False)
                self.print_dict(final_scores)
            else:
                print(" can not find data to evaluate, skip obj description")

            print(" ======================= dialog description ===================== ")
            print(f"dialog test length: {len(dialog_lan)}")
            if len(dialog_gt):
                final_scores = self.evaluate(dialog_gt,dialog_lan,verbose=False)
                self.print_dict(final_scores)
            else:
                print(" can not find data to evaluate, skip dialogue")
            
            print(" ======================= planning description ===================== ")
            print(f"planning test length: {len(planning_lan)}")
            if len(planning_gt):
                final_scores = self.evaluate(planning_gt,planning_lan,verbose=False)
                self.print_dict(final_scores)
            else:
                print(" can not find data to evaluate, skip planning")

        print(" ======================== END OF TEST ==========================")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--directory_path', type=str, help='path to json files')
    parser.add_argument('--statistics', type=lambda x: (str(x).lower() == 'true'), help='detection')
    parser.add_argument('--test_scanrefer', type=lambda x: (str(x).lower() == 'true'), help='evaluate scanrefer')
    parser.add_argument('--test_m3drefer', type=lambda x: (str(x).lower() == 'true'), help='evaluate m3drefer')
    parser.add_argument('--test_language', type=lambda x: (str(x).lower() == 'true'), help='evaluate language')
    parser.add_argument('--test_detection', type=lambda x: (str(x).lower() == 'true'), help='evaluate detection')
    args = parser.parse_args()
    directory_path = args.directory_path
    statistics = args.statistics
    test_scanrefer = args.test_scanrefer
    test_m3drefer = args.test_m3drefer
    test_language = args.test_language
    test_detection = args.test_detection

    print(f"test directory: {directory_path}")
    print("test_m3drefer",test_m3drefer)
    print("test_scanrefer",test_scanrefer)
    print("statistics",statistics)
    eval = Evaluator(
        statistics=statistics,
        test_language=test_language,
        test_m3drefer=test_m3drefer,
        test_scanrefer=test_scanrefer,
        test_detection=test_detection,
        directory_path=directory_path
    )
    eval.load_data_and_eval()

        