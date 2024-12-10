
import torch 
import torchvision
import torch.nn as nn
import math
from  torch.utils.data import DataLoader 
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection.rpn import AnchorGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_iou(boxes1,boxes2):
  area1 = (boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1])
  area2 = (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1])

  x_left = torch.max(boxes1[:,None,0],boxes2[:,0])
  y_top = torch.max(boxes1[:,None,1],boxes2[:,1])

  x_right = torch.min(boxes1[:,None,2],boxes2[:,2])
  y_bottom = torch.min(boxes1[:,None,3],boxes2[:,3])

  intersection_area = (x_right - x_left).clamp(min=0)*(y_bottom-y_top).clamp(min=0)
  union_area = area1[:,None] + area2 - intersection_area
  return intersection_area/union_area

def apply_regression_pred_to_anchors_or_proposals(box_transforma_pred,anchors_or_proposals):
  box_transform_pred = box_transform_pred.reshape(
      box_transforma_pred.size(0),-1,4
  )
  # get xs,cy,w,h, from x1,y1,x2,y2 
  w = anchors_or_proposals[:,2] - anchors_or_proposals[:,0]
  h = anchors_or_proposals[:,3] - anchors_or_proposals[:,1]
  center_x = anchors_or_proposals[:,0] + 0.5*w
  center_y = anchors_or_proposals[:,1] + 0.5*h
  #apply prediciotns 

  dx = box_transform_pred[...,0]
  dy = box_transform_pred[...,1]
  dw = box_transform_pred[...,2]
  dh = box_transform_pred[...,3]

  pred_center_x = dx*w[:,None] + center_x[:,None]
  pred_center_y = dy*h[:,None] + center_y[:,None]
  pred_w = torch.exp(dw)*w[:,None]
  pred_h = torch.exp(dh)*h[:,None]

  pred_box_x1 = pred_center_x - 0.5* pred_w 
  pred_box_y1 = pred_center_y - 0.5*pred_h
  pred_box_x2 = pred_center_x + 0.5*pred_w
  pred_box_y2 = pred_center_y + 0.5*pred_h

  pred_boxes = torch.stack((pred_box_x1,pred_box_y1,pred_box_x2,pred_box_y2),dim=2)
  return pred_boxes


def clamp_boxes_to_image_boundary(boxes,image_shape):
  boxes_x1 = boxes[...,0]
  boxes_y1 = boxes[...,1]
  boxes_x2 = boxes[...,2]
  boxes_y2 = boxes[...,3]
  height,width = image_shape[-2:]
  boxes_x1 = torch.clamp(min=0,max=width)
  boxes_y1 = torch.clamp(min=0,max=height)
  boxes_x2 = torch.clamp(min=0,max=width)
  boxes_y2 = torch.clamp(min=0,max=height)
  
  boxes = torch.cat((
      boxes_x1[...,None],
      boxes_y1[...,None],
      boxes_x2[...,None],
      boxes_y2[...,None]
  ),dim=-1)
  return boxes


def boxes_to_transformation_targets(ground_truth_boxes,anchors_or_proposals):
  widths = anchors_or_proposals[:,2] - anchors_or_proposals[:,0]
  heights = anchors_or_proposals[:,3] - anchors_or_proposals[:,1]
  center_x = anchors_or_proposals[:,0] + 0.5*widths
  center_y = anchors_or_proposals[:,1] + 0.5 *heights

  #get center_x,center_y,w,h,from x1,y1,x2,y2 

  gt_widths = ground_truth_boxes[:,2]-ground_truth_boxes[:,0]
  gt_heights = ground_truth_boxes[:,3]-ground_truth_boxes[:,1]
  gt_center_x = ground_truth_boxes[:,0] + 0.5*gt_widths
  gt_center_y = ground_truth_boxes[:,1] + 0.5*gt_heights

  target_dx = (gt_center_x - center_x)/widths
  target_dy = (gt_center_y - center_y)/heights
  target_dw = torch.log(gt_widths/widths)
  target_dh = torch.log(gt_heights/heights)

  regression_targets = torch.stack((target_dx,target_dy,target_dw,target_dh),dim=1)
  return regression_targets


def sample_positive_negative(labels,positive_count,total_count):
  positive = torch.where(labels>=1)[0]
  negative = torch.where(labels==0)[0]
  num_pos = positive_count 
  num_pos = min(positive.numel(),num_pos)
  num_neg = total_count - num_pos
  num_neg = min(negative.numel(),num_neg)
  perm_positive_idx = torch.randperm(positive.numel(),device=positive.device)[:num_pos]
  perm_negative_idx = torch.randperm(negative.numel(),device=negative.device)[:num_neg]
  pos_idxs = positive[perm_positive_idx]
  neg_idxs = negative[perm_negative_idx]
  sampled_pos_idx_mask = torch.zeros_like(labels,dtype=torch.bool)
  sampled_neg_idx_mask =torch.zeros_like(labels,dtype=torch.bool)

  sampled_pos_idx_mask[pos_idxs] = True
  sampled_neg_idx_mask[neg_idxs] = True
  return sampled_pos_idx_mask,sampled_neg_idx_mask


def transform_boxes_to_original_size(boxes,new_size,original_size):
    ratios = [
        torch.tensor(s_orig,device=boxes.device,dtype= torch.float32)/
        torch.tensor(s,dtype=torch.float32,device=boxes.device)
        for s_orig,s in zip(original_size,new_size)
    ]
    ratio_height,ratio_width = ratios
    xmin,ymin,xmax,ymax = boxes.unbind(1)
    xmin = xmin/ratio_width
    xmax = xmax/ratio_width
    ymin = ymin/ratio_height
    ymax = ymax/ratio_height
    return torch.stack((xmin,ymin,xmax,ymax),dim=1)
