#RegionProposalNetwork 
class RegionProposalNetwork(nn.Module):
  def __init__(self,in_channel = 512):
    super(RegionProposalNetwork,self).__init__()
    self.scales = [128,256,512]
    self.aspect_ratios = [0.5,1,2]
    self.num_anchors = len(self.scales) * len(self.aspect_ratios)

    #3*3 conv 
    self.rpn_conv = nn.Conv2d(in_channel,in_channel,kernel_size=3,stride=1,padding=1)
    #1*1 conv 
    self.cls_layer = nn.Conv2d(in_channel,self.num_anchors,kernel_size=1,stride=1)
    #1*1 regression
    self.bbox_reg_layer = nn.Conv2d(in_channel,self.num_anchors*4,kernel_size=1,stride=1)

  def assign_targets_to_anchors(self,anchors,gt_boxes):
    iou_matrix = get_iou(gt_boxes,anchors)
    best_match_iou ,best_match_gt_index = iou_matrix.max(dim=0)
    best_match_gt_idx_pre_threshold = best_match_gt_index.clone()
    below_low_threshold = best_match_iou < 0.3 
    between_threshold = (best_match_iou>=0.3) &(best_match_iou<0.7)
    best_match_gt_index[below_low_threshold] = -1
    best_match_gt_index[between_threshold]=-2

    #low quality anchor boxes 
    best_anchor_iou_for_gt,_ = iou_matrix.max(dim=1)
    gt_pred_pair_with_highest_iou = torch.where(iou_matrix== best_anchor_iou_for_gt)

    pred_inds_to_update = gt_pred_pair_with_highest_iou[1]
    best_match_gt_index[pred_inds_to_update] = best_match_gt_idx_pre_threshold[pred_inds_to_update]

    matched_gt_boxes = gt_boxes[best_match_gt_index.clamp(min=0)]

    #set alll forgrounf anchor labels as 1 
    labels = best_match_gt_index >=0 

    labels = labels.to(dtype=torch.float32)

    #set all background labels as  0 
    background_anchors = best_match_gt_index == -1 
    labels[background_anchors] = 0.0 
    ignored_anchors = best_match_gt_index == -2 
    labels[ignored_anchors] = -1.0
    return labels,matched_gt_boxes

  def filter_proposals(self,proposals,cls_scores,image_shape):
    cls_scores = cls_scores.reshape(-1)
    cls_scores = torch.sigmoid(cls_scores)
    _,top_n_idx = cls_scores.topk(10000)
    proposals = proposals[top_n_idx]
    cls_scores = cls_scores[top_n_idx]
    proposals = clamp_boxes_to_image_boundary(proposals,image_shape)
    #NMS Based on objectness 
    keep_mask = torch.zeros_like(cls_scores,dtype = torch.bool)
    keep_indices = torch.ops.torchvision.nms(proposals,cls_scores,0.7)

    post_nms_keep_indices = keep_indices[
        cls_scores[keep_indices].sort(descending=True)
    ]
    proposals = proposals[post_nms_keep_indices[:2000]]
    cls_scores = cls_scores[post_nms_keep_indices[:2000]]
    return proposals,cls_scores


  def generate_anchors(self,image,feat):
    grid_h,grid_w = feat.shape[-2:]
    image_h,image_w = image.shape[-2:]
    stride_h = torch.tensor(image_h//grid_h,dtype=torch.int64,device = feat.device)
    stride_w = torch.tensor(image_w//grid_w,dtype = torch.int64,device = feat.device)
    scales = torch.as_tensor(self.scales,dtype=feat.dtype,device= feat.device)
    aspect_ratios = torch.as_tensor(self.aspect_ratios,dtype=feat.dtype,device=feat.device)
    h_ratios = torch.sqrt(aspect_ratios)
    w_ratios = 1/h_ratios 
    ws = (w_ratios[:,None]* scales[None,:]).view(-1)
    hs = (h_ratios[:,None]* scales[None,:]).view(-1)

    base_anchors = torch.stack([-ws,-hs,ws,hs],dim=1)/2
    base_anchors = base_anchors.round()
    shifts_x = torch.arange(0,grid_w,dtype = torch.int32,device=feat.device)*stride_w
    shifts_y = torch.arange(0,grid_h,dtype = torch.int32,device=feat.device)*stride_h
    shifts_x,shifts_y = torch.meshgrid(shifts_x,shifts_y,indexing='ij')
    #(H-feat,W_feat)
    shifts_x = shifts_x.reshape(-1)
    shifts_y = shifts_y.reshape(-1)
    shifts = torch.stack((shifts_x,shifts_y,shifts_x,shifts_y),dim=1)

    #shift -> (H-feat*W_feat,4)
    #base anchor -> num_anchor_per_location,4 
    #shift -> (H_feat*W_feat,4)
    anchors = (shifts.view(-1,1,4) + base_anchors.view(1,-1,4))
    anchors = anchors.reshape(-1,4)
    #anchors => H_feat*W-feat,num_anchor_per_location ,4
    return anchors


  def forward(self,image,feat,target):
    #call RPN Layers 
    rpn_feat = nn.ReLU()(self.rpn_conv(feat))
    cls_scores = self.cls_layer(rpn_feat)
    box_transform_pred = self.bbox_reg_layer(rpn_feat)
    #generate anchors 
    anchors = self.generate_anchors(image,feat)
    number_of_anchors_per_location = cls_scores.size(1)

    cls_scores = cls_scores.permute(0,2,3,1)
    cls_scores = cls_scores.reshape(-1,1)

    box_transform_pred = box_transform_pred.view(
        box_transform_pred.size(0),
        number_of_anchors_per_location,
        4,
        rpn_feat.shape[-2],
        rpn_feat.shape[-1]
    )
    box_transform_pred = box_transform_pred.permute(0,3,4,1,2)
    box_transform_pred = box_transform_pred.reshape(-1,4)
    #box_transform_pred - > (B*H_feat*W_feat+Number of anchors per location,H_feat,W_fear)

    proposals = apply_regression_pred_to_anchors_or_proposals(
        box_transform_pred.detach().reshape(-1,1,4),anchors
    )
    proposals = proposals.reshape(proposals.size(0),4)

    proposals,scores = self.filter_proposals(proposals,cls_scores.detach(),image.shape)

    rpn_output = {
        'proposals':proposals,'scores':scores
    }
    if not self.training or target is None:
      return rpn_output 
    else :
      labels_for_anchors,matched_gt_boxes_for_Anchors = self.assign_targets_to_anchors(
          anchors,target['bboxes'][0]
      )

      regression_targets = boxes_to_transformation_targets(
          matched_gt_boxes_for_Anchors,anchors
      )

      sampled_neg_idx_mask,sampled_pos_idx_mask = sample_positive_negative(
          labels_for_anchors,positive_count=128,total_count=256
      )
      sampled_idx = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]

      localization_loss = (
          torch.nn.functional.smooth_l1_loss(
              box_transform_pred[sampled_pos_idx_mask],
              regression_targets[sampled_pos_idx_mask],
              beta = 1/9,
              reduction = 'sum'
          )/(sampled_idx.numel())
      )

      cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
          cls_scores[sampled_idx].flatten(),
          labels_for_anchors[sampled_idx].flatten()
      )

      rpn_output['rpn_classification_loss'] = cls_loss 
      rpn_output['rpn_classification_loss'] = localization_loss
      return rpn_output

