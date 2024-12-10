class ROIHead(nn.Module):
  def __init__(self,num_classes = 21,in_channels=512):
    super(ROIHead,self).__init__()
    self.num_classes = num_classes
    self.in_channels = in_channels
    self.pool_size = 7 
    self.fc_inner_dim = 1024
    self.fc6 = nn.Linear(in_channels*self.pool_size*self.pool_size,self.fc_inner_dim)
    self.fc7 = nn.Linear(self.fc_inner_dim,self.fc_inner_dim)
    self.cls_score = nn.Linear(self.fc_inner_dim,self.num_classes)
    self.bbox_pred = nn.Linear(self.fc_inner_dim,self.num_classes*4)

  def assign_target_to_proposals(self,proposals,gt_boxes,gt_labels):
    iou_matrix = get_iou(gt_boxes,proposals)
    best_match_iou ,best_match_gt_idx = iou_matrix.max(dim=0)
    below_low_threshold = best_match_iou < 0.5 
    best_match_gt_idx[below_low_threshold] = -1
    matched_gt_boxes_for_proposals = gt_boxes[best_match_gt_idx.clamp(min=0)]

    labels = gt_labels[best_match_gt_idx.clamp(min=0)]
    labels = labels.to(dtype=torch.int64)

    background_proposals = best_match_gt_idx == -1
    labels[background_proposals] = 0
    return labels,matched_gt_boxes_for_proposals

  def fiter_predictions(self,pred_boxes,pred_labels,pred_scores):
    keep = torch.where(pred_scores > 0.05)[0]
    pred_boxes,pred_scores,pred_labels = pred_boxes[keep],pred_scores[keep],pred_labels[keep]

    min_size =1 
    ws,hs = pred_boxes[:,2] - pred_boxes[:,0],pred_boxes[:,3] - pred_boxes[:,1]
    keep = torch.where((ws >= min_size) & (hs >= min_size))[0]
    pred_boxes ,pred_scores ,pred_labels = pred_boxes[keep],pred_scores[keep],pred_labels[keep]

    #class wise nms 

    keep_mask = torch.zeros_like(pred_scores,dtype=torch.bool)
    for class_id in torch.unique(pred_labels):
      curr_indices = torch.where(pred_labels == class_id)[0]
      curr_keep_indices = torch.ops.torchvision.nms(
          pred_boxes[curr_indices],
          pred_scores[curr_indices],
          iou_threshold=0.5
      )
      keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = torch.where(keep_mask)[0]
    post_nms_keep_indices = keep_indices[pred_labels[keep_indices].sort(
        descending=True
    )[1]]
    keep = post_nms_keep_indices[:100]
    pred_boxes,pred_scores,pred_labels = pred_boxes[keep],pred_scores[keep],pred_labels[keep]
    return pred_boxes,pred_scores,pred_labels



  def forward(self, feat, proposals, image_shape, target):
      # Training mode with ground truth available
      if self.training and target is not None:
          gt_boxes = target['bboxes'][0]
          gt_labels = target['labels'][0]
          
          # Assign labels and ground truth boxes to proposals
          labels, matched_gt_boxes_for_proposals = self.assign_target_to_proposals(proposals, gt_boxes, gt_labels)
          
          # Sample positive and negative proposals
          sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(
              labels, positive_count=32, total_count=128
          )
          sampled_idxs = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]

          # Filter proposals based on sampling
          proposals = proposals[sampled_idxs]
          labels = labels[sampled_idxs]
          matched_gt_boxes_for_proposals = matched_gt_boxes_for_proposals[sampled_idxs]
          
          # Compute regression targets
          regression_targets = boxes_to_transformation_targets(
              matched_gt_boxes_for_proposals, proposals
          )

          # ROI Pooling
          spatial_scale = 0.00625
          proposal_roi_pool_feats = torchvision.ops.roi_pool(
              feat,
              boxes=proposals,
              output_size=self.pool_size,
              spatial_scale=spatial_scale
          )

          # Feature processing
          proposal_roi_pool_feats = proposal_roi_pool_feats.flatten(start_dim=1)
          box_fc_6 = torch.nn.functional.relu(self.fc6(proposal_roi_pool_feats))
          box_fc_7 = torch.nn.functional.relu(self.fc7(box_fc_6))
          
          # Classification and regression predictions
          cls_scores = self.cls_score(box_fc_7)
          box_transform_pred = self.bbox_pred(box_fc_7)

          # Compute losses
          classification_loss = torch.nn.functional.cross_entropy(
              cls_scores, labels
          )

          # Localization loss only for foreground proposals
          fg_proposal_idx = torch.where(labels > 0)[0]
          localization_loss = torch.nn.functional.smooth_l1_loss(
              box_transform_pred[fg_proposal_idx, labels[fg_proposal_idx] * 4 : (labels[fg_proposal_idx] + 1) * 4],
              regression_targets[fg_proposal_idx],
              beta=1/9,
              reduction='sum'
          ) / max(1, fg_proposal_idx.numel())

          return {
              'frcnn_classification_loss': classification_loss,
              'frcnn_localization_loss': localization_loss
          }
      
      # Inference mode
      else:
          # ROI Pooling
          spatial_scale = 0.00625
          proposal_roi_pool_feats = torchvision.ops.roi_pool(
              feat,
              boxes=proposals,
              output_size=self.pool_size,
              spatial_scale=spatial_scale
          )

          # Feature processing
          proposal_roi_pool_feats = proposal_roi_pool_feats.flatten(start_dim=1)
          box_fc_6 = torch.nn.functional.relu(self.fc6(proposal_roi_pool_feats))
          box_fc_7 = torch.nn.functional.relu(self.fc7(box_fc_6))
          
          # Classification and regression predictions
          cls_scores = self.cls_score(box_fc_7)
          box_transform_pred = self.bbox_pred(box_fc_7)

          # Convert predictions to probabilities
          pred_scores = torch.softmax(cls_scores, dim=-1)
          pred_labels = torch.argmax(pred_scores, dim=-1)
          
          # Apply box regression
          pred_boxes = apply_regression_pred_to_anchors_or_proposals(
              box_transform_pred, proposals
          )

          # Clamp boxes to image boundary
          pred_boxes = clamp_boxes_to_image_boundary(pred_boxes, image_shape)

          # Filter predictions
          pred_boxes, pred_scores, pred_labels = self.filter_predictions(
              pred_boxes, pred_scores, pred_labels
          )

          return {
              'pred_boxes': pred_boxes,
              'pred_scores': pred_scores,
              'pred_labels': pred_labels
          }
