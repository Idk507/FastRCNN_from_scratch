class FastRCNN(nn.Module):
  def __init__(self,num_classes=21):
    super(FastRCNN,self).__init__()
    vgg16 = torchvision.models.vgg16(pretrained=True)
    self.backbone = vgg16.features[:-1]
    self.rpn = RegionProposalNetwork(in_channel=512)
    self.roi_head = ROIHead(num_classes=num_classes,in_channels=512)

    for layer in self.backbone[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    self.image_mean = [0.485,0.456,0.406]
    self.image_std = [0.229,0.224,0.225]
    self.min_size = 600
    self.max_size = 1000
  
  def normalize_resize_image_and_boxes(self,image,bboxes):
    mean = torch.as_tensor(self.image_mean,dtype=image.dtype,device=image.device)
    std = torch.as_tensor(self.image_std,dtype=image.dtype,device=image.device)
    image = (image-mean[:,None,None])/std[:,None,None]
    h,w = image.shape[-2:]
    im_shape = torch.tensor(image.shape[-2:])
    min_size = float(torch.min(im_shape))
    max_size = float(torch.max(im_shape))
    scale_factor = torch.min(float(self.min_size)/min_size,float(self.max_size)/max_size)
    scale_factor = scale_factor.item()
    image = torch.nn.functional.interpolate(
        image,size=None,scale_factor = scale_factor,mode='bilinear',recompute_scale_factor=True,align_corners=False
    )
    #resize bboxes 
    if bboxes is not None:
      ratios = [torch.tensor(s,dtype=torch.float32,device=bboxes.device)/ 
                torch.tensor(s_orig,dtype=torch.float32,device=bboxes.device)
                for s,s_orig in zip(image.shape[-2:],(h,w))

      ]
      ratio_height,ratio_width= ratios 
      xmin,ymin,xmax,ymax = bboxes.unbind(2)
      xmin = xmin*ratio_width
      xmax = xmax*ratio_width
      ymin = ymin*ratio_height
      ymax = ymax*ratio_height
      bboxes = torch.stack((xmin,ymin,xmax,ymax),dim=2)
      return image,bboxes

  def forward(self,image,target=None):
    old_shape=image.shape[-2:]
    if self.training:
      image,bboxes = self.normalize_resize_image_and_boxes(image,target['bboxes']
                                                           )
      target['bboxes'] = bboxes
    else:
      image,bboxes = self.normalize_resize_image_and_boxes(image,None)
    features = self.backbone(image)

    #call RPN and get proposals
    rpn_output = self.rpn(image,features,target)
    proposals = rpn_output['proposals']

    #call ROI head and convert proposals to boxes 
    frcnn_output = self.roi_head(features,proposals,image.shape[-2:],target)

    if not self.training:
      frcnn_output['boxes'] = transform_boxes_to_original_size(
          frcnn_output['boxes'],
          image.shape[-2:],
          old_shape
      )
    
    return rpn_output,frcnn_output
