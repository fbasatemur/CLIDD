import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import BasicLayer, ResNetLayer
from .triton_plugin import deformable_sample_project

class Model(nn.Module):

    def __init__(self, c1, c2, c3, r1, r2, r3, cdesc, cdetect, M, max_offset):
        super().__init__()

        csum = c1 + c2 + c3
        self.max_offset = max_offset

        self.block1 = nn.Sequential(
            BasicLayer(3, c1, 4, 2, 1),
            BasicLayer(c1, c1),
            *[ResNetLayer(c1, c1) for _ in range(r1)],
        )
        self.block2 = nn.Sequential(
            nn.AvgPool2d(4, 4),
            *[ResNetLayer(c1, c2) if _ == 0 else ResNetLayer(c2, c2) for _ in range(r2)],
        )
        self.block3 = nn.Sequential(
            nn.AvgPool2d(4, 4),
            *[ResNetLayer(c2, c3) if _ == 0 else ResNetLayer(c3, c3) for _ in range(r3)],
        )
        
        self.conv1 = nn.Conv2d(c1, cdetect, 1)
        self.conv2 = nn.Conv2d(c2, cdetect, 1)
        self.conv3 = nn.Conv2d(c3, cdetect, 1)
        
        self.desc_head = nn.Identity()
        self.score_head = nn.Sequential(
            nn.Conv2d(cdetect, cdetect, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(cdetect, 4, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        
        self.dcn = nn.Conv2d(csum, M*2*3, 1)
        self.post_conv = nn.Conv2d(csum, cdesc, (1, M))

    def forward(self, x):

        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        desc = (x1, x2, x3)

        feat = self.conv2(x2) + F.interpolate(self.conv3(x3), scale_factor=4, mode='bilinear', align_corners=False)
        feat = self.conv1(x1) + F.interpolate(feat, scale_factor=4, mode='bilinear', align_corners=False)
        
        score = self.score_head(feat)
        
        return desc, score
    
    def sample(self, dense, kpts, *, align_corners=False):
        H, W = dense[0].shape[-2:]
        H = H * 2
        W = W * 2
        c0 = dense[0].shape[1]
        c1 = dense[1].shape[1]
        c2 = dense[2].shape[1]

        weight_offset1, weight_offset2, weight_offset3 = self.dcn.weight.split([c0, c1, c2], 1)
        bias_offset = self.dcn.bias
        weight_post1, weight_post2, weight_post3 = self.post_conv.weight.split([c0, c1, c2], 1)
        bias_post = self.post_conv.bias
        
        dense[0] = dense[0].permute(0,2,3,1).contiguous()
        dense[1] = dense[1].permute(0,2,3,1).contiguous()
        dense[2] = dense[2].permute(0,2,3,1).contiguous()
        is_input_nhwc = True

        offset = deformable_sample_project(dense[0], kpts, weight_offset1, None, is_input_nhwc=is_input_nhwc, align_corners=align_corners) + \
                 deformable_sample_project(dense[1], kpts, weight_offset2, None, is_input_nhwc=is_input_nhwc, align_corners=align_corners) + \
                 deformable_sample_project(dense[2], kpts, weight_offset3, bias_offset, is_input_nhwc=is_input_nhwc, align_corners=align_corners)
        
        offset = offset.unflatten(-1, (-1, 2)).clamp(-self.max_offset, self.max_offset)
        offset_1, offset_2, offset_3 = (offset / torch.tensor([W, H]).to(kpts) * 2).chunk(3, -2)
        kpts = kpts.to(offset)
        
        desc = deformable_sample_project(dense[0], kpts + offset_1, weight_post1, None, is_input_nhwc=is_input_nhwc, align_corners=align_corners) + \
               deformable_sample_project(dense[1], kpts + offset_2, weight_post2, None, is_input_nhwc=is_input_nhwc, align_corners=align_corners) + \
               deformable_sample_project(dense[2], kpts + offset_3, weight_post3, bias_post, is_input_nhwc=is_input_nhwc, align_corners=align_corners)
        return F.normalize(desc, 2, -1)
    
