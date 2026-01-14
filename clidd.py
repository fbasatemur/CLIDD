import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.model import Model

class CLIDD(nn.Module):
    
    cfgs = {
        'A48': {'c1': 4, 'c2': 4, 'c3': 4, 'r1': 1, 'r2': 1, 'r3': 1, 'cdesc': 48, 'cdetect': 4, 'M': 4, 'max_offset': 128},
        
        'N64': {'c1': 8, 'c2': 8, 'c3': 8, 'r1': 1, 'r2': 1, 'r3': 1, 'cdesc': 64, 'cdetect': 8, 'M': 8, 'max_offset': 128},
        
        'T64': {'c1': 8, 'c2': 16, 'c3': 24, 'r1': 1, 'r2': 1, 'r3': 1, 'cdesc': 64, 'cdetect': 8, 'M': 8, 'max_offset': 128},
        
        'S64': {'c1': 8, 'c2': 24, 'c3': 32, 'r1': 1, 'r2': 1, 'r3': 1, 'cdesc': 64, 'cdetect': 8, 'M': 16, 'max_offset': 128},
        
        'M64': {'c1': 16, 'c2': 32, 'c3': 48, 'r1': 1, 'r2': 1, 'r3': 1, 'cdesc': 64, 'cdetect': 8, 'M': 16, 'max_offset': 128},
        
        'L64': {'c1': 16, 'c2': 48, 'c3': 96, 'r1': 1, 'r2': 1, 'r3': 1, 'cdesc': 64, 'cdetect': 8, 'M': 16, 'max_offset': 128},
        
        'G128': {'c1': 16, 'c2': 64, 'c3': 256, 'r1': 1, 'r2': 1, 'r3': 1, 'cdesc': 128, 'cdetect': 8, 'M': 32, 'max_offset': 128},

        'E128': {'c1': 16, 'c2': 64, 'c3': 256, 'r1': 1, 'r2': 2, 'r3': 2, 'cdesc': 128, 'cdetect': 8, 'M': 32, 'max_offset': 128},
        
        'U128': {'c1': 32, 'c2': 128, 'c3': 256, 'r1': 1, 'r2': 2, 'r3': 2, 'cdesc': 128, 'cdetect': 8, 'M': 32, 'max_offset': 128},
    }
    
    def __init__(self, cfg, top_k, radius=2, score=-5):
        super().__init__()
        assert top_k is None or top_k > 0
        self.top_k = top_k
        self.radius = radius
        self.score_thresh = score
        
        self.model = Model(**self.cfgs[cfg])
        self.model.load_state_dict(torch.load(f'./weights/{cfg}.pth', 'cpu'))
        
        if radius > 0:
            self.mp = nn.MaxPool2d(radius * 2 + 1, 1, radius)
        
    @torch.inference_mode()
    def forward(self, x):
        B, _, oH, oW = x.shape
        nH = oH // 32 * 32
        nW = oW // 32 * 32
        size = torch.tensor([nW, nH], dtype=x.dtype, device=x.device)
        scale = torch.tensor([oW/nW, oH/nH], dtype=x.dtype, device=x.device)
        if oW != nW or oH != nH:
            x = F.interpolate(x, (nH, nW), mode='bilinear', align_corners=True)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        raw_desc, raw_detect = self.model(x)
        
        if self.radius > 0:
            detect1 = raw_detect == self.mp(raw_detect)
        else:
            detect1 = torch.ones_like(raw_detect, dtype=torch.bool)
        detect1[..., :, :4] = False
        detect1[..., :, -4:] = False
        detect1[..., :4, :] = False
        detect1[..., -4:, :] = False
        
        detect2 = raw_detect > self.score_thresh
        detect = torch.logical_and(detect1, detect2)[:,0]
        H = torch.arange(detect.shape[-2], dtype=x.dtype, device=x.device)
        W = torch.arange(detect.shape[-1], dtype=x.dtype, device=x.device)
        H, W = torch.meshgrid(H, W)
        ind = torch.stack([W, H], dim=-1)
        kpts = [ind[detect[b]] for b in range(B)]
        scores = [raw_detect[b,0,detect[b]] for b in range(B)]
        
        if self.top_k is not None:
            for i in range(B):
                score, idx = scores[i].topk(min(self.top_k, scores[i].shape[0]))
                scores[i] = score
                kpts[i] = kpts[i][idx]
        
        descs = [self.model.sample([r[b:b+1] for r in raw_desc], (kpts[b] + 0.5).reshape(1, -1, 1, 2) / size * 2 - 1)[0] if kpts[b].shape[0] > 0 else raw_detect.new_zeros([0, 0]) for b in range(B)]
        
        return [  
				   {'keypoints': kpts[b] * scale,
					'scores': scores[b],
					'descriptors': descs[b]} for b in range(B) 
			   ]
        
    def match(self, desc0: torch.Tensor, desc1: torch.Tensor, beta=20):
        # beta 30 for non-nms sparse indoor scenes
        if desc0.shape[0] == 0 or desc1.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        else:
            dist: torch.Tensor = desc0 @ desc1.t()
            # memory efficient
            dist.sub_(1).multiply_(beta).exp_()
            sum1 = dist.sum(dim=-1, keepdim=True)
            sum2 = dist.sum(dim=-2, keepdim=True)
            dist = dist.square_().div_(sum1).div_(sum2)
            
            nn12 = torch.max(dist, dim=1)[1]
            nn21 = torch.max(dist, dim=0)[1]
            ids1 = torch.arange(0, dist.shape[0], device=dist.device)
            mask = (ids1 == nn21[nn12])
            matches = torch.stack([ids1[mask], nn12[mask]])
            
            dist = dist[ids1[mask], nn12[mask]]
            mask = dist > 0.01
            matches = matches[:, mask]
            idxs1 = matches[0].data.cpu().numpy()
            idxs2 = matches[1].data.cpu().numpy()
            return idxs1, idxs2