# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as tF

eps = 1e-10
class L2DIS:
    def __init__(self, factor=512) -> None:
        self.factor = factor

    def __call__(self, X, Y):
        x_col = X.unsqueeze(-2)
        y_row = Y.unsqueeze(-3)
        # print("[XY]:", x_col.shape, y_row.shape)
        C = torch.norm(x_col - y_row, dim=-1)
        C = C / self.factor
        return C

class P2PLoss(nn.modules.loss._Loss):
    def __init__(self, factor=1, reduction='mean') -> None:
        super().__init__()
        self.factor = factor
        self.cost = L2DIS(1)
        self.min_radis = 8
        self.max_radis = 96

        self.cost_class = 1
        self.cost_point = 8


    def forward(self, dens, seqs, down, masks = None, crop_den_masks=None):
        bs = len(seqs)
        oot_loss, cnt_loss = 0, 0
        for i in range(bs):
            den, seq = dens[i], seqs[i]
            if crop_den_masks is not None:
                crop_den_mask = crop_den_masks[i]
            if masks is not None:
                mask = masks[i]
            den = den.permute(1, 2, 0)
            H, W = den.shape[:2]
            if seq.size(0) < 1:
               cnt_loss = cnt_loss + tF.binary_cross_entropy_with_logits(den, torch.zeros_like(den), weight=torch.ones_like(den) * 0.5)
            else:
                A_coord = torch.stack(torch.meshgrid(
                    torch.arange(H), torch.arange(W)),
                dim=-1).view(1, -1, 2) * down + (down - 1) / 2
                A = den.view(1, -1, 1)
                A_coord = A_coord.to(seq).float().view(1, -1, 2)
                
                B_coord = seq[None, :, :2].float()
                B = torch.ones(seq.size(0)).float().to(A).view(1, -1, 1)
                if masks is not None:
                    MB = mask.view_as(B).to(B)
                with torch.no_grad():
                    C = self.cost(A_coord, B_coord)
                    minC, mcidx = C.min(dim=-1, keepdim=True)
                    M = torch.zeros_like(C).scatter_(-1, mcidx, 1.0) * (C < self.max_radis)
                    
                    maxC = (minC.view_as(A) * M).amax(dim=1, keepdim=True)
                    maxC = torch.clip(maxC, min=self.min_radis, max=self.max_radis)
                    C = C / maxC

                    C = C * self.cost_point - A * self.cost_class
                    vid = (M.sum(dim=1) > 0).view(-1)
                    C, M = C[..., vid], M[..., vid]
                    B, B_coord = B[:, vid, :], B_coord[:, vid, :]

                    C2 = M * C + (1 - M) * (C.max() + 1)
                    minC2, mcidx2 = C2.min(dim=1, keepdim=True)
                    T = torch.zeros_like(C2).scatter_(1, mcidx2, 1.0).sum(dim=-1).view(1, -1, 1)
                    T = (T > 0.5).to(A).view_as(A)
                    W = T + 1

                    if masks is not None:
                        M = (M @ MB[:, vid, :]) + 1 - M.sum(dim=-1).view_as(A)
                        W = W * M
                if crop_den_masks is not None:
                    W = W * crop_den_mask.view_as(W)
                cnt_loss = cnt_loss + tF.binary_cross_entropy_with_logits(A, T, weight=W)
                

        loss = (oot_loss + cnt_loss) / bs
        return loss
    