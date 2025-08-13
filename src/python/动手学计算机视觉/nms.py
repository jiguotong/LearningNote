"""""
@Author     :   jiguotong
@Contact    :   1776220977@qq.com
@site       :   
-----------------------------------------------
@Time       :   2025/4/30
@Description:   mask-based的nms(Non Maximum Suppression, 非极大值抑制)各种版本
"""""
import torch

import torch
import numpy as np

def mask_nms(masks, scores, labels=None, iou_threshold=0.5):
    """
    Perform mask NMS, optionally class-wise if labels are provided.

    Args:
        masks (Tensor): [N, H, W] binary or soft masks.
        scores (Tensor): [N] confidence scores.
        labels (Tensor or None): [N] class indices. If None, standard NMS is applied.
        iou_threshold (float): IoU threshold for suppression.

    Returns:
        keep_inds (List[int]): indices of masks to keep.
    """
    def _nms_single_class(masks, scores, iou_threshold):
        N = masks.shape[0]
        masks_flat = masks.view(N, -1)
        areas = masks_flat.sum(dim=1)
        sorted_inds = torch.argsort(scores, descending=True)
        keep = []

        while sorted_inds.numel() > 0:
            i = sorted_inds[0]
            keep.append(i.item())

            if sorted_inds.numel() == 1:
                break

            i_mask = masks_flat[i].unsqueeze(0)
            rest_masks = masks_flat[sorted_inds[1:]]

            inter = (rest_masks * i_mask).sum(dim=1)
            union = areas[i] + areas[sorted_inds[1:]] - inter
            iou = inter.float() / union.float()

            below_threshold_inds = (iou <= iou_threshold).nonzero(as_tuple=True)[0]
            sorted_inds = sorted_inds[below_threshold_inds + 1]

        return keep

    if labels is None:
        return _nms_single_class(masks, scores, iou_threshold)
    else:
        keep_inds = []
        unique_labels = labels.unique()
        for cls in unique_labels:
            cls_inds = (labels == cls).nonzero(as_tuple=True)[0]
            cls_masks = masks[cls_inds]
            cls_scores = scores[cls_inds]

            keep = _nms_single_class(cls_masks, cls_scores, iou_threshold)
            keep_inds.extend(cls_inds[keep].tolist())

        return keep_inds

def soft_mask_nms(
    masks, scores, labels=None,
    iou_threshold=0.5,
    method='gaussian',
    sigma=0.5,
    score_threshold=0.5
):
    """
    Soft-NMS for masks, supporting both single-class and multi-class.
    Reference: https://arxiv.org/abs/1704.04503 Soft-NMS -- Improving Object Detection With One Line of Code
    Args:
        masks (Tensor): [N, H, W] masks.
        scores (Tensor): [N] scores.
        labels (Tensor or None): [N] class indices. If None, assume single-class.
        iou_threshold (float): used for 'linear' method.
        method (str): 'linear' or 'gaussian'.
        sigma (float): used for 'gaussian' method.
        score_threshold (float): drop masks with scores below this value.

    Returns:
        keep_inds (List[int]): indices to keep.
    """
    def _soft_nms_single_class(masks, scores):
        N = masks.size(0)
        masks_flat = masks.view(N, -1)
        scores = scores.clone()
        areas = masks_flat.sum(dim=1)

        indices = torch.arange(N)
        keep = []

        while indices.numel() > 0:
            max_idx = torch.argmax(scores[indices])
            curr_idx = indices[max_idx]
            keep.append(curr_idx.item())

            if indices.numel() == 1:
                break

            curr_mask = masks_flat[curr_idx].unsqueeze(0)
            rest = indices[indices != curr_idx]
            rest_masks = masks_flat[rest]

            inter = (rest_masks * curr_mask).sum(dim=1)
            union = areas[rest] + areas[curr_idx] - inter
            iou = inter / union

            if method == 'linear':
                decay = torch.ones_like(iou)
                decay[iou > iou_threshold] -= iou[iou > iou_threshold]
            elif method == 'gaussian':
                decay = torch.exp(- (iou ** 2) / sigma)
            else:
                raise ValueError("Unknown method: " + method)

            scores[rest] *= decay
            indices = rest[scores[rest] > score_threshold]

        return keep

    if labels is None:
        return _soft_nms_single_class(masks, scores)
    else:
        keep_inds = []
        unique_labels = labels.unique()
        for cls in unique_labels:
            cls_inds = (labels == cls).nonzero(as_tuple=True)[0]
            cls_masks = masks[cls_inds]
            cls_scores = scores[cls_inds]

            keep = _soft_nms_single_class(cls_masks, cls_scores)
            keep_inds.extend(cls_inds[keep].tolist())

        return keep_inds

def fast_mask_nms(masks, scores, labels=None, iou_threshold=0.5, top_k=200):
    """
    Fast mask NMS supporting both single-class and multi-class cases.
    Reference: https://arxiv.org/abs/1904.02689 YOLACT: Real-time Instance Segmentation
    Args:
        masks (Tensor): [N, H, W] float or binary masks.
        scores (Tensor): [N] scores for each mask.
        labels (Tensor or None): [N] class indices. If None, apply single-class NMS.
        iou_threshold (float): IoU threshold for suppression.
        top_k (int): Only consider top-k candidates by score.

    Returns:
        keep_indices (List[int]): indices of masks to keep.
    """
    def _fast_nms_single_class(masks, scores):
        N = scores.size(0)
        if N == 0:
            return []

        scores_sorted, idx = scores.sort(descending=True)
        masks = masks[idx][:top_k]
        scores_sorted = scores_sorted[:top_k]

        N = masks.size(0)
        flattened_masks = masks.view(N, -1).float()
        inter = torch.mm(flattened_masks, flattened_masks.t())  # [N, N]
        area = flattened_masks.sum(dim=1).view(-1, 1)
        union = area + area.t() - inter
        iou = inter / (union + 1e-6)

        iou.triu_(diagonal=1)
        iou_max, _ = iou.max(dim=0)
        keep = (iou_max <= iou_threshold).nonzero(as_tuple=True)[0]

        return idx[keep].tolist()

    if labels is None:
        return _fast_nms_single_class(masks, scores)
    else:
        keep_indices = []
        unique_labels = labels.unique()

        for cls in unique_labels:
            cls_inds = (labels == cls).nonzero(as_tuple=True)[0]
            cls_masks = masks[cls_inds]
            cls_scores = scores[cls_inds]

            keep = _fast_nms_single_class(cls_masks, cls_scores)
            keep_indices.extend(cls_inds[keep].tolist())

        return keep_indices

def matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=2.0):
    """Matrix NMS for multi-class masks.
    # Reference: https://arxiv.org/abs/2003.10152 SOLOv2: Dynamic and Fast Instance Segmentation
    Args:
        seg_masks (Tensor): shape (n, h, w)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss' 
        sigma (float): std in gaussian method

    Returns:
        Tensor: cate_scores_update, tensors of shape (n) 
    """
    N = len(cate_scores)
    if N == 0:
        return []
    
    seg_masks = seg_masks.reshape(N, -1).float()
    
    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.T)
    
    # union.
    mask_areas = seg_masks.sum(dim=1).float()
    sum_masks_x = mask_areas.expand(N, N)
    union_matrix = sum_masks_x + sum_masks_x.T - inter_matrix
    
    # iou.
    iou_matrix = (inter_matrix / union_matrix).triu(diagonal=1)
    
    # label_specific matrix.w
    cate_labels_x = cate_labels.expand(N, N)

    # label_matrix
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)    # 上三角矩阵，标记相同类别的掩膜对（1为同类，0为不同类）。

    # IoU single class
    iou_matrix = iou_matrix * label_matrix   # 仅保留同类掩膜的IoU
    
    # max IoU for each: N
    ious_max, _ = iou_matrix.max(0)  # 同类掩膜的最大IoU
    ious_max = ious_max.expand(N, N).transpose(1, 0)
    
    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (iou_matrix ** 2)) / torch.exp(-1 * sigma * (ious_max ** 2))
    elif kernel == 'linear':
        decay_matrix = (1 - iou_matrix) / (1 - ious_max)
    else:
        raise NotImplementedError

    # update the score.
    decay_matrix, _= decay_matrix.min(0)        # 取最大惩罚
    new_cate_scores = cate_scores * decay_matrix    # 最终分数 = 原始分数 × 衰减系数。
    return new_cate_scores


if __name__ == "__main__":
    masks = torch.randint(0, 2, (6, 28, 28))
    scores = torch.tensor([0.9, 0.88, 0.85, 0.83, 0.7, 0.65])
    labels = torch.tensor([1, 1, 1, 2, 2, 2])  # 两个类别

    keep_indices = mask_nms(masks, scores, labels, iou_threshold=0.5)
    print("Kept:", keep_indices)

    keep = soft_mask_nms(masks, scores, labels,
                                    iou_threshold=0.5,
                                    method='gaussian',
                                    sigma=0.5)
    print("Kept indices:", keep)

    keep_inds = fast_mask_nms(masks, scores, labels, iou_threshold=0.5)
    print("Keep:", keep_inds)