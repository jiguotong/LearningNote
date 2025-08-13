"""""
@Author     :   jiguotong
@Contact    :   1776220977@qq.com
@site       :   
-----------------------------------------------
@Time       :   2025/7/23
@Description:   faster rcnn见https://github.com/jiguotong/faster-rcnn
"""""
import numpy as np
import torch


def roi_pooling(feature_map, rois, output_size):
    # feature_map: 输入特征图，shape为 (H, W, C)
    # rois: 包含感兴趣区域的坐标和大小，shape为 (num_rois, 4)，其中坐标已经映射到特征图上
    # output_size: RoI池化后输出的大小，为一个标量或一个长度为2的元组

    # 将RoI坐标和大小转换为整数
    rois = np.round(rois).astype(np.int32)

    # 计算每个RoI的高度和宽度
    roi_heights = rois[:, 2] - rois[:, 0]  # (num_rois,)
    roi_widths = rois[:, 3] - rois[:, 1]  # (num_rois,)

    # 计算每个RoI的垂直和水平步长
    stride_y = roi_heights / output_size[0]
    stride_x = roi_widths / output_size[1]

    # 初始化输出数组
    pooled_rois = np.zeros((rois.shape[0], output_size[0], output_size[1], feature_map.shape[2]))

    # 对每个RoI进行池化操作
    for i, roi in enumerate(rois):
        # 获取RoI的坐标
        y1, x1, y2, x2 = roi

        # 计算每个RoI的垂直和水平池化步长
        dy = (y2 - y1) / output_size[0]
        dx = (x2 - x1) / output_size[1]

        # 对每个通道进行池化
        for c in range(feature_map.shape[2]):
            for y in range(output_size[0]):
                for x in range(output_size[1]):
                    # 计算当前池化窗口的坐标
                    y_start = int(np.round(y1 + y * dy))
                    x_start = int(np.round(x1 + x * dx))
                    y_end = int(np.round(y_start + dy))
                    x_end = int(np.round(x_start + dx))

                    # 取出当前池化窗口对应的特征图区域
                    patch = feature_map[y_start:y_end, x_start:x_end, c]

                    # 对当前池化窗口进行池化
                    pooled_rois[i, y, x, c] = np.max(patch)

    return pooled_rois