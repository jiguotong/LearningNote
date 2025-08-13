"""""
@Author     :   jiguotong
@Contact    :   1776220977@qq.com
@site       :   
-----------------------------------------------
@Time       :   2025/7/9
@Description:   《动手学计算机视觉》书中实现的无监督图像分割算法k-means
"""""
from sklearn.cluster import KMeans
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# 为每一个类别赋予一个对应的颜色，用于展示
def decode_segmap(label_mask, plot=False):
    label_colours = np.asarray([[79, 103, 67], [143, 146, 126], 
                                [129, 94, 64], [52, 53, 55],
                                [96, 84, 70], [164, 149, 129]])
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    # 为每个类别赋予对应的R、G、B值
    for ll in range(0, 6):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]

    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    return rgb

def segmentation_single():
    # 输入一张图像，图像来源于参考文献[5]
    image = imread('segmentation.jpeg')[:,:,:3]
    # 将RGB值统一到0-1内
    if np.max(image)>1:
        image = image / 255

    X = image.reshape(-1, image.shape[2])

    # 利用k-means算法进行聚类
    segmented_imgs = []

    # 设定聚类中心个数
    n_cluster= 4
    kmeans = KMeans(n_clusters=n_cluster, random_state=42).fit(X)
    print(np.unique(kmeans.labels_))

    # 获得预测的标签
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_imgs = decode_segmap(kmeans.labels_.reshape(image.shape[0],image.shape[1]))

    # 展示结果
    plt.imshow(image[:,:,:3])
    plt.title('Original image')
    plt.axis('off')
    plt.show(block=True)

    plt.imshow(segmented_imgs.astype(np.uint8))
    plt.title('{} center'.format(n_cluster))
    plt.axis('off')
    plt.show(block=True)
    
def segmentation_with_coord():
    image = imread('segmentation.jpeg')[:,:,:3]
    # 将RGB值统一到0-255内
    if np.max(image)>1:
        image = image / 255
        
    sp = image.shape
    # 增加xy坐标的信息
    # 设定一个权重，对坐标信息加权
    weight = 2
    y = weight * np.array([[i for i in range(sp[1])] 
                        for j in range(sp[0])]) / sp[0] / sp[1]
    x = weight * np.array([[j for i in range(sp[1])] 
                        for j in range(sp[0])])/ sp[0] / sp[1]
    image = np.append(image, x.reshape(sp[0], sp[1], 1), axis=2)
    image = np.append(image, y.reshape(sp[0], sp[1], 1), axis=2)

    X = image.reshape(-1, image.shape[2])
    segmented_imgs = []

    # 将 K 分别设置为6、5、4、3、2
    n_colors = (6, 5, 4, 3, 2)
    for n_cluster in n_colors:
        kmeans = KMeans(n_clusters=n_cluster, random_state=42).fit(X)
        segmented_img = kmeans.cluster_centers_[kmeans.labels_]
        segmented_imgs.append(decode_segmap(
            kmeans.labels_.reshape(image.shape[0],
                                image.shape[1])).astype(np.uint8))

    # 展示结果
    plt.figure(figsize=(12,8))
    plt.subplot(231)
    plt.imshow(image[:,:,:3])
    plt.title('Original image')
    plt.axis('off')
    
    for idx,n_clusters in enumerate(n_colors):
        plt.subplot(232+idx)
        plt.imshow(segmented_imgs[idx])
        plt.title('{} center'.format(n_clusters))
        plt.axis('off')
    plt.show(block=True)
    
if __name__=="__main__":
    segmentation_single()
    segmentation_with_coord()