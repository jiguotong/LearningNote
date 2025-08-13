"""""
@Author     :   jiguotong
@Contact    :   1776220977@qq.com
@site       :   
-----------------------------------------------
@Time       :   2025/6/20
@Description:   使用opencv调用sift算法，并进行图像特征点对应
"""""
import cv2
import numpy as np
from matplotlib import pyplot as plt

def sift_feature_matching(img1_path, img2_path):
    # 读取图像
    img1_color = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img1_color = cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB)
    img2_color = cv2.imread(img2_path, cv2.IMREAD_COLOR)
    img2_color = cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB)
    
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_RGB2GRAY)
    # 检查图像是否成功加载
    if img1 is None or img2 is None:
        print("无法加载图像，请检查路径")
        return
    
    # 初始化SIFT检测器
    sift = cv2.SIFT_create()
    
    # 检测关键点和计算描述符
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    
    # 使用FLANN匹配器进行特征匹配
    # FLANN参数
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # 或传递空字典
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # 应用比率测试（Lowe's ratio test）筛选好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)
    
    # 绘制匹配结果
    # 绘制匹配线
    matched_img = cv2.drawMatches(
        img1_color, keypoints1, 
        img2_color, keypoints2, 
        good_matches, None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0),  # 绿色连线
        singlePointColor=(255, 0, 0)  # 蓝色关键点
    )
    
    # 显示结果
    plt.figure(figsize=(20, 10))
    plt.imshow(matched_img)
    plt.title("SIFT Feature Matching")
    plt.axis('off')
    plt.show(block=True)
    
    # 返回匹配信息
    return {
        'keypoints1': keypoints1,
        'keypoints2': keypoints2,
        'matches': good_matches
    }

# 使用示例
if __name__ == "__main__":
    # 替换为你的图像路径
    img1_path = "sift1.png"
    img2_path = "sift2.png"
    
    # 调用函数
    matching_result = sift_feature_matching(img1_path, img2_path)
    
    # 打印匹配信息
    print(f"图像1找到 {len(matching_result['keypoints1'])} 个关键点")
    print(f"图像2找到 {len(matching_result['keypoints2'])} 个关键点")
    print(f"找到 {len(matching_result['matches'])} 组良好匹配")