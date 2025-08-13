"""""
@Author     :   jiguotong
@Contact    :   1776220977@qq.com
@site       :   
-----------------------------------------------
@Time       :   2025/7/11
@Description:   《动手学计算机视觉》书中实现的基于词袋模型的图像分类算法
"""""
import os
import cv2
import math
import numpy as np
from imutils import paths
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import AdditiveChi2Sampler

## 1.获取数据和标签
## 2.划分训练、测试集
## 3.从训练图像集中提取SIFT特征点及描述子，形成特征库
## 4.对特征库中的特征向量进行聚类，构建"视觉词典"
## 5.用所学到的视觉词典对每幅图像的特征进行编码，使用SPM算法进行位置优化
## 6.训练svm分类器，拟合训练图像
## 7.使用训练好的svm分类器，对测试数据进行预测

def extract_SIFT(img):
    sift = cv2.SIFT_create()
    descriptors = []
    for disft_step_size in DSIFT_STEP_SIZE:
        keypoints = [cv2.KeyPoint(x, y, disft_step_size)
                for y in range(0, img.shape[0], disft_step_size)
                    for x in range(0, img.shape[1], disft_step_size)]

        descriptors.append(sift.compute(img, keypoints)[1])
    
    return np.concatenate(descriptors, axis=0).astype('float64')


# 获取图像的SPM特征
def getImageFeaturesSPM(L, img, kmeans, k):
    W = img.shape[1]
    H = img.shape[0]   
    h = []
    for l in range(L+1):
        w_step = math.floor(W/(2**l))
        h_step = math.floor(H/(2**l))
        x, y = 0, 0
        for _ in range(2**l):
            x = 0
            for __ in range(2**l):
                desc = extract_SIFT(img[y:y+h_step, x:x+w_step])                
                predict = kmeans.predict(desc)
                histo = np.bincount(predict, 
                        minlength=k).reshape(1,-1).ravel()
                weight = 2**(l-L)
                h.append(weight*histo)
                x = x + w_step
            y = y + h_step
            
    hist = np.array(h).ravel()
    hist /= np.sum(hist)
    return hist


if __name__=="__main__":
    ## 1.获取数据和标签
    # 图像数据
    data = []
    # 图像对应的标签
    labels = []
    # 储存标签信息的临时变量
    labels_tep = []
    image_paths = list(paths.list_images('./caltech-101'))

    for image_path in image_paths:
        # 获取图像类别
        label = image_path.split(os.path.sep)[-2]
        # 读取每个类别的图像
        image = cv2.imread(image_path)
        # 将图像通道从BGR转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 统一输入图像的尺寸
        image = cv2.resize(image, (200,200), interpolation=cv2.INTER_AREA)
        data.append(image)
        labels_tep.append(label)

    # 建立标签名称与标签代码的对应关系
    name2label = {}
    tep = {}
    for idx, name in enumerate(labels_tep):
        tep[name] = idx
    for idx, name in enumerate(tep):
        name2label[name] = idx
    for idx, image_path in enumerate(image_paths):
        labels.append(name2label[image_path.split(os.path.sep)[-2]])

    data = np.array(data)
    labels = np.array(labels)
    
    ## 2.划分训练、测试集
    (x_train, X_remain, y_train, Y_remain) = train_test_split(data, labels, test_size=0.4, stratify=labels, random_state=42)
    (x_val, x_test, y_val, y_test) = train_test_split(X_remain, Y_remain, test_size=0.5, random_state=42)
    print(f"x_train examples: {x_train.shape}\n\
        x_test examples: {x_test.shape}\n\
            x_val examples: {x_val.shape}")
    
    ## 3.从训练图像集中提取SIFT特征点及描述子，形成特征库
    # 构建一个词典，储存每一个类别的sift信息
    num_classes = 101       # 数据集标签类别101
    vec_dict = {i:{'kp':[], 'des':{}} for i in range(num_classes+1)}

    sift = cv2.SIFT_create()
    for i in range(x_train.shape[0]):
        # 对图像正则化
        tep = cv2.normalize(x_train[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        # 计算图像的SIFT特征
        kp_vector, des_vector = sift.detectAndCompute(tep, None)
        # 特征点和描述符信息储存进词典中
        vec_dict[y_train[i]]['kp'] += list(kp_vector)
        for k in range(len(kp_vector)):
            # des使用kp_vector将其一一对应
            vec_dict[y_train[i]]['des'][kp_vector[k]] = des_vector[k]
            
    # 设置最少特征点的数目
    bneck_value = float("inf")

    # 确定bneck_value的数值
    for i in range(102):
        if len(vec_dict[i]['kp']) < bneck_value:
            bneck_value = len(vec_dict[i]['kp'])

    # 按照每一个SIFT特征点的响应值对特征进行降序排序
    for i in range(102):
        kp_list = vec_dict[i]['kp'] = sorted((vec_dict[i]['kp']), 
                                            key=lambda x: x.response, 
                                            reverse=True)
            
    # 为每个类别选择同样多的特征点用于聚类。特征点个数bneck_value
    feature_vec_list=[]     # 最终用于聚类的特征库
    for i in range(bneck_value):
        feature_vec_list.append(vec_dict[0]['des'][vec_dict[0]['kp'][i]])

    for i in range(1, 102):
        for j in range(bneck_value):
            feature_vec_list.append(vec_dict[i]['des'][vec_dict[i]['kp'][j]])

    feature_vec_list = np.float64(feature_vec_list)
    
    ## 4.对特征库中的特征向量进行聚类，构建"视觉词典"
    N_clusters = 200
    kmeans = KMeans(n_clusters=N_clusters, random_state=0).fit(feature_vec_list)
    
    ## 5.用所学到的视觉词典对每幅图像的特征进行编码，使用SPM算法进行位置优化  
    # SPM 的一些超参数
    pyramid_level = 2
    DSIFT_STEP_SIZE = [4, 8]

    # 训练图像处理编码
    hist_train_vector = []
    for i in range(x_train.shape[0]):
        tep = cv2.normalize(x_train[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        # 提取图像的SPM特征
        hist_SPM = getImageFeaturesSPM(pyramid_level, tep, kmeans, N_clusters)  # hist_SPM.shape(200*(1+4+16))即4200
        # 将提取的特征加入直方图中
        hist_train_vector.append(hist_SPM)
    hist_train_vector = np.array(hist_train_vector)
    
    # 测试图像处理编码
    hist_test_vector = []
    for i in range(x_test.shape[0]):
        tep = cv2.normalize(x_test[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        hist_SPM = getImageFeaturesSPM(pyramid_level, tep, kmeans, N_clusters)
        hist_test_vector.append(hist_SPM)
    hist_test_vector = np.array(hist_test_vector)
    
    # transform使用一种特征映射的方法
    transformer = AdditiveChi2Sampler() #将原始特征通过加性卡方核（Additive Chi² kernel）映射到一个近似的线性空间，使得线性模型可以更好地处理非线性问题。
    transformer = transformer.fit(np.concatenate([hist_train_vector, hist_test_vector], axis=0))

    ## 6.训练svm分类器，拟合训练图像
    # 构建SVM分类器
    classifier = svm.LinearSVC()
    
    # 将训练的直方图进行特征映射
    hist_train_vector = transformer.transform(hist_train_vector)

    # 对数据进行拟合
    classifier.fit(hist_train_vector, y_train)
    
    ## 7.使用训练好的svm分类器，对测试数据进行预测
    # 将测试的直方图进行特征映射
    hist_test_vector = transformer.transform(hist_test_vector)

    # 计算分类 top-1 错误率
    top1_error = classifier.predict(hist_test_vector) - y_test
    tep = len(top1_error[top1_error!=0])
    print('top-1 error: ', tep/len(y_test))
