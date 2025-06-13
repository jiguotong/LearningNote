import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sns

# 一维卷积
class conv_1d():
    def __init__(self, a, b):
        # 输入信号
        self.a = a
        # 卷积核
        self.b = b
        #输入信号的坐标，默认从0开始
        self.ax = [i for i in range(len(a))]
        #卷积核的坐标，默认从0开始
        self.bx = [i for i in range(len(b))]
    
    def conv(self):
        lst1 = self.a
        lst2 = self.b
        l1 = len(lst1)
        l2 = len(lst2)
        lst1 = [0] * (l2 - 1) + lst1 + [0] * (l2 - 1)   # 填充大小为卷积核的大小减1
        lst2.reverse()  # 卷积核翻转！！！
        c = [0 for x in range(0, l1 + l2 - 1)]
        for i in range(l1 + l2 - 1):
            for j in range(l2):
                c[i] += lst1[i + j] * lst2[j]
        return c
    
    def plot(self):
        a = self.a
        b = self.b
        ax = self.ax
        bx = self.bx

        # 为了更直观的查看结果，我们分别绘制a、b与它们卷积得到的信号
        plt.figure(1)
        # 图一包含1行2列子图，当前画在第一行第一列图上
        plt.subplot(1, 2, 1)
        plt.title('Input')
        plt.bar(ax, a, color='lightcoral', width=0.2)
        plt.plot(ax, a, color='lightcoral')

        plt.figure(1)
        # 当前画在第一行第2列图上
        plt.subplot(1, 2, 2) 
        plt.title('Kernel')
        plt.bar(bx, b, color='lightgreen', width=0.2)
        #plt.plot(bx, b, color='lightgreen')


        # 计算输出信号以及其坐标，并画图 
        c = self.conv()
        length = len(c)
        cx = [i for i in range(length)]
        plt.figure()
        plt.title('Output')
        plt.bar(cx, c, color='lightseagreen', width=0.2)
        plt.plot(cx, c, color='lightseagreen')
           
# 二维卷积
class conv_2d():
    def __init__(self, image, kernel):
        self.img = image
        self.k = kernel
        
    # 定义二维卷积
    def conv(self):
        #return cv2.filter2D(self.img, -1, self.k)       # 直接调用库函数进行卷积操作
        return self.filter2d_numpy(self.img, self.k, 'same')    # 调用自写函数进行卷积操作

    def filter2d_numpy(self, image, kernel, padding='same'):
        """
        使用 NumPy 实现类似 cv2.filter2D 的二维卷积（逐通道、单核、自动翻转 kernel）。

        Args:
            image (ndarray): 输入图像，shape: (H, W) 或 (H, W, C)
            kernel (ndarray): 卷积核，shape: (kH, kW)
            padding (str): 'same' 或 'valid'

        Returns:
            output (ndarray): 与输入图像相同 shape 的滤波结果
        """
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # 转为 (H, W, 1)
        
        H, W, C = image.shape
        kH, kW = kernel.shape

        # 翻转 kernel（cv2.filter2D 是卷积，非互相关）
        kernel_flipped = np.flip(kernel, axis=(0, 1))

        # 设置 padding
        if padding == 'same':
            pad_h = kH // 2
            pad_w = kW // 2
        elif padding == 'valid':
            pad_h = pad_w = 0
        else:
            raise ValueError("padding must be 'same' or 'valid'")

        # Pad image
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

        out_h = H if padding == 'same' else H - kH + 1
        out_w = W if padding == 'same' else W - kW + 1
        output = np.zeros((out_h, out_w, C), dtype=np.float32)

        # 卷积（逐通道处理）
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    region = padded[i:i+kH, j:j+kW, c]
                    output[i, j, c] = np.sum(region * kernel_flipped)

        if output.shape[2] == 1:
            output = output[:, :, 0]  # 若原始是灰度图，还原为 (H, W)

        # output = np.clip(np.round(output), 0,255).astype(np.uint8)
        self.new_img = output
        return output

    def plot(self):
        # 展示输入图像
        plt.imshow(self.img[:, :, ::-1])
        plt.axis('off')
        plt.title('Input')
        plt.show(block=True)
    
        # 展示卷积核
        fig = plt.figure(figsize=(2, 1.5))
        sns.heatmap(self.k)
        plt.axis('off')
        plt.title('Kernel')
        plt.show(block=True)
        
        # 卷积结果可视化
        plt.imshow(self.new_img[:, :, ::-1])
        plt.axis('off')
        plt.title('Output')
        plt.show(block=True)
        

# 添加椒盐噪声
def add_Salt(img, pro):
    # img为输入图像
    # pro为椒盐噪声的比例
    
    # 添加黑色像素点
    noise = np.random.uniform(0, 255, img[:, :, 0].shape)
    # mask为添加噪声的掩模
    mask = noise < pro * 255
    # 扩展mask的维度
    mask = np.expand_dims(mask, axis=2)
    mask = np.repeat(mask, 3, axis=2)    
    img = img * (1 - mask)
     
    # 添加白色像素点
    mask = noise > 255 - pro * 255    
    mask = np.expand_dims(mask, axis=2)
    mask = np.repeat(mask, 3, axis=2)
    noise_img = 255 * mask + img * (1 - mask)
    
    return noise_img
    
    
def add_Gaussian(img, sigma=20, mean=0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    # 生成高斯噪声
    noise = np.random.normal(mean, sigma, lab[:, :, 0].shape)
    lab = lab.astype(float)
    # 添加高斯噪声
    lab[:, :, 0] = lab[:, :, 0] + noise
    lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
    lab = lab.astype(np.uint8)
    noise_img = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return noise_img


# 定义高斯核
def gaussian_kernel(kernel_size, sigma):
    
    kernel = np.zeros((kernel_size, kernel_size))
    # 定义中心点坐标
    center = kernel_size // 2
    s = sigma ** 2
    sum_val =  0
    # 计算每个位置的高斯核权重
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i-center, j-center
            
            kernel[i, j] = np.exp(-(x**2 + y**2)/ 2 * s)
            sum_val += kernel[i, j]
    
    kernel = kernel/sum_val
    
    return kernel


def plot_image(img, name, camp='gray'):
    plt.imshow(img, cmap=camp)
    plt.title(name)
    plt.axis('off')
    plt.show()

#归一化互相关
def CCORR(img, temp, normalize=True):
    w, h = temp.shape[::-1]
    W, H = img.shape[::-1]
    res = np.zeros((W-w+1, H-h+1))
    img = np.array(img, dtype='float')
    temp = np.array(temp, dtype='float')
    t = np.sqrt(np.sum(temp**2))
    for i in range(W-w+1):
        for j in range(H-h+1):
            res[i,j] = np.sum(temp*img[j:j+h, i:i+w])   # 对应位置乘积并累加
            # 在这里进行归一化操作
            if normalize:
                res[i,j] = res[i,j] / t / np.sqrt(np.sum(img[j:j+h, i:i+w]**2)) 
    return res


# 构建单目标匹配类
class temp_match_single():
    def __init__(self, img, temp):
        self.img = img
        self.temp = temp
        
    def match(self):
        # 输入目标图像
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # 输入模板图像
        temp = cv2.cvtColor(self.temp, cv2.COLOR_BGR2GRAY)
        w, h = temp.shape[::-1]
        
        # 计算互相关
        res = CCORR(img, temp)
        
        # 找到互相关值最大的位置
        loc = np.where(res == np.max(res))
        top_left = [int(loc[0]), int(loc[1])]
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        # 将其框出
        cv2.rectangle(self.img, top_left, bottom_right, (255,255,255), 1)
        plot_image(self.img[:, :, ::-1], 'Matching Result by CCORR')
        

# 利用快速排序算法找到数列中第k大的值
def findKth(s, k):
    return findKth_c(s, 0, len(s) - 1, k)


def findKth_c(s, low, high, k):
    m = partition(s, low, high)
    if m == len(s) - k:
        return s[m]
    elif m < len(s) - k:
        return findKth_c(s, m + 1, high, k)
    else:
        return findKth_c(s, low, m - 1, k)


def partition(s, low, high):
    pivot, j = s[low], low
    for i in range(low + 1, high + 1):
        if s[i] <= pivot:
            j += 1
            s[i], s[j] = s[j], s[i]
    s[j], s[low] = s[low], s[j]
    return j


# 构建多目标模板匹配类
class temp_match_multi():
    def __init__(self, img, temp, k=1):
        self.img = img
        self.temp = temp
        # 定义需要匹配的个数，由k间接确定阈值大小
        self.k = k
        
    def match(self):
        # 输入模板图像和目标图像
        temp = cv2.cvtColor(self.temp, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        w, h = temp.shape[::-1]
        method = eval('cv2.TM_CCORR_NORMED')
        res = cv2.matchTemplate(img, temp, method)
        temp = list(np.array(res).flatten())
        
        # 寻找res中第k大的值
        threshold = findKth(temp, self.k)
        print('设定的阈值为：', threshold)
        loc = np.where(res >= threshold)        # loc是该子图的左上角点相较于原图的位置坐标
        
        # 将找到的子图框选出
        for pt in zip(*loc[::-1]):
            cv2.rectangle(self.img, pt, (pt[0] + w, pt[1] + h), (255,255,255), 1)
            plt.imshow(self.img[:, :, ::-1]), plt.xticks([]), plt.yticks([])
        plt.show()
        

# 获得x方向的梯度幅值
def partial_x(img):
    Hi, Wi = img.shape[0], img.shape[1]
    out = np.zeros((Hi, Wi))
    
    # 这里我们将卷积核均值化
    k = np.array([[0,0,0],[1,0,-1],[0,0,0]])
    
    # 对图像进行卷积
    conv_ = conv_2d(img, k)
    out = conv_.conv()
    return out


# 获得y方向的梯度幅值
def partial_y(img):
    Hi, Wi = img.shape[0], img.shape[1]
    out = np.zeros((Hi, Wi))
    
    # 这里我们将卷积核均值化
    k = np.array([[0,1,0],[0,0,0],[0,-1,0]])
    
    # 对图像进行卷积
    conv_ = conv_2d(img, k)
    out = conv_.conv()
    return out


# 计算边缘强度以及方向
def gradient(img):
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)
    dx = partial_x(img)
    dy = partial_y(img)
    
    # 获得图像梯度
    G = np.sqrt(dx**2 + dy**2)
    
    # 获得梯度方向
    theta = np.rad2deg(np.arctan2(dy, dx))
    
    # 将梯度方向的大小调整为0-360之间
    theta %= 360

    return G, theta

# 基于图像边缘强度的非极大值抑制算法
def non_maximum_suppression(E, theta):
    H, W = E.shape[0], E.shape[1]
    
    # 将梯度方向投影到最近的45度角空间中
    theta = np.floor((theta + 22.5) / 45) * 45
    theta %= 360
    
    # 最后输出的图像梯度
    out = E.copy()
    
    for i in range(1, H-1):
        for j in range(1,W-1):
            
            # 当前像素点的角度大小，可以将其分为4类
            angle = theta[i,j]
            
            if angle == 0 or angle == 180:
                ma = max(E[i, j-1], E[i, j+1])
                
            elif angle == 45 or angle == 45 + 180:
                ma = max(E[i-1, j-1], E[i+1, j+1])
                
            elif angle == 90 or angle == 90 + 180:
                ma = max(E[i-1, j], E[i+1,j])
                
            elif angle == 135 or angle == 135 + 180:
                ma = max(E[i-1, j+1], E[i+1, j-1])
                
            else:
                print(angle)
                raise
            # ma是当前像素点相邻两个邻居点的像素梯度的最大值
            
            # 如果ma的值大于当前像素点的梯度值，则认为当前点非边缘
            # 并将该点的梯度值设置为0
            if ma > E[i,j]:
                out[i,j]=0
    return out

def double_thresholding(img, high, low):
    # 初始化强弱边缘为布尔矩阵
    strong_edges = np.zeros(img.shape, dtype=np.bool_)
    weak_edges = np.zeros(img.shape, dtype=np.bool_)
    
    # 获得输入图像大小
    a,b = img.shape
    for i in range(a):
        for j in range(b):
            # 大于Th阈值则为强边缘
            if img[i,j] > high:
                strong_edges[i,j] = 1
                
            # 小于Th阈值大于Tl阈值则为弱边缘
            elif img[i,j] <= high and img[i,j] > low:
                weak_edges[i, j] = 1

    return strong_edges, weak_edges

# 获得在(x，y)处点的邻居，3*3邻域内
def get_neighbors(x, y, H, W):
    neighbors = []
    for i in (x - 1, x, x + 1):
        for j in (y - 1, y, y + 1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == x and j == y):
                    continue
                neighbors.append((i, j))

    return neighbors

# 获得在(x，y)处点的邻居，k*k邻域内
def get_neighbors_k(x,y,H,W,k):
    neighbors = []
    for i in (x - k//2, x, x + k//2):
        for j in (y - k//2, y, y + k//2):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == x and j == y):
                    continue
                neighbors.append((i, j))

    return neighbors
    
# 判断弱边缘点是否为边缘
def link_edges(strong_edges, weak_edges):
    # 获得图的大小
    H, W = strong_edges.shape
    
    # 获得强边缘点的位置
    indices = np.stack(np.nonzero(strong_edges)).T
    
    # 初始化最终的结果，为布尔矩阵
    edges = np.zeros((H, W), dtype=np.bool_)

    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)
    
    # 将所有强边缘点的坐标组合成一个列表
    q = [(i,j) for i in range(H) for j in range(W) if strong_edges[i,j]]
    
    while q:
        
        # pop()函数返回列表中的末尾元素，并将其从列表中删除
        i, j = q.pop()
        
        # (a, b)是(i, j)的邻居
        for a, b in get_neighbors(i, j, H, W):
            
            # 如果当前点是弱边缘点且在强边缘点的邻域内，即可认为该点是边缘点
            if weak_edges[a][b]:
                # 为了避免对同一弱边缘点重复判断，便将其值设置为0
                weak_edges[a][b] = 0
                
                # 在边缘点中添加该弱边缘点
                edges[a][b] = 1
                
                # 由于该弱边缘点已经变成边缘点，因此需要将其加入q中
                # 通过将该点放进q中，在下一次迭代中可判断该点周围是否存在弱边缘点
                q.append((a,b))
                
    return edges


def canny(img, kernel_size=5, sigma=1.5, high=6, low=3):
    
    # 获取高斯滤波器
    kernel = gaussian_kernel(kernel_size, sigma)
    # 将图像进行高斯滤波
    smoothed = cv2.filter2D(img, -1, kernel)    
    # 得到图像的梯度图
    G, theta = gradient(smoothed)
    # 对梯度图进行nms
    nms = non_maximum_suppression(G, theta)
    # 获得强弱边缘信息
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    # 对弱边缘进行分类，得到最终的边缘图
    edge = link_edges(strong_edges, weak_edges)

    return edge


# harris角点检测——自己实现，阈值可自我调节
def harris(img, window=(3,3), alpha=0.04, threshold=0.01):
    H,W = img.shape[:2]
    h,w = window
    corners = np.zeros((H,W))
    
    # 计算梯度幅值
    Ix = partial_x(img)
    Iy = partial_y(img)
    
    # 得到每个像素值的二阶偏导数
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy
    
    # 遍历图像每个像素点
    thetas = np.zeros((H,W), dtype=np.float32)
    for x in range(H):
        for y in range(W):
            hessian_matrix_xy = np.zeros((2, 2), dtype=np.float32)
            # 获取当前像素的所有有效邻居点坐标
            neighbors = get_neighbors_k(x,y,H,W,k=h)

            # 的带每个邻居点的海塞矩阵，求和得到针对于(x,y)像素的求和黑塞矩阵
            for i,j in neighbors:
                hessian_matrix = np.array([
                        [Ixx[i,j], Ixy[i,j]], 
                         [Ixy[i,j], Iyy[i,j]]], dtype=np.float32)
                hessian_matrix_xy+=hessian_matrix
                      
            # 计算当前像素的角点响应函数
            theta = get_theta(hessian_matrix_xy, alpha)
            thetas[x,y] = theta
    pass# i,j
    
    max_thetas = np.max(thetas)
    
    # 阈值处理
    for x in range(H):
        for y in range(W):
            if (thetas[x,y]>threshold*max_thetas):
                corners[x,y]=1

    return corners


# 依据求和的黑塞矩阵计算角点响应函数
def get_theta(M, alpha=0.04):
    det_M = np.linalg.det(M)
    trace_M = np.trace(M)
    theta = det_M - alpha * trace_M ** 2
    
    return theta

# Harris角点检测算法
def harris_corners(src, NMS=False):
    
    # 获得输入图像的长宽
    h, w = src.shape[:2]
    
    # 将图像转为灰度图像
    gray_image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    # 初始化角点矩阵
    cornerPoint = np.zeros_like(gray_image,dtype=np.float32)
    
    # 计算图像沿横轴和纵轴的梯度
    grad = np.zeros((h, w, 2), dtype=np.float32)
    # x轴梯度
    grad[:,:,0] = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0)
    # y轴梯度
    grad[:,:,1] = cv2.Sobel(gray_image, cv2.CV_16S, 0, 1)
    
    # 计算黑塞矩阵内元素的值，此时的值是求和的结果，即M矩阵的元素
    Ixx = grad[:,:,0] ** 2
    Iyy = grad[:,:,1] ** 2
    Ixy = grad[:,:,0] * grad[:,:,1]
     
    # 计算窗口内黑塞矩阵元素的值，窗函数使用高斯函数
    Ixx = cv2.GaussianBlur(Ixx, (3, 3), sigmaX=2)
    Iyy = cv2.GaussianBlur(Iyy, (3, 3), sigmaX=2)
    Ixy = cv2.GaussianBlur(Ixy, (3, 3), sigmaX=2)
    
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            
            # 构建M矩阵
            struture_matrix = [[Ixx[i][j], Ixy[i][j]], 
                               [Ixy[i][j], Iyy[i][j]]]

            # 计算角点响应函数
            R = get_theta(struture_matrix)
            cornerPoint[i][j] = R

    # 非最大抑制
    corners = np.zeros_like(gray_image, dtype=np.float32)
    threshold = 0.01
    
    # 返回所有角点响应的最大值
    maxValue = np.max(cornerPoint)
    
    # 我们将角点响应函数的阈值设定为 threshold * maxValue
    
    for i in range(cornerPoint.shape[0]):
        for j in range(cornerPoint.shape[1]):
            
            # 如果进行NMS操作
            if NMS:
                # 当前角点响应值大于阈值，同时也是邻居周围的最大值
                if cornerPoint[i][j] > threshold * maxValue and \
                    cornerPoint[i][j] == np.max(
                    cornerPoint[max(0, i - 1):min(i + 1, h - 1), 
                    max(0, j - 1):min(j + 1, w - 1)]):
                    
                    corners[i][j] = 255
            else:
                # 当前角点响应值大于阈值
                if cornerPoint[i][j] > threshold * maxValue:
                    corners[i][j] = 255
                    
    # 返回检测到的角点
    return corners


if __name__ == "__main__":
    pass