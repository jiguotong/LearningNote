"""""
@Author     :   jiguotong
@Contact    :   1776220977@qq.com
@site       :   
-----------------------------------------------
@Time       :   2025/7/10
@Description:   《动手学计算机视觉》书中实现的无监督图像分割算法Ncut
"""""
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigs, svds
import skimage

def cal_dist_weighted_matrix(size, r, sig_X):
    h, w = size
    X_matrix = np.zeros((h*w, h*w))
    for i in range(h*w):
        for j in range(h*w):
            i_row, i_col = i // w, i % w
            j_row, j_col = j // w, j % w
            dist = np.power(i_row - j_row, 2) + \
                np.power(i_col - j_col, 2)
            
            if np.sqrt(dist) < r:    
                X_matrix[i, j] = np.exp(-dist / sig_X)
    return X_matrix
    
def set_weighted_matrix(img, sig_I=0.01, sig_X=5, r=10):
    vec_img = img.flatten()
    F_matrix = np.power(vec_img[None, :] - vec_img[:, None], 2)
    F_matrix /= sig_I
    F_matrix = np.exp(-F_matrix)
    
    X_matrix = cal_dist_weighted_matrix(img.shape, r, sig_X)
    return F_matrix * X_matrix


def n_cuts(W, image):
    n = W.shape[0]
    s_D = sparse.csr_matrix((n, n))
    d_i = W.sum(axis=0)
    for i in range(n):
        s_D[i, i] = d_i[i]
    s_W = sparse.csr_matrix(W)
    # print(s_W.shape)
    s_D_nhalf = np.sqrt(s_D).power(-1)
    # print(s_D.shape)

    L = s_D_nhalf @ (s_D - s_W) @ s_D_nhalf
    # print(L.shape)

    _, eigenvalues, eigenvectors = svds(L, which='SM')
    print(eigenvectors.shape)

    for i in range(1, 5):
        print(eigenvectors[i].shape)
        Partition = eigenvectors[i] > \
            np.sum(eigenvectors[i])/len(eigenvectors[i])
        print(Partition.shape)
        skimage.io.imshow(Partition.reshape(image.shape))
        plt.title('Ncut')
        plt.show()
        
if __name__=="__main__": 
    sample_img = cv2.imread('segmentation.jpeg', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    plt.imshow(sample_img, cmap='gray')
    plt.show(block=True)
    
    weighted_matrix = set_weighted_matrix(sample_img)
    n_cuts(weighted_matrix, sample_img)