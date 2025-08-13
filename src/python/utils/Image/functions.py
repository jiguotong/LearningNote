# -*- coding: UTF-8 -*-
"""
@Author     :   jiguotong
@Contact    :   1776220977@qq.com
@site       :   
-----------------------------------------------
@Time       :   2024/7/22
@Description:   本文件内含关于图像处理的若干函数，随用随加
"""


import os
from PIL import Image
from tqdm import tqdm
import numpy as np

def bmp2png(input_folder, output_folder):
    """给定一个根目录，将目录内所有.bmp图像转为.png图像"""
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有文件
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('.bmp'):
            # 打开bmp图像
            img = Image.open(os.path.join(input_folder, filename))
            
            # 将bmp图像保存为png格式
            img.save(os.path.join(output_folder, filename.replace('.bmp', '.png')))

    print('转换完成！')


def process_dark_pixels(input_folder, output_folder, threhold=20):
    """给定一个根目录(内部全是图像)，将目录内所有图像做处理，具体处理是将一张图像中所有rgb值低于某个阈值的像素位置设置mask为0，其余为1"""
        # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有文件
    for filename in tqdm(os.listdir(input_folder)):
        # 打开bmp图像
        img = Image.open(os.path.join(input_folder, filename))
        
        # 获取图像的宽度和高度
        width, height = img.size
        
        # 创建一个新的8位灰度图像，大小为100x100
        mask = Image.new('L', (width, height), color=255)
        
        # 遍历每个像素
        for x in range(width):
            for y in range(height):
                r, g, b = img.getpixel((x, y))
                
                # 如果r、g、b均低于30，则将对应位置的mask赋为0
                if r < 30 and g < 30 and b < 30:
                    mask.putpixel((x, y), 0)

        # 将bmp图像保存为png格式
        mask.save(os.path.join(output_folder, filename))
        
        
    print('处理完成！')
    
    
def fusion_mask(input_folder, mask_folder, output_folder, color=(0, 0, 0)):
    """给定两个目录，一个是rgb原图，一个是L8位图，找到mask中值为0的像素位置，将rgb图中的该位置的rgb赋为指定颜色"""
        # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历文件夹中的所有文件
    for filename in tqdm(os.listdir(input_folder)):    
        src_img = Image.open(os.path.join(input_folder, filename))
        mask_img = Image.open(os.path.join(mask_folder, filename))
        # 获取图像的宽度和高度
        width, height = src_img.size
        
        # 遍历每个像素
        for x in range(width):
            for y in range(height):
                mask = mask_img.getpixel((x, y))
                # 如果r、g、b均低于30，则将对应位置的mask赋为0
                if mask==0:
                    src_img.putpixel((x, y), color)

        # 将bmp图像保存为png格式
        src_img.save(os.path.join(output_folder, filename))
    
    print('处理完成！')
        

def static_accuracy(input1_folder, input2_folder, output_txt):
    """统计两个8位图像文件夹下每一对同名图像的像素相同比例"""
    # 1.分别获取两个文件夹内的文件名list，并判断是否一致
    
    files1 = os.listdir(input1_folder)
    files2 = os.listdir(input2_folder)
    
    files1_set = set(files1)
    files2_set = set(files2)
    
    # 如果没有共同的文件，返回
    if tuple(files1_set) != tuple(files2_set):
        print("两个文件夹内容不一致")
        return
    del files1_set, files2_set
    
    # 2. 遍历读取list，读取图像，转为数组
    results = []
    
    for index, filename in enumerate(files1):
        img1_path = os.path.join(input1_folder, filename)
        img2_path = os.path.join(input2_folder, filename)
        
        # 读取图像并转换为数组
        img1 = np.array(Image.open(img1_path))
        img2 = np.array(Image.open(img2_path))
        
        # 3. 将同名图像统计相同像素的个数，统计相同率
        if img1.shape != img2.shape:
            print(f"图像尺寸不一致: {filename}")
            continue
        
        # 统计相同像素
        equal_pixels = np.sum(img1 == img2)
        total_pixels = img1.size  # 或者 img1.shape[0] * img1.shape[1]
        
        similarity_ratio = equal_pixels / total_pixels
        
        # 4. 将结果添加到结果列表
        results.append((filename, similarity_ratio, index+1))
    
    # 按照相同率由低到高排序
    results.sort(key=lambda x: x[1])
    
    # 将结果输出到给定的txt文件中
    with open(output_txt, 'w') as f:
        for filename, ratio, index in results:
            f.write(f"{filename}: {ratio:.4f}: {index}\n")
    
    print(f"结果已写入到 {output_txt}")
