# -*- coding: UTF-8 -*-
"""
@Author     :   jiguotong
@Contact    :   1776220977@qq.com
@site       :   
-----------------------------------------------
@Time       :   2024/7/22
@Description:   本文件内含关于文件处理的若干函数，随用随加
"""

import os
import random
import shutil
from tqdm import tqdm

def sample_files_copy(source_folder, target_folder, sample_radio=0.1):
    """给定一个根目录，将目录内所有文件按照名称，以比例的形式完成采样并复制至指定文件夹"""
    # 获取文件夹内的所有文件
    files = os.listdir(source_folder)

    # 对文件进行排序
    files.sort()

    # 创建目标文件夹
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 设置步长
    step = 1/sample_radio
    
    # 复制抽样到的文件到目标文件夹
    for i, file in tqdm(enumerate(files)):
        if i % step == 0:
            source_file = os.path.join(source_folder, file)
            target_file = os.path.join(target_folder, file)
            shutil.copy(source_file, target_file)

    print("抽样并复制完成")


def sample_files_cut(source_folder, target_folder, sample_radio=0.5):
    """给定一个根目录，将目录内所有文件按照名称，以比例的形式完成采样并剪切至指定文件夹"""
    # 获取文件夹内的所有文件
    files = os.listdir(source_folder)

    # 对文件进行排序
    files.sort()

    # 创建目标文件夹
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 设置步长
    step = 1/sample_radio
    
    # 复制抽样到的文件到目标文件夹
    for i, file in tqdm(enumerate(files)):
        if i % step == 0:
            source_file = os.path.join(source_folder, file)
            target_file = os.path.join(target_folder, file)
            shutil.move(source_file, target_file)

    print("抽样并剪切完成")
    
    
def create_dirs(root_path):
    """给定一个根目录，遍历查找每一个文件，给每一个文件建一个同名父文件夹，并将该文件放到父文件夹中"""
    pass
    for root, directories, files in os.walk(root_path):
        print(f"当前目录: {root}")
        print("子目录:")
        for directory in directories:
            print(os.path.join(root, directory))
        print("文件:")
        for file in files:
            print(os.path.join(root, file))
            file_name = os.path.splitext(file)[0]
            if not os.path.exists(os.path.join(root, file_name)):
                new_dir = os.path.join(root, file_name)
                os.mkdir(new_dir)
                shutil.move(os.path.join(root, file), os.path.join(new_dir, file))
        print()
        pass


def generate_file_tree(path: str, depth: int = 0, site=None):
    """
    递归打印文件目录树状图（使用局部变量）

    :param path: 根目录路径
    :param depth: 根目录、文件所在的层级号
    :param site: 存储出现转折的层级号
    :return: None
    """
    if site is None:
        site = []
    void_num = 0
    filenames_list = os.listdir(path)

    for item in filenames_list:
        string_list = ["│   " for _ in range(depth - void_num - len(site))]
        for s in site:
            string_list.insert(s, "    ")

        if item != filenames_list[-1]:
            string_list.append("├── ")
        else:
            # 本级目录最后一个文件：转折处
            string_list.append("└── ")
            void_num += 1
            # 添加当前已出现转折的层级数
            site.append(depth)
        print("".join(string_list) + item)

        new_item = path + '/' + item
        if os.path.isdir(new_item):
            generate_file_tree(new_item, depth + 1, site)
        if item == filenames_list[-1]:
            void_num -= 1
            # 移除当前已出现转折的层级数
            site.pop()


def copy_matching_files(source_dir, target_dir, directory_path):
    try:
        # 获取文件夹下的所有文件名称
        files = os.listdir(directory_path)
        file_names = [f for f in files if os.path.isfile(os.path.join(directory_path, f))]
    except Exception as e:
        print(f"Error reading directory {directory_path}: {e}")
        return []
    
    # 确保目标文件夹存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 读取参考文件夹中的文件名
    reference_files = file_names
    
    for file_name in reference_files:
        source_file_path = os.path.join(source_dir, file_name)
        target_file_path = os.path.join(target_dir, file_name)
        
        # 检查源文件是否存在，并拷贝
        if os.path.isfile(source_file_path):
            try:
                shutil.copy(source_file_path, target_file_path)
                print(f"Copied: {file_name} to {target_dir}")
            except Exception as e:
                print(f"Error copying {file_name}: {e}")
        else:
            print(f"File not found in source directory: {file_name}")
