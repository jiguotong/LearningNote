from Image.functions import bmp2png, process_dark_pixels, fusion_mask, static_accuracy
from File.functions import sample_files_copy, sample_files_cut, copy_matching_files


# 执行函数Image.functions.bmp2png
def main_bmp2png():
    # 设置输入和输出文件夹路径
    input_folder = 'D:\\AI\\dataset\\2D-data\\口扫机器保存图像\\分层数据\\rgb_'
    output_folder = 'D:\\AI\\dataset\\2D-data\\口扫机器保存图像\\分层数据\\rgb'
    bmp2png(input_folder, output_folder)
    print("函数执行完毕")
   

# 执行函数File.functions.sample_files_copy
def main_sample_files_copy():
    # 指定原始文件夹和目标文件夹路径
    source_folder = 'D:/AI/转移杆/转移杆png'
    target_folder = 'D:/AI/转移杆/转移杆sam'
    sample_files_copy(source_folder, target_folder, sample_radio=0.1)
    print("函数执行完毕")   
    
    
# 执行函数File.functions.sample_files_copy
def main_sample_files_cut():
    # 指定原始文件夹和目标文件夹路径
    source_folder = 'D:/AI/转移杆/愈合基台sam'
    target_folder = 'D:/AI/转移杆/愈合基台sam-1'
    sample_files_cut(source_folder, target_folder, sample_radio=0.5)
    print("函数执行完毕")   
    
# 执行函数Image.functions.process_dark_pixels
def main_process_dark_pixels():
    # 指定原始文件夹和目标文件夹路径
    source_folder = 'D:/AI/dataset/2D-data/颊侧/2_png'
    target_folder = 'D:/AI/dataset/2D-data/颊侧/2_png_00'
    process_dark_pixels(source_folder, target_folder, threhold=30)
    print("函数执行完毕")  
    

# 执行函数Image.functions.fusion_mask
def main_fusion_mask():
    # 指定原始文件夹和目标文件夹路径
    source_folder = 'D:/AI/dataset/2D-data/颊侧/2_predmask'
    mask_folder = 'D:/AI/dataset/2D-data/颊侧/2_cvmask'
    target_folder = 'D:/AI/dataset/2D-data/颊侧/2_mask'
    fusion_mask(source_folder, mask_folder, target_folder, color=(0,0,128))
    print("函数执行完毕")  
    
# 执行函数Image.functions.static_accuracy
def main_static_accuracy():
    input1_folder = 'D:/AI/dataset/2D-data/generated/train/label_png'
    input2_folder = 'D:/AI/dataset/2D-data/generated/train/predict_mask'
    txt_path = 'D:/AI/dataset/2D-data/generated/train/static.txt'
    static_accuracy(input1_folder, input2_folder, txt_path)

# 执行函数File.copy_matching_files
def main_copy_matching_files():
    source_dir = "C:/Users/Administrator/Desktop/jiace_temp/gnr_2/上颌/image_rgb"
    target_dir = "D:/AI/code/Deep_Learning_Deploy/src/ONNX/2DDeploy/x64/问题图片查漏标注"
    directory_path = "D:/AI/code/Deep_Learning_Deploy/src/ONNX/2DDeploy/x64/问题图片查漏"
    copy_matching_files(source_dir, target_dir, directory_path)
    
if __name__ == '__main__':
    main_bmp2png()
    pass
