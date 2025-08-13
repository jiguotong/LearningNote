"""""
@Author     :   jiguotong
@Contact    :   1776220977@qq.com
@site       :   
-----------------------------------------------
@Time       :   2025/7/11
@Description:   《动手学计算机视觉》书中实现的基于深度卷积网络的图像分类算法
"""""
import torch.optim as optim
import torch
from imutils import paths
import cv2
import os
import torch.optim
import torch.utils.data as Data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np

# 数据集
class ImageDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return (len(self.X))
    
    # 用于深度学习训练构建的数据类中，__getitem__()函数非常重要
    # __getitem__()决定着数据如何传入模型
    # 在下面的代码中，可以发现，当transforms非空时:
    # 数据将先经过transforms进行数据增强，再返回进行后续操作
    def __getitem__(self, i):
        data = self.X[i][:]

        if self.transforms:
            data = self.transforms(data)

        if self.y is not None:
            return (data, self.y[i])
        else:
            return data


# 定义训练函数
def fit(model, dataloader):

    model.train()
    
    # 初始化模型损失以及模型 top-1 错误率
    running_loss = 0.0
    running_top1_error = 0
    
    # 开始迭代
    for i, data in enumerate(dataloader):
        
        # 准备好训练的图像和标签，每次传入的数据量为batch_size
        x, y = data[0].to(device), data[1].to(device)
        # 需要在开始进行反向传播之前将梯度设置为零
        # 因为PyTorch会在随后的反向传递中累积梯度
        optimizer.zero_grad()
        
        # 将数据传入模型中，获得输出的预测标签
        outputs = model(x)
        
        # 将预测标签与真实标签计算损失
        loss = criterion(outputs, y.to(torch.int64))
        
        # 记录当前损失
        running_loss += loss.item()
        
        # 记录当前 top-1 错误率
        _, preds = torch.max(outputs.data, 1)
        running_top1_error += torch.sum(preds != y)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        

    loss = running_loss / len(dataloader.dataset)
    top1_error = running_top1_error / len(dataloader.dataset)

    print(f"Train Loss: {loss:.4f}, Train Top-1 Error: {top1_error:.4f}")

    return loss, top1_error


# 定义验证函数，可以通过验证集对模型的参数进行调整
def validate(model, dataloader):

    model.eval()
    
    # 初始化模型损失以及模型 top-1 错误率
    running_loss = 0.0
    running_top1_error = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # 流程同训练
            x, y = data[0].to(device), data[1].to(device)
            outputs = model(x)
            loss = criterion(outputs, y.to(torch.int64))

            running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            running_top1_error += torch.sum(preds != y)
            
        loss = running_loss / len(dataloader.dataset)
        top1_error = running_top1_error / len(dataloader.dataset)
        print(f'Val Loss: {loss:.4f}, Val Top-1 Error: {top1_error:.4f}')

        return loss, top1_error
    

# 定义测试函数，评估模型的效果
def test(model, dataloader):
    error = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            x, y = data[0].to(device), data[1].to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            error += torch.sum(predicted != y)
    return error, total


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
        
    # 定义训练图像增强（变换）的方法
    train_transform = transforms.Compose(
        [transforms.ToPILImage(),
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])])

    # 定义训练图像增强（变换）的方法
    val_transform = transforms.Compose(
        [transforms.ToPILImage(),
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])])  
      
    # 生成不同的类用于训练、验证以及测试
    train_data = ImageDataset(x_train, y_train, train_transform)
    val_data = ImageDataset(x_val, y_val, val_transform)
    test_data = ImageDataset(x_test, y_test, val_transform)

    BATCH_SIZE = 32

    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # 根据当前设备选择使用CPU或者GPU训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # 加载ImageNet预训练的模型
    model = torchvision.models.resnet34(pretrained='imagenet')
    num_features = model.fc.in_features
    num_classes = 102
    model.fc = nn.Linear(num_features, num_classes)
    model.to(device)

    # 多张GPU并行训练
    model = nn.DataParallel(model)

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 开始对模型进行训练

    # 设定迭代的轮次
    epochs = 25
    # 设定训练以及验证的损失与 top-1 错误率
    train_loss , train_top1_error = [], []
    val_loss , val_top1_error = [], []

    print(f"Training on {len(train_data)} examples, \
        validating on {len(val_data)} examples...")

    # 开始迭代
    for epoch in range(epochs):
        # 输出迭代信息
        print(f"\nEpoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_top1_error = fit(model, trainloader)
        val_epoch_loss, val_epoch_top1_error = validate(model, valloader)
        
        # 指标记录，也可用tensorboard进行记录
        train_loss.append(train_epoch_loss)
        train_top1_error.append(train_epoch_top1_error.cpu())
        val_loss.append(val_epoch_loss)
        val_top1_error.append(val_epoch_top1_error.cpu())
        
    # 可以保存训练的模型
    torch.save(model.state_dict(), "Resnet34_classification_model.pth")

    # 绘制 top-1 错误率曲线
    plt.figure(figsize=(10, 7))
    # 训练集的top-1 错误率曲线
    plt.plot(train_top1_error, color='green', label='train top-1 error')
    # 验证集top-1 错误率曲线
    plt.plot(val_top1_error, color='blue', label='validataion top-1 error')
    plt.xlabel('迭代次数')
    plt.ylabel('top-1 错误率')
    plt.legend()

    # 绘制损失函数曲线
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('迭代次数')
    plt.ylabel('损失函数')
    plt.legend()
    plt.show(block=True)

    error, total = test(model, testloader)
    print('Top-1 error of the network on test images: %0.3f %%' \
        % (100 * (error / total)))