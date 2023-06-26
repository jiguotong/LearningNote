from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from LeNet import LeNet
import torch
import torch.nn as nn
import numpy

def main():
    data_test = MNIST('./data',
                    train=False, 
                    transform=transforms.Compose([
                        transforms.Resize((32,32)),
                        transforms.ToTensor()]))
    
    data_test_loader = DataLoader(data_test,batch_size=1024,shuffle=False,num_workers=8)

    ## 载入模型
    model = LeNet()
    model_path = "./checkpoints/model.pth"
    save_info = torch.load(model_path)
    model.load_state_dict(save_info["model"])
    model.eval()                            # 切换到测试状态

    criterion = nn.CrossEntropyLoss()       # 定义损失函数

    test_loss = 0
    correct = 0
    total=0
    with torch.no_grad():       # 关闭计算图
        for batch_idx, (inputs,targets) in enumerate(data_test_loader):
            outputs = model(inputs)
            loss = criterion(outputs,targets)       # nn.CrossEntropyLoss()内置了softmax，因此不需要自己用softmax进行归一化

            test_loss+=loss.item()
            _, predicted = outputs.max(1)
            total+=targets.size(0)
            correct+= predicted.eq(targets).sum().item()

            print(batch_idx,len(data_test_loader),'Loss: %.3f | Acc: %.3f%%  (%d/%d)' 
                  %(test_loss/(batch_idx+1),100.*correct/total,correct,total))

if __name__=='__main__':
    main()