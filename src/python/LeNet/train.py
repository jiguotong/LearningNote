from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from LeNet import LeNet
import torch
import torch.nn as nn
import torch.optim as optim
import numpy

def main():
    data_train = MNIST('./data',
                    download=True, 
                    transform=transforms.Compose([
                        transforms.Resize((32,32)),
                        transforms.ToTensor()]))
    data_train_loader = DataLoader(data_train,batch_size=256, shuffle=True, num_workers=8)

    model = LeNet()
    model.train()
    lr =0.01        # 定义学习率
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=5e-4)      # 定义随机梯度下降优化器

    train_loss =0
    correct=0
    total=0

    for batch_idx,(inputs,targets) in enumerate(data_train_loader):
        optimizer.zero_grad()
        outputs= model(inputs)
        loss = criterion(outputs,targets)
        loss.backward()
        optimizer.step()
        
        train_loss+=loss.item()
        _, predicted =outputs.max(1)
        total+=targets.size(0)
        correct+=predicted.eq(targets).sum().item()

        print(batch_idx,len(data_train_loader),'Loss: %.3f | Acc: %.3f%%  (%d/%d)' 
                  %(train_loss/(batch_idx+1),100.*correct/total,correct,total))
        # example---------233 235 Loss: 2.258 | Acc: 18.032%(10802/59904)     %%是转义字符%

    ## 保存模型信息
    save_info={ 
        "optimizer":optimizer.state_dict(),     # 优化器的状态字典
        "model":model.state_dict()              # 模型的状态字典
    }
    save_path = "./checkpoints/model.pth"
    torch.save(save_info,save_path)
    print('Successfully save pth')


if __name__=='__main__':
    main()