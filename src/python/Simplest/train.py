from torch.utils.data import DataLoader
from dataset import MyDataset
from net import Net
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
sys.path.append("./src/python/Simplest")

def main():
    dataset = MyDataset(rootpth='dataset')
    dataloader = DataLoader(dataset, batch_size=16,shuffle = True)

    net = Net(nclasses=3)       # 共有nclasses类，影响最后一层的输出通道数
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr =0.01)

    max_epoch = 100
    for epoch in range(max_epoch):
        print("Epoch: ",epoch)
        for batch_idx, (inputs,labels) in enumerate(dataloader):
            optimizer.zero_grad()       # 梯度归零！极度重要！
            outputs = net(inputs)
            labels = torch.squeeze(labels, 1)
            loss = criterion(outputs, labels)
            loss.backward()         # 反向传播，计算梯度   
            optimizer.step()        # 利用梯度，更新参数

        save_path = os.path.join(sys.path[0], 'checkpoints')     # sys.path[0]是当前运行脚本的目录
        torch.save(net.state_dict(),'{}/epoch_{:04d}.pth'.format(save_path, epoch))

    print("\033[32m[Info]:train finished!\033[0m")

if __name__ == '__main__':
    main()