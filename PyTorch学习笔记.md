## torch函数应用

1、torch.argmax(input: Tensor, dim: Optional[_int]=None, keepdim: _bool=False, *, out: Optional[Tensor]=None)
返回指定维度最大值的序号（索引）。实际就是以该维度为标准划分为各个队伍，不个队伍的同一位置进行pk，选出最大的值所在的索引。
若x.shape(2,3,4),y=torch.argmax(x,dim=0),则y.shape(3,4)
若x.shape(2,3,4),y=torch.argmax(x,dim=1),则y.shape(2,4)
若x.shape(2,3,4),y=torch.argmax(x,dim=2),则y.shape(2,3)

```python
import torch

x = torch.rand(3,2,3)
print(x)
y0 = torch.argmax(x, dim=0)
print(y0)
y1 = torch.argmax(x, dim=1)
print(y1)
y2 = torch.argmax(x, dim=2)
print(y2)
```

![1685685696830](image/视频分割笔记/1685685696830.png)

2、DataLoader类
数据加载器，结合了数据集和取样器，并且可以提供多个线程处理数据集。
在训练模型时使用到此函数，用来把训练数据分成多个小组 ，此函数每次抛出一组数据。直至把所有的数据都抛出。就是做一个数据的初始化。
![1685699996707](image/视频分割笔记/1685699996707.png)

```python
"""
    批训练，把数据变成一小批一小批数据进行训练。
    DataLoader就是用来包装所使用的数据，每次抛出一批数据
"""
import torch
import torch.utils.data as Data

BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)
# 把数据放在数据库中
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)


def show_batch():
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            # training
            print("steop:{}, batch_x:{}, batch_y:{}".format(step, batch_x, batch_y))


if __name__ == '__main__':
    show_batch()
```

3、torch.cat()和torch.stack()用法
都是对张量进行拼接操作
torch.cat(): 用于连接两个相同大小的张量，不扩展维度
torch.stack(): 用于连接两个相同大小的张量，并扩展维度

```python
import torch
x=torch.zeros(2,3)
y=torch.ones(2,3)
print(x)
print(y)
print('-------------------------torch.cat---------------------------')
# torch.cat
a=torch.cat([x,y],dim=0)
print(a.shape)
print(a)

print('-------------------------troch.stack---------------------------')
# torch.stack
b = torch.stack((x,y),dim=0)
print(b.shape)
print(b)

print('--------------------------end--------------------------')
```

结果如图：![1686561179317](image/视频分割笔记/1686561179317.png)
https://github.com/suhwan-cho/TMO

4、.item()和a.eq(b)
.item()作用是取出单元素张量的元素值并返回该值，保持原元素类型不变。
例如loss是tensor(0.569),若total=loss.item(),则total为0.569，仍然是float类型
.eq()是用于比较的包装器，返回同等维度的True/False合集

5、批量标准化（Batch Normalization）
不仅仅对输入层做标准化处理，还要对 每一中间层的输入(激活函数前) 做标准化处理，使得输出服从均值为 0，方差为 1 的正态分布，从而避免内部协变量偏移的问题。
```python
nn.BatchNorm2d(in_channels)
```
**优点**：
+ 首先，通过对输入和中间网络层的输出进行标准化处理后，减少了内部神经元分布的改变，使降低了不同样本间值域的差异性，得大部分的数据都其处在非饱和区域，从而保证了梯度能够很好的回传，避免了梯度消失和梯度爆炸
+ 其次，通过减少梯度对参数或其初始值尺度的依赖性，使得我们可以使用较大的学习速率对网络进行训练，从而加速网络的收敛
+ 最后，由于在训练的过程中批量标准化所用到的均值和方差是在一小批样本(mini-batch)上计算的，而不是在整个数据集上，所以均值和方差会有一些小噪声产生，同时缩放过程由于用到了含噪声的标准化后的值，所以也会有一点噪声产生，这迫使后面的神经元单元不过分依赖前面的神经元单元。所以，它也可以看作是一种正则化手段，提高了网络的泛化能力，使得我们可以减少或者取消 Dropout，优化网络结构

## 2.9 PyTorch中的损失函数
（1）交叉熵损失函数
+ BCE(binary_cross_encrypt,二值交叉熵)
F.binary_cross_entropy_with_logits:该损失函数已经内部自带了计算logit的操作，无需在传入给这个loss函数之前手动使用sigmoid/softmax将之前网络的输入映射到[0,1]之间
```python
from torch.nn import functional as F
bce_loss = F.binary_cross_entropy(F.sigmoid(input), target)
bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boudary_targets_pyramid)
```

+ dice_loss:Dice Loss常用于语义分割问题中，可以缓解样本中前景背景（面积）不平衡带来的消极影响.
```python
def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()
```
（2）ohemCELoss
图像分割领域使用的损失函数，其中 Online hard example mining 的意思是，在训练过程中关注 hard example，对其施加更高权重的一种训练策略。cross-entropy loss 就是普通的交叉熵损失函数。
```python
class OhemCELoss(nn.Module):
    """
    Online hard example mining cross-entropy loss:在线难样本挖掘
    if loss[self.n_min] > self.thresh: 最少考虑 n_min 个损失最大的 pixel，
    如果前 n_min 个损失中最小的那个的损失仍然大于设定的阈值，
    那么取实际所有大于该阈值的元素计算损失:loss=loss[loss>thresh]。
    否则，计算前 n_min 个损失:loss = loss[:self.n_min]
    """
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()     # 将输入的概率 转换为loss值
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')   #交叉熵
 
    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)     # 排序
        if loss[self.n_min] > self.thresh:       # 当loss大于阈值(由输入概率转换成loss阈值)的像素数量比n_min多时，取所以大于阈值的loss值
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)
```

## 2.10 PyTorch中数据的输入和预处理

1、torch.utils.data.DataLoader()

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, work_init_fn=None)
```

2、python的类中函数__getitem__的作用及使用方法
__getitem__方法的作用是，可以将类中的数据像数组一样读出，操作符为[]，使用索引访问元素

```python
class Test():
    def __init__(self):
        self.a=[1,2,3,4,5]
    def __getitem__(self,idx):
        return(self.a[idx])
data=Test()
print(data[2])

--------------
>>>3

class Test():
    def __init__(self):
        self.a=[1,2,3,4,5]
    def __getitem__(self,idx):
        return(self.a)
data=Test()
print(data[2])

--------------
>>>[1,2,3,4,5]
```

3、python的内置__getitem__方法与PyTorch的Dataloader结合使用：
https://blog.csdn.net/virus111222/article/details/128210099
过程剖析：
（1）定义一个数据集dataset，要重写__len__函数和__getitem__函数，因为后面要用遍历dataloader（需要用索引获取数组）
（2）定义一个dataloader，用来封装该数据集，并且指定batch_size以及是否打乱顺序等
（3）遍历dataloader时，比如batch_size等于4，那么dataloader就找4个下标（如果没打乱 就是 0 1 2 3 / 4 5 6 7，如果打乱，就会随机下标），去dataset里面通过这4个下标idx从__getitem__获取相应的数据(也可以选择不使用下标)

```python
import torch
import numpy as np
from torch.utils.data import Dataset
 
# 创建MyDataset类
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.data = torch.from_numpy(x).float()
        self.label = torch.LongTensor(y)
 
    def __getitem__(self, idx):
        print("当前idx:{}".format(idx))
        return self.data[idx], self.label[idx], idx
 
    def __len__(self):
        return len(self.data)
 
Train_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
Train_label = np.array([10, 11, 12, 13])
TrainDataset = MyDataset(Train_data, Train_label) # 创建实例对象
print('len:', len(TrainDataset))
 
# 创建DataLoader
loader = torch.utils.data.DataLoader(
    dataset=TrainDataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=False)
 
# 按batchsize打印数据
for batch_idx, (data, label, index) in enumerate(loader):
    print('batch_idx:',batch_idx, '\ndata:',data, '\nlabel:',label, '\nindex:',index)
    print('---------')
```

输出结果

```python
len: 4
当前idx:0
当前idx:3
batch_idx: 0 
data: tensor([[ 1.,  2.,  3.],
        [10., 11., 12.]]) 
label: tensor([10, 13]) 
index: tensor([0, 3])
---------
当前idx:1
当前idx:2
batch_idx: 1 
data: tensor([[4., 5., 6.],
        [7., 8., 9.]]) 
label: tensor([11, 12]) 
index: tensor([1, 2])
---------
```

## 2.11 PyTorch模型的保存和加载

```python
torch.save(obj, f, pickle_module=pickle, pickle_protocol=2)
torch.load(f, map_location=None, pickle_module=pickle, **pickle_load_args)
```

**obj**：可以被序列化的对象，包括模型和张量等。
f：存储文件的路径
**pickle_module**：序列化的库，默认是pickle
**pickle_protocol**：**对象转为字符串的规范协议
**map_location**：cpu或者cpu，以此支持保存/加载模型时的设备不一样，例如保存的时候是gpu，但是加载的时候只有cpu，此时设map_location='cpu'，若是gpu下，map_location='cuda:0'
**pickle_load_args**：存放参数，指定传给pickle_module.load的参数
使用案例如下：

```python
torch.save(self.model.state_dict(), "checkpoints/TMO.pth")
torch.load("checkpoints/TMO.pth",map_location='cpu')
```

## 2.12 PyTorch的分布式训练

单机多卡及常见问题：https://blog.csdn.net/u013531940/article/details/127858330
多机多卡的基本概念：https://blog.csdn.net/a545454669/article/details/128772522

## 3.1 常见网络架构及解析

### 3.1.1 ResNet
https://blog.csdn.net/qq_39770163/article/details/126169080 
### 3.1.5 BiseNet

https://blog.csdn.net/rainforestgreen/article/details/85157989
https://blog.csdn.net/lx_ros/article/details/126515733
http://t.csdn.cn/Rv60I
