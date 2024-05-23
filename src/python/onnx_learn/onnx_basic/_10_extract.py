"""
@Author     :   jiguotong
@Contact    :   1776220977@qq.com
@site       :   
-----------------------------------------------
@Time       :   2024/5/14
@Description:   模型部署系统学习：从onnx模型中提取子模型
@Note       :   onnx版本为1.14.1时会出错，降级到1.12.0
"""

import torch 
 
class Model(torch.nn.Module): 
 
    def __init__(self): 
        super().__init__() 
        self.convs1 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3), 
                                          torch.nn.Conv2d(3, 3, 3), 
                                          torch.nn.Conv2d(3, 3, 3)) 
        self.convs2 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3), 
                                          torch.nn.Conv2d(3, 3, 3)) 
        self.convs3 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3), 
                                          torch.nn.Conv2d(3, 3, 3)) 
        self.convs4 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3), 
                                          torch.nn.Conv2d(3, 3, 3), 
                                          torch.nn.Conv2d(3, 3, 3)) 
    def forward(self, x): 
        x = self.convs1(x) 
        x1 = self.convs2(x) 
        x2 = self.convs3(x) 
        x = x1 + x2 
        x = self.convs4(x) 
        return x 
 
model = Model() 
input = torch.randn(1, 3, 20, 20) 

torch.onnx.export(model,
                  input,
                  'whole_model.onnx',
                  input_names=['input'],
                  output_names=['output'],
                  opset_version=11
)

import onnx  
model = onnx.utils.extract_model('whole_model.onnx', 'partial_model.onnx', ['/convs1/convs1.1/Conv_output_0'], ['/Add_output_0']) 

pass
print("Done")

"""子模型提取的实现原理：
新建一个模型，把给定的输入和输出填入。
之后把图的所有有向边反向，从输出边开始遍历节点，碰到输入边则停止，把这样遍历得到的节点做为子模型的节点。
"""