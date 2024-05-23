"""""
@Author     :   jiguotong
@Contact    :   1776220977@qq.com
@site       :   
-----------------------------------------------
@Time       :   2024/5/14
@Description:   模型部署系统学习：用 torch.autograd.Function 封装C++模块
""" ""
import torch 
import my_lib 


class MyAddFunction(torch.autograd.Function): 
 
    @staticmethod 
    def forward(ctx, a, b): 
        return my_lib.my_add(a, b) 
 
    @staticmethod 
    def symbolic(g, a, b): 
        two = g.op("Constant", value_t=torch.tensor([2]))   # 2 
        a = g.op('Mul', a, two)     # 2 * a
        return g.op('Add', a, b)    # 2 * a + b         # symbolic中若使用g.op(), 必须是已经注册过的onnx算子
    
my_add = MyAddFunction.apply 
 
class MyAdd(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
 
    def forward(self, a, b): 
        return my_add(a, b) 
    
model = MyAdd() 
input = torch.rand(1, 3, 10, 10) 
torch.onnx.export(model, (input, input), 'my_add.onnx', input_names=['a', 'b'], output_names=['output']) 
torch_output = model(input, input).detach().numpy() 
 
import onnxruntime 
import numpy as np 
sess = onnxruntime.InferenceSession('my_add.onnx', providers=['CUDAExecutionProvider']) 
ort_output = sess.run(None, {'a': input.numpy(), 'b': input.numpy()})[0] 
 
assert np.allclose(torch_output, ort_output) 