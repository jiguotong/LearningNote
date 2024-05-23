"""
@Author     :   jiguotong
@Contact    :   1776220977@qq.com
@site       :   
-----------------------------------------------
@Time       :   2024/5/15
@Description:   模型部署系统学习：为ATen算子添加与onnx算子的映射关系
"""
import torch
import numpy as np


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.asinh(x)


from torch.onnx import register_custom_op_symbolic


# custom symbolic
def asinh_symbolic(g, input, *, out=None):
    return g.op("Asinh", input)


register_custom_op_symbolic('aten::asinh', asinh_symbolic,
                            9)  # str必须是domain::name

model = Model()
input = torch.rand(1, 3, 10, 10)
torch.onnx.export(model,
                  input,
                  'asinh.onnx',
                  input_names=['input'],
                  output_names=['output'])

import onnx
import onnxruntime

torch_output = model(input).detach().numpy()

onnx_model = onnx.load("asinh.onnx")
try:
    onnx.checker.check_model(onnx_model)
except Exception:
    print("Model incorrect")
else:
    print("Model correct")

ort_session = onnxruntime.InferenceSession("asinh.onnx",
                                           providers=['CUDAExecutionProvider'])
ort_inputs = {'input': input.numpy()}
ort_output = ort_session.run(['output'], ort_inputs)[0]
assert np.allclose(torch_output, ort_output)  # 检查torch的导出跟onnx的导出是否一致
pass
