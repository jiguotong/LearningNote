"""
@Author     :   jiguotong
@Contact    :   1776220977@qq.com
@site       :   
-----------------------------------------------
@Time       :   2024/5/14
@Description:   模型部署系统学习：torch.jit.script和torch.jit.trace两种将torch模型转为torch.jit.ScriptModule的方法对比，graph对比，输出结果对比
"""
import torch
from torch import nn
import numpy as np


class Model(torch.nn.Module):

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.conv = nn.Conv2d(3, 3, 3)
        self.conv2 = nn.Conv2d(3, 16, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(self.n):
            x = self.relu(self.conv(x))
        x = self.conv2(x)
        return x


models = [Model(2), Model(3)]
model_names = ['model_2', 'model_3']

for model, model_name in zip(models, model_names):
    dummy_input = torch.rand(1, 3, 10, 10)
    dummy_output = model(dummy_input)
    model_trace = torch.jit.trace(model, dummy_input)
    model_script = torch.jit.script(model)

    # 跟踪法与直接 torch.onnx.export(model, ...)等价
    torch.onnx.export(model_trace, dummy_input, f'{model_name}_trace.onnx')

    # 记录法必须先调用 torch.jit.sciprt
    torch.onnx.export(model_script, dummy_input, f'{model_name}_script.onnx')

    print("Done")

import onnx
import onnxruntime

input = np.random.random((1, 3, 10, 10)).astype(np.float32)

# test for script
onnx_path = "model_3_script.onnx"
onnx_model = onnx.load(onnx_path)
try:
    onnx.checker.check_model(onnx_model)
except Exception:
    print("Model incorrect")
else:
    print("Model correct")

ort_session = onnxruntime.InferenceSession(onnx_path,
                                           providers=['CUDAExecutionProvider'])
ort_inputs = {ort_session._inputs_meta[0].name: input}
ort_output = ort_session.run([ort_session._outputs_meta[0].name],
                             ort_inputs)[0]

print("script result mean: ", np.mean(ort_output))

# test for trace
onnx_path = "model_3_trace.onnx"
onnx_model = onnx.load(onnx_path)
try:
    onnx.checker.check_model(onnx_model)
except Exception:
    print("Model incorrect")
else:
    print("Model correct")
ort_session = onnxruntime.InferenceSession(onnx_path,
                                           providers=['CUDAExecutionProvider'])
ort_inputs = {ort_session._inputs_meta[0].name: input}
ort_output = ort_session.run([ort_session._outputs_meta[0].name],
                             ort_inputs)[0]

print("script result mean: ", np.mean(ort_output))
