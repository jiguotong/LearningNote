"""
@Author     :   jiguotong
@Contact    :   1776220977@qq.com
@site       :   
-----------------------------------------------
@Time       :   2024/5/15
@Description:   模型部署系统学习：为TorchScript算子添加与onnx算子的映射关系
"""
import torch 
import torchvision 
import numpy as np


class Model(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.conv1 = torch.nn.Conv2d(3, 18, 3) 
        self.conv2 = torchvision.ops.DeformConv2d(3, 3, 3) 
 
    def forward(self, x): 
        return self.conv2(x, self.conv1(x)) 

from torch.onnx import register_custom_op_symbolic 
from torch.onnx.symbolic_helper import parse_args 
 
@parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "none") 
def symbolic(g,  
        input, 
        weight, 
        offset, 
        mask, 
        bias, 
        stride_h, stride_w, 
        pad_h, pad_w, 
        dil_h, dil_w, 
        n_weight_grps, 
        n_offset_grps, 
        use_mask): 
    return g.op("custom::deform_conv2d", input, offset)     # custom::deform_conv2d是自定义的onnx算子名称
 
register_custom_op_symbolic("torchvision::deform_conv2d", symbolic, 9)      # 

model = Model()
input = torch.rand(1, 3, 10, 10)
torch.onnx.export(model,
                  input,
                  'dcn.onnx',
                  opset_version=12,
                  input_names=['input'],
                  output_names=['output'])

# import onnx
# import onnxruntime

# torch_output = model(input).detach().numpy()

# onnx_model = onnx.load("dcn.onnx")
# try:
#     onnx.checker.check_model(onnx_model)
# except Exception:
#     print("Model incorrect")
# else:
#     print("Model correct")

# ort_session = onnxruntime.InferenceSession("dcn.onnx",
#                                            providers=['CUDAExecutionProvider'])
# ort_inputs = {'input': input.numpy()}
# ort_output = ort_session.run(['output'], ort_inputs)[0]
# assert np.allclose(torch_output, ort_output)  # 检查torch的导出跟onnx的导出是否一致
# pass
