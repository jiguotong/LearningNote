"""
@Author     :   jiguotong
@Contact    :   1776220977@qq.com
@site       :   
-----------------------------------------------
@Time       :   2024/5/14
@Description:   模型部署系统学习：部署动态输入、动态输出的onnx模型
"""
import os

import cv2
import numpy as np
import requests
import torch
import torch.onnx
from torch import nn


# define the model
class SuperResolutionNet(nn.Module):

    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.img_upsample = nn.Upsample(scale_factor=self.upscale_factor,
                                        mode='bicubic',
                                        align_corners=False)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.img_upsample(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        out = self.relu(self.conv3(x))

        return out


# Download checkpoint and test image, if failed, please download the image manually
urls = [
    'https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth',
    'https://raw.githubusercontent.com/open-mmlab/mmediting/master/tests/data/face/000001.png'
]
names = ['srcnn.pth', 'face.png']
for url, name in zip(urls, names):
    if not os.path.exists(name):
        open(name, 'wb').write(requests.get(url).content)


def init_torch_model():
    torch_model = SuperResolutionNet(upscale_factor=3)
    state_dict = torch.load('srcnn.pth')['state_dict']

    # Adapt the checkpoint
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)

    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model


model = init_torch_model()
input_img = cv2.imread('face.png').astype(np.float32)

# HWC to NCHW
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)

# Inference
torch_output = model(torch.from_numpy(input_img)).detach().numpy()

# NCHW to HWC
torch_output = np.squeeze(torch_output, 0)
torch_output = np.clip(torch_output, 0, 255)
torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)

# Show image
cv2.imwrite("face_torch.png", torch_output)

# export to onnx
x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    torch.onnx.export(model,
                      x,
                      "srcnn_dynamic.onnx",
                      opset_version=11,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={
                          'input': {
                              2: 'height',
                              3: 'width',
                          },
                          'output': {
                              2: 'height',
                              3: 'width',
                          }
                      })

import onnx
import onnxruntime

onnx_model = onnx.load("srcnn_dynamic.onnx")
try:
    onnx.checker.check_model(onnx_model)
except Exception:
    print("Model incorrect")
else:
    print("Model correct")

ort_session = onnxruntime.InferenceSession("srcnn_dynamic.onnx",
                                           providers=['CUDAExecutionProvider'])
ort_inputs = {'input': input_img}
ort_output = ort_session.run(['output'], ort_inputs)[0]

ort_output = np.squeeze(ort_output, 0)
ort_output = np.clip(ort_output, 0, 255)
ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)
# cv2.imwrite("face_ort_dynamic.png", ort_output)
print("input image width: {}, height: {}, channel: {}".format(
    input_img.shape[2], input_img.shape[3], input_img.shape[1]))
print("output image width: {}, height: {}, channel: {}".format(
    ort_output.shape[0], ort_output.shape[1], ort_output.shape[2]))


input_img = cv2.imread('face_torch.png').astype(np.float32)

# HWC to NCHW
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)

ort_inputs = {'input': input_img}
ort_output = ort_session.run(['output'], ort_inputs)[0]

ort_output = np.squeeze(ort_output, 0)
ort_output = np.clip(ort_output, 0, 255)
ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)
# cv2.imwrite("face_ort_dynamic.png", ort_output)
print("input image width: {}, height: {}, channel: {}".format(
    input_img.shape[2], input_img.shape[3], input_img.shape[1]))
print("output image width: {}, height: {}, channel: {}".format(
    ort_output.shape[0], ort_output.shape[1], ort_output.shape[2]))