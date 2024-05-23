"""
@Author     :   jiguotong
@Contact    :   1776220977@qq.com
@site       :   
-----------------------------------------------
@Time       :   2024/5/14
@Description:   模型部署系统学习：利用ONNX Python API手动构造模型
@Note       :   onnx版本为1.14.1时会出错，降级到1.12.0
"""
import onnx
from onnx import helper
from onnx import TensorProto

# input and output 
a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [10, 10])
x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 10])
b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [10, 10])
output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [10, 10])

# Mul
mul = helper.make_node('Mul', ['a', 'x'], ['c'])

# Add
add = helper.make_node('Add', ['c', 'b'], ['output'])

# graph and model
graph = helper.make_graph([mul, add], 'linear_func', [a, x, b], [output])
model = helper.make_model(graph)

# save modle
onnx.checker.check_model(model)
print(model)
onnx.save(model, 'linear_func.onnx')

import onnxruntime 
import numpy as np 
 
sess = onnxruntime.InferenceSession('linear_func.onnx', providers=['CPUExecutionProvider']) 
a = np.random.rand(10, 10).astype(np.float32) 
b = np.random.rand(10, 10).astype(np.float32) 
x = np.random.rand(10, 10).astype(np.float32) 
 
output = sess.run(['output'], {'a': a, 'b': b, 'x': x})[0] 
 
assert np.allclose(output, a * x + b) 

# get onnx info
import onnx 
model = onnx.load('linear_func.onnx') 
print(model)
graph = model.graph 
node = graph.node 
input = graph.input 
output = graph.output 
print("***************Node***************\n", node)
print("***************Input***************\n",input)
print("***************Output***************\n",output)