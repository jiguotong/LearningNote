"""""
@Author     :   jiguotong
@Contact    :   1776220977@qq.com
@site       :   
-----------------------------------------------
@Time       :   2024/5/14
@Description:   模型部署系统学习：测试由C++转到python的模块函数调用
""" ""
import torch
import my_lib

a = torch.rand((3, 3))
b = torch.rand((3, 3))

c = my_lib.my_add(a, b)
print(a)
print(b)
print(c)
