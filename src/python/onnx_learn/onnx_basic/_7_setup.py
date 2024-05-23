"""""
@Author     :   jiguotong
@Contact    :   1776220977@qq.com
@site       :   
-----------------------------------------------
@Time       :   2024/5/14
@Description:   模型部署系统学习：在python环境中安装c++的模块
""" ""
from setuptools import setup
from torch.utils import cpp_extension

setup(name='my_add',
      ext_modules=[
          cpp_extension.CppExtension(name='my_lib', sources=['_7_my_add.cpp'])
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

# python _7_setup.py develop
