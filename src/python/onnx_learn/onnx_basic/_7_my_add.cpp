/*
@Author     :   jiguotong
@Contact    :   1776220977@qq.com
@site       :   
-----------------------------------------------
@Time       :   2024/5/15
@Description:   模型部署系统学习：定义C++函数，开放为python接口
*/
#include <pybind11/pybind11.h>
#include <torch/torch.h>

torch::Tensor my_add(torch::Tensor a, torch::Tensor b) 
{ 
    return 2 * a + b; 
} 

PYBIND11_MODULE(my_lib, m) 
{ 
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("my_add", my_add); 
} 