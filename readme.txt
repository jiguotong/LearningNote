"""
@Author     :   jiguotong
@Contact    :   1776220977@qq.com
@site       :   
-----------------------------------------------
@Software   :   VScode
@Project    :   utils
@File       :   
@Version    :   v0.1
@Time       :   2023/9/6
@License    :   (C)Copyright    2021-2023,  jiguotong
@Reference  :
@Description:   
@Thought    :
"""

"""""
@Author     :   jiguotong
@Contact    :   1776220977@qq.com
@site       :   
-----------------------------------------------
@Time       :   2024/3/12
@Description:   Description
"""""


/*
* 函数功能描述
* @param a      参数a解释
* @param b      参数b解释
* @param c      参数c解释
* @return       返回信息解释
*/

Google开源项目风格指南——C++风格指南：
英文版：https://google.github.io/styleguide/cppguide.html
中文版：https://zh-google-styleguide.readthedocs.io/en/latest/google-cpp-styleguide/headers.html

vscode调试配置模板：  
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: debugpy Train Script",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/train.py",
      "console": "integratedTerminal",
      "justMyCode": false
      "env": {
          "PYTHONPATH": "."
      },
      "args": [
        "train",
        "--env=fasterrcnn",
        "--plot-every=100"
      ],
    }
  ]
}

养成好习惯：
1.测试程序或者脚本前，如需对数据进行修改或覆盖，务必提前做好备份，留出测试数据。
2.

