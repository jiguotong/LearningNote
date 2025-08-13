1、安装包制作软件

inno setup

2、U盘占用，无法弹出
 磁盘管理->右键脱机->右键联机->弹出U盘


3、setlocale(LC_ALL, "chs");//
能解决很多情况下的中文编码问题！！！


4、实现两台同一局域网下的两台电脑共享文件夹
https://jingyan.baidu.com/article/bad08e1e300b3509c8512131.html

5、C++是如何调用C接口的
https://zhuanlan.zhihu.com/p/430687729

6、计算机书籍大全
https://github.com/XiangLinPro/IT_book

7、在线加密解密工具大全
http://tool.chacuo.net/cryptaes

AES算法实现C++
https://blog.csdn.net/witto_sdy/article/details/83375999

8、打开点云各种格式的工具：
Uedit64 一个文本编辑器


c++理解复杂的数组声明
![1754476225726](image/from_onenote/1754476225726.png)

遍历文件夹中的文件
https://blog.csdn.net/qq_43612802/article/details/108959366 

string转double、int方法
https://blog.csdn.net/qq_32273417/article/details/88318161 C++ 中string转double、int的方法
https://www.cnblogs.com/carsonzhu/p/5859552.html C++中string用多个分隔符分割的方法

常见排序方法汇总
https://www.cnblogs.com/flyingdreams/p/11161157.html  常见排序

vector跟list的区别
https://www.cnblogs.com/shijingjing07/p/5587719.html vector 跟list的区别

malloc/free与new/delete的区别
1、本质区别
malloc/free是C/C++语言的标准库函数，new/delete是C++的运算符。
对于用户自定义的对象而言，用maloc/free无法满足动态管理对象的要求。对象在创建的同时要自动执行构造函数，对象在消亡之前要自动执行析构函数。由于malloc/free是库函数而不是运算符，不在编译器控制权限之内，不能够把执行构造函数和析构函数的任务强加于malloc/free。因此C++需要一个能完成动态内存分配和初始化工作的运算符new，以及一个能完成清理与释放内存工作的运算符delete。
2、联系
既然new/delete的功能完全覆盖了malloc/free，为什么C++还保留malloc/free呢？因为C++程序经常要调用C函数，而C程序只能用malloc/free管理动态内存。如果用free释放“new创建的动态对象”，那么该对象因无法执行析构函数而可能导致程序出错。如果用delete释放“malloc申请的动态内存”，理论上讲程序不会出错，但是该程序的可读性很差。所以new/delete、malloc/free必须配对使用。

C1451
https://social.msdn.microsoft.com/Forums/en-US/c70af811-ae9c-497d-a5db-c93ae499fb9f/c-amp-fatal-error?forum=parallelcppnative
 
 

一些计算机书籍，密码是4321： https://github.com/imarvinle/awesome-cs-books  