# 一、gcc
## （一）gcc、g++
![1681889330524](image/Make与CMake学习及使用笔记/1681889330524.png)
## （二）gcc、make和cmake

gcc（GNU Compiler Collection）将源文件编译（Compile）成可执行文件或者库文件；
而当需要编译的东西很多时，需要说明先编译什么，后编译什么，这个过程称为构建（Build），常用的工具是make，对应的定义构建过程的文件为Makefile；
而编写Makefile对于大型项目又比较复杂，通过CMake就可以使用更加简洁的语法定义构建的流程，CMake定义构建过程的文件为CMakeLists.txt。
它们的大致关系如下图：
![1681459157267](image/Make与CMake学习及使用笔记/1681459157267.png)
使用流程如下：
***有一些源码*.c *.h***
***编写CMakeLists.txt，通过cmake命令进行构建----->生成Makefile***
***通过make命令进行多个文件的编译链接（使用到gcc）------>生成可执行文件***

# 二、Make

## （一）

make -j 参数加快编译效率

# 三、CMake

## （一）CMake概述

CMake是一个开源、跨平台的编译、测试和打包工具，它使用比较简单的语言描述编译、安装的过程，输出Makefile或者project文件，再去执行构建。
使用cmake一般流程为：

- 生成构建系统（buildsystem，比如make工具对应的Makefile）；
- 执行构建（比如make），生成目标文件；
- 执行测试、安装或打包。

## （二）CMake基本使用

参考链接：
https://blog.csdn.net/qq_34796146/article/details/108877159
1.文件准备 /home/tmp

```txt
.
└── test_for_cmake
    ├── CMakeLists.txt
    └── main.c
```

***main.c***

```c
#include <stdio.h>
int main(){
    printf("Hello World from Main!\n");
    return 0;
}
```

***CMakeLists.txt***

```cmake
PROJECT(HELLO)
SET(SRC_LIST main.c)
MESSAGE(STATUS "This is BINERY dir: ${HELLO_BINARY_DIR}")
MESSAGE(STATUS "This is SOURCE dir: ${HELLO_SOURCE_DIR}")

ADD_EXECUTABLE(hello ${SRC_LIST})
```

2.构建
->执行cmake
$ cmake -B build  #会将生成的所有临时文件放到build里面，包括Makefile
![1681463492776](image/Make与CMake学习及使用笔记/1681463492776.png)
->执行make
$ cd build
$ make      #会生成hello可执行文件
![1681463386264](image/Make与CMake学习及使用笔记/1681463386264.png)
![1681463535717](image/Make与CMake学习及使用笔记/1681463535717.png)
->执行可执行文件
$ ./hello

## （三）CMakeLists.txt进阶使用

参考链接：
https://blog.csdn.net/qq_34796146/article/details/108877159
https://zhuanlan.zhihu.com/p/500002865

### 1.CMakeLists.txt概述

CMakeLists.txt由命令、注释和空格组成,其中命令是不区分大小写的，符号"#"后面的内容被认为是注释。命令由命令名称、小括号和参数组成，参数之间使用空格进行间隔。
以下为具体的CMakeLists.txt：

```cmake
CMAKE_MINIMUM_REQUIRED(VERSION 3.0)                         # 需要的CMake最低版本
PROJECT(example_person)                                     # 项目名称

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")        # 如果代码需要支持C++11
add_definitions("-Wall -g")                                 # 使得想要生成的可执行文件拥有符号表，可以gdb调试

set(GOOGLE_PROTOBUF_DIR ${PROJECT_SOURCE_DIR}/protobuf)     # 设置变量
set(PROTO_PB_DIR ${PROJECT_SOURCE_DIR}/proto_pb2)
set(PROTO_BUF_DIR ${PROJECT_SOURCE_DIR}/proto_buf)

add_subdirectory(proto_pb2)                                 # 同时构建子文件夹下的CMakeLists.txt

include_directories(${PROJECT_SOURCE_DIR} ${PROTO_PB_DIR} ${PROTO_BUF_DIR}) # 规定.h头文件路径
link_directories(LD_LIBRARY_DIR)                                      # 规定.so/.a库文件路径

add_executable(${PROJECT_NAME} example_person.cpp )         # 生成可执行文件
target_link_libraries(${PROJECT_NAME} general_pb2)          # 进行链接操作
```

### 2.CMakeLists.txt常用命令详解

参考链接：
https://blog.csdn.net/songyuc/article/details/128018692

#### find_path

⭐命令：find_path(\<var> NAMES name PATHS [path1] [path2] [path])
⭐用途：搜索包含某个指定文件的路径，若在以上路径中找到name文件，则将该路径返回给变量var，否则var的值为var-NOTFOUND
⭐示例：find_path(OPENCV_INCLUDE_DIR opencv2.h /usr/include)
⭐注意事项：find_path不会递归搜索，只能在path中搜索，如果想要在path子文件夹中找到文件，可以使用PATH_SUFFIXES修饰符，例find_path(GSLINCLUDE NAMES "fred.h" PATHS /usr/include PATH_SUFFIXES opencv)

#### find_package

⭐命令：
⭐用途：查找依赖包，理想情况下，可以把一整个依赖包的头文件包含路径、库路径、库名字、版本号等情况都获取到。有两种模式，模块模式(Module mode)和配置模式(Config mode)
1）Module模式：
在该模式下，Cmake会搜索一个名为Find\<PackageName>.cmake的文件，其中\<PackageName>为待搜索包的名称。搜索路径的顺序依次是：

- 从变量CMAKE_MODULE_PATH指定的路径中进行查找 预先 set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
- 从Cmake安装路径中查找。Cmake会在其安装路径下提供很多.cmake文件，例如/usr/local/cmake/Modules/目录下（不同的系统安装目录可能不一致）

2）Config模式：
要找\<packageName>Config.cmake或\<lower-case-package-name>-config.cmake文件，查找顺序：

- \<packageName>_ROOT变量路径
- \<package_name>_DIR变量路径，有了就不用上面那个了（可以用这个，定义这个变量）

⭐示例：find_package(OpenCV REQUIRED)
⭐注意事项：

#### find_library

⭐命令：find_library(\<var> NAMES name PATHS [path1] [path2] [path])
⭐用途：在指定路径中搜索库的名字，若找到，将路径赋值给var
⭐示例：find_library(OPENCV_LIB_DIR NAMES opencv_core)或find_library(OPENCV_LIB_DIR NAMES libopencv_core.a)
⭐注意事项：

#### set

⭐命令：set(var "content")
⭐用途：将""中的内容赋值给var
⭐示例：set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
⭐注意事项：

#### MESSAGE

⭐命令：MESSAGE([\<mode>] "message text" ...)
⭐用途：在终端显示cmake过程中的信息
⭐示例：#MESSAGE( STATUS "Pangolin_LIBRARIES: " ${Pangolin_LIBRARIES})
⭐注意事项：

#### include_directories

⭐命令：
⭐用途：
⭐示例：
⭐注意事项：

#### add_executable

⭐命令：
⭐用途：
⭐示例：
⭐注意事项：

#### target_link_libraries

⭐命令：
⭐用途：
⭐示例：
⭐注意事项：

### 3.CMakeLists.txt实用案例

```CMake

```
