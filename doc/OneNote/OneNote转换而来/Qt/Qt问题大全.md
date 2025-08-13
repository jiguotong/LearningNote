---
title: Qt问题大全
updated: 2023-04-28T14:44:23
created: 2021-11-14T15:55:14
---

1、**解决Qt中文乱码问题：**
在pro文件中添加如下代码：
msvc {
QMAKE_CFLAGS += /utf-8

QMAKE_CXXFLAGS += /utf-8
}

**2、在Qt中添加背景音乐的方法(仅.wav格式)**
\#include \<QSound\>

QSound\* player = new QSound("./res/music.wav");//设置背景音乐
player-\>play();

**3、除了主界面新增其他页面UI的方法**
①新增
![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image1.png)

②修改对应的头文件以及源文件

**4、样式表的使用**
①Designer界面下：右键控件或者窗口，改变样式表，添加新的样式，对A改变了样式，则在A内所有的控件都会遵循这个样式，除非写了大括号{}限定。
②写代码的形式：
// 对单个控件;
ui.pushButton-\>setStyleSheet("QPushButton{border-radius:5px;background:rgb(150, 190, 60);color:red;font-size:15px;}")
// 对整个界面（包括界面上所有的控件）
this-\>setStyleSheet("QPushButton{border-radius:5px;background:rgb(150, 190, 60);color:red;font-size:15px; \\
QToolButton{border-radius:5px;background:rgb(34, 231, 131);color:brown;font-size:15px;}")
③写进文件中
![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image2.png)
//用以下代码读出
void loadStyleSheet(const QString &sheetName)
{
QFile file(sheetName);
file.open(QFile::ReadOnly);
if (file.isOpen())
{
QString styleSheet = this-\>styleSheet();
styleSheet += QLatin1String(file.readAll());
this-\>setStyleSheet(styleSheet);
}
}

**5、QT5.11编译出现undefined reference to \`\_imp\_\_\_ZN12QApplicationC1ERiPPci’**
-\><https://blog.csdn.net/lsfreeing/article/details/85859050>
确保qt库与编译器的一致性，如mingw和msvc对应的库是不同的。典型的错误如下，用了msvc的qt库，但用错了编译器

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image3.png)

**6、QtCreator调试时出现 Unable to create a debugging engine**
-\><https://blog.csdn.net/weixin_48267104/article/details/113508984>

**7、Qt 为.h和.cpp文件添加ui文件**
假设在工程中已经有了一个纯类A的头文件a.h和源文件a.cpp，现在想给这个纯类文件添加UI,可以通过以下操作来实现：

给工程添加一个和类同名的UI文件a.ui;
在a.cpp中添加UI的头文件，头文件的名字应该形如ui_xxx.h，但在添加时会发现，索引不到a.ui对应的头文件ui_a.h;这时需要先编译一下，再去添加头文件时就可以成功索引到UI文件的头文件了;
**在构造函数的函数名后加上ui(new ui::a).**

8、Qt解决页面中文乱码以及qDebug中文乱码问题
使用中文的地方都用QStringLiteral来改写，
setWindowTitle(QStringLiteral("QWidget右键菜单栏事件"));

9、使窗口在桌面中居中
Demo w;
/\*\*2. center mainwindow\*/
const QRect availableGeometry = QApplication::desktop()-\>availableGeometry(&w);
w.resize(availableGeometry.width() / 2, (availableGeometry.height() \* 2) / 3);
w.move((availableGeometry.width() - w.width()) / 2, (availableGeometry.height() - w.height()) / 2);

# 10、Qt Creator转VS2017(VS2019)遇到‘常量中有换行符‘
解决方法：
项目 -\> 属性-\>常规,将字符集设置为多字节字符集
在C/C++ =\> 命令行中最后添加/utf-8

11、Qt中文件的编码格式问题
（1）在读取文本文件时，先用QByteArray进行读取， 然后判断根据字节规律判断原始文件是什么类型(utf-8还是ANSI)，之后转为QString，并记录下当前打开的文件的编码类型；
判断函数如下：
QString GetCorrectUnicode(const QByteArray& text, QString& code)
{
QTextCodec::ConverterState state;
QTextCodec\* codec = QTextCodec::codecForName("UTF-8");
QString strtext = codec-\>toUnicode(text.constData(), text.size(), &state);
if (state.invalidChars \> 0){
code = "GBK";
strtext = QTextCodec::codecForName("GBK")-\>toUnicode(text);
}
else{
code = "UTF-8";
strtext = text;
}
return strtext;
}
（2）在写入文本文件时，使用QTextStream进行输出，并设置想要输出的编码类型(如果不设置，默认为ANSI格式），设置语句为out.setCodec("utf-8")；

12、通过 Visual Studio 打不开 ui 文件的问题
[(4条消息) 解决通过 Visual Studio 打不开 ui 文件的问题_ui文件用什么程序打开_tianyvHon的博客-CSDN博客](https://blog.csdn.net/weixin_44916154/article/details/123609499)

13、关于多线程的信号和槽的传递使用自定义类型的问题
在多线程编程时，往往需要跨线程传递参数，这时候会用到信号和槽进行传递，但是传递的参数不一定是Qt所能识别的基本类型，可能是一些自定义的类，这时候传参就需要先将参数注册，让Qt能识别此参数，就会使用qRegisterMetaType\<类型名\>("类型名")；进行注册。但是在单线程的信号和槽传递时则无需考虑这些。
例如：
enum NameType{
JIGUOTONG,
ZHANGYU
}
Q_DECLARE_METATYPE(NameType); // 声明一下自定义的类型

qRegisterMetaType\<OperateType\>(); // 在connect之前需要注册一下
qRegisterMetaType\<OperateType\>("OperateType&");

