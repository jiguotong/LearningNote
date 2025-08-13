---
title: （二）Qt学习笔记
updated: 2023-02-21T22:51:19
created: 2021-11-22T22:32:48
---

<https://doc.qt.io/qt-6/classes.html> qt帮助文档
一、下载与安装
二、Qt目录结构
1、Qt整体目录结构
![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image1.png)

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image2.png)
2、Qt 类库目录
![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image3.png)

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image4.png)
三、Qt初始程序代码分析
main.cpp
![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image5.png)
mainwindow.h
![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image6.png)
mainwindow.cpp
![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image7.png)

四、Qt控件与事件
⭐Qt 中的每个控件都由特定的类表示，每个控件类都包含一些常用的属性和方法，所有的控件类都直接或者间接继承自 QWidget 类。
⭐事件指的是应用程序和用户之间的交互过程或者操作系统与应用程序之间的交互过程。

五、信号和槽⭐（核心）
⭐信号函数：
signals下声明
⭐槽函数
⭐信号函数可以不用实现，槽函数一定要实现。
⭐一个信号可以连接多个槽，多个信号可以连接同一个槽
⭐信号可以直接相互连接
信号和槽的连接方式：
boolQObject::connect (constQObject \* sender, constchar\* signal, constQObject \* receiver, constchar\* slot) \[static\]
connect(发送者，发送信号，接收者，接受者做出的槽函数);

信号和槽的断开连接方式：
boolQObject::disconnect (constQObject \* sender, constchar\* signal, constObject \* receiver, constchar\* slot) \[static\]
disconnect(发送者，发送信号，接收者，接受者做出的槽函数);

六、widget体系概述
Qobject
Qwidget

七、事件系统
事件+事件循环（队列）+事件函数
队列处理顺序：发布事件→自发事件→发布事件
![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image8.png)

**过滤器：**使得所有事件先经过过滤器，比如对一个label安装了过滤器， 本来label是接收不到鼠标的点击事件的，这时候label就会收到鼠标事件，进而做一定的处理；再比如，对一个按钮btn安装了过滤器，本来点击此按钮的时候会触发一些处理，但是安装了过滤器之后，可以实现事件的拦截，直接返回true，表明已经处理过了，继续往下传递吧。

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image9.png)
画板小程序
-\><https://blog.csdn.net/dead_g/article/details/88891668>
八、信号和槽
信号和槽的连接方式：
boolQObject::connect (constQObject \* sender, constchar\* signal, constQObject \* receiver, constchar\* slot) \[static\]
connect(发送者，发送信号，接收者，接受者做出的槽函数);
事件一般只涉及一个对象，而信号和槽一般涉及到两个对象

九、对象树
QObject 的构造函数中会传入一个 Parent 父对象指针，children() 函数返回 QObjectList。即每一个 QObject 对象有且仅有一个父对象，但可以有很多个子对象。按照这种形式排列就会形成一个对象树的结构，最上层是父对象，下面是子对象，在再下面是孙子对象，以此类推。

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image10.png)
这种机制在 GUI 程序开发过程中是相当实用的。有一个很明显的现象就是我们会在窗口中new很多控件，但是却没有delete，因为在父控件销毁时这些子控件以及布局管理器对象会一并销毁。

十、定时器
三种实现方法：  
1、重写timerEvent
void timerEvent(QTimerEvent\* event) override;
void MyDrawBoard::timerEvent(QTimerEvent\* event)
{
qDebug("Hello!");
}
//需要在声明对象的时候开启计时
mydrawboard.startTimer(1000); //启动计时器

2、使用QTimer类
\#include \<QTimer\>
……
……
QTimer\* timer = new QTimer;
QObject::connect(timer, &QTimer::timeout, \[\]() {
qDebug("Hello QTimer!");
});
timer-\>start(1000);

3、使用Qtimer::singleshot，只会在当前语句执行后相应的时间过完之后才会触发
\#include \<QTimer\>
……
……
QTimer::singleShot(5000, \[\]() {
qDebug("Hello singleShot!");
});

十一、Qapplication
![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image11.png)

