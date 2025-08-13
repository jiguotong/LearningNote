---
title: ——QWidget
updated: 2023-03-20T20:32:00
created: 2023-02-21T22:48:54
---

**学习网址：https://www.bilibili.com/video/BV1a44y1d7Qj/?spm_id_from=333.788&vd_source=1d204308936e108c95b2ecb8fcdbd781**

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image1.png)

**Qwidget 01创建**

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image2.png)

- 当一个控件没有父控件时，会被包装成一个顶层窗口(有标题栏、等等)

- 可以通过构造函数中的第二个参数来设置控件属性，例如Qt::Widget、Qt::Window、Qt::Dialog等等
**QWidget 02坐标系、大小、位置**

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image3.png)

**QWidget 03坐标系、大小、位置**

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image4.png)

**QWidget 04-1事件处理之显示关闭事件**

**showEvent(QShowEvent evt)** 控件显示时会被调用（第一次显示或者隐藏后显示），**需要重写事件**

**closeEvent(QCloseEvent evt)** 控件关闭时会被调用（点右上角×），**需要重写事件**

**QWidget 04-2事件处理之控件移动事件**

**moveEvent(QMoveEvent evt)** 控件移动时调用，**需要重写事件**

**QWidget 04-3事件处理之控件调整大小事件**

**resizeEvent(QMoveEvent evt)** 控件大小改变时调用，**需要重写事件**

**⭐**要注意的是，窗口移动仅仅指的是窗口的左上角坐标改变，而调整大小是整个形状大小改变，二者是不同的

**QWidget 04-4事件处理之鼠标事件**

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image5.png)

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image6.png)

**QWidget 04-5事件处理之键盘事件**

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image7.png)

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image6.png)

**QWidget 04-6事件处理之焦点事件**

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image8.png)

**QWidget 04-7事件处理之拖拽事件**

**首先要设置控件状态为可接受拖拽**

**setAcceptDrops(true);**

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image9.png)

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image10.png)

**QWidget 04-8事件处理之绘制事件**

**paintEvent()** 控件显示、手动调用update时触发

class MyWidget9 :public QWidget

{

public:

MyWidget9(QWidget\* parent = nullptr)

:QWidget(parent)

{

setWindowTitle("QWidget绘制事件");

resize(400, 400);

}

protected:

void paintEvent(QPaintEvent\* event) override

{

QPainter painter(this);

painter.setViewport(50, 50, width() - 100, height() - 100);

painter.setWindow(-10, 2, 20, -4);

painter.fillRect(-10, 2, 20, -4, Qt::black);

QPen pen;

pen.setColor(Qt::white);

pen.setWidthF(3.0);

pen.setCosmetic(true);

painter.setPen(pen);

painter.drawLine(QPointF(-10, 0), QPointF(10, 0)); // x axis

painter.drawLine(QPointF(0, 2), QPointF(0, -2)); // y axis

pen.setColor(Qt::green);

pen.setWidth(0);

painter.setPen(pen);

for (double x = -10; x \< 10; x += 0.01)

{

double y = qSin(x);

painter.drawPoint(QPointF(x, y));

}

}

};

**QWidget 04-9事件处理之绘制事件**

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image11.png)

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image12.png)

**QWidget 04-10事件处理之鼠标右键事件**

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image13.png)

class MyWidget11 :public QWidget

{

public:

MyWidget11(QWidget\* parent = nullptr)

:QWidget(parent)

{

setWindowTitle(QStringLiteral("QWidget右键菜单栏事件"));

resize(400, 400);

m_pContextMenu = new QMenu(this);

m_pBackAct = new QAction(QIcon(":/images/back.png"), "返回", this);

connect(m_pBackAct, &QAction::triggered, \[\]() {qDebug() \<\< QStringLiteral("返回了........."); });

}

protected:

void contextMenuEvent(QContextMenuEvent\* event) override

{

qDebug() \<\< "right mouse press.";

m_pContextMenu-\>clear();

m_pContextMenu-\>addAction(m_pBackAct);

m_pContextMenu-\>addAction(QStringLiteral("前进"));

m_pContextMenu-\>addAction(QStringLiteral("重新加载"));

m_pContextMenu-\>addSeparator();

m_pContextMenu-\>addAction(QStringLiteral("另存为"));

m_pContextMenu-\>addSeparator();

m_pContextMenu-\>addAction(QStringLiteral("检查"));

m_pContextMenu-\>move(event-\>globalPos());

m_pContextMenu-\>show();

}

private:

QMenu\* m_pContextMenu;

QAction\* m_pBackAct;

};

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image14.png)

**QWidget 04-10事件处理之事件传递**

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image15.png)

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image16.png)

**QWidget 05光标相关**

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image17.png)

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image18.png)

**QWidget 06父子关系**

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image19.png)

**QWidget 07层级关系**

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image20.png)

**QWidget 08顶层窗口**

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image21.png)

**若要窗口无标题栏，可设置为**

setWindowFlags(Qt::FramelessWindowHint);

实现自己的拖动窗口（无标题栏）

class MyWindow :public QWidget

{

public:

MyWindow(QWidget\* parent = nullptr)

:QWidget(parent)

{

setWindowFlags(Qt::FramelessWindowHint);

resize(500, 500);

setUi();

m_pSizeGrip = new QSizeGrip(this);

}

protected:

void resizeEvent(QResizeEvent\* event) override

{

m_pCloseBtn-\>move(width() - m_nBtnWidth, m_nTopMargin);

m_pMaxBtn-\>move(width() - 2 \* m_nBtnWidth, m_nTopMargin);

m_pMinBtn-\>move(width() - 3 \* m_nBtnWidth, m_nTopMargin);

}

void mousePressEvent(QMouseEvent\* event) override

{

if (event-\>button() == Qt::LeftButton)

{

m_bMoveWindow = true;

m_tOrigin2Press = event-\>globalPos() - pos();

}

}

void mouseMoveEvent(QMouseEvent\* event) override

{

if (m_bMoveWindow)

{

move(event-\>globalPos() - m_tOrigin2Press);

}

}

void mouseReleaseEvent(QMouseEvent\* event) override

{

m_bMoveWindow = false;

}

private:

void setUi()

{

addCloseMaxMinControl();

}

void addCloseMaxMinControl()

{

m_pCloseBtn = new QPushButton(QIcon(":/images/close.png"), "", this);

m_pCloseBtn-\>resize(m_nBtnWidth, m_nBtnHeight);

connect(m_pCloseBtn, &QPushButton::clicked, \[=\]() {this-\>close(); });

m_pMaxBtn = new QPushButton(QIcon(":/images/max.png"), "", this);

m_pMaxBtn-\>resize(m_nBtnWidth, m_nBtnHeight);

connect(m_pMaxBtn, &QPushButton::clicked, \[=\]() {this-\>isFullScreen() ? this-\>showNormal() : this-\>showFullScreen(); });

m_pMinBtn = new QPushButton(QIcon(":/images/min.png"), "", this);

m_pMinBtn-\>resize(m_nBtnWidth, m_nBtnHeight);

connect(m_pMinBtn, &QPushButton::clicked, \[=\]() {this-\>showMinimized(); });

}

int m_nBtnWidth = 40;

int m_nBtnHeight = 25;

int m_nTopMargin = 0;

QPushButton\* m_pCloseBtn;

QPushButton\* m_pMaxBtn;

QPushButton\* m_pMinBtn;

boolm_bMoveWindow = false;

QPoint m_tOrigin2Press; // 鼠标按下时,原点到按下点的向量

QSizeGrip\* m_pSizeGrip; // 可拖动句柄

};

**QWidget 09交互状态**

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image22.png)

**QWidget 10信息提示**

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image23.png)

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image24.png)

**QWidget 11焦点控制**

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image25.png)

**QWidget 12坐标转换**

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image26.png)

void mousePressEvent(QMouseEvent\* event) override

{

qDebug() \<\< "self pos is " \<\< event-\>pos();

qDebug() \<\< "parent pos is " \<\< mapToParent(event-\>pos());

qDebug() \<\< "global pos is " \<\< mapToGlobal(event-\>pos());

}

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image27.png)

**map坐标系是自己的物体坐标系**

**parent坐标系是自己在父控件中的坐标**

**global坐标系是自己在桌面中的坐标**
