# 五、QT中的线程

## （一）初识线程

## （二）

# 九、样式表大全

## 1.样式表概述

## 2.QLabel样式表

**（1）字体样式**
**font-family** 为设置字体类型，标准形式需要加双引号，不加也可能会生效，具体看系统是否支持，中英文都支持，但要保证字体编码支持，一般程序编码为"utf-8"时没问题。
**font-size** 为设置字体大小，单位一般使用 px 像素
**font-style** 为设置字体斜体样式，italic 为斜体， normal 为不斜体
**font-weight** 为设置字体加粗样式，bold 为加粗， normal 为不加粗
**font** 为同时设置字体 style weight size family 的样式，但是 style 和 weight 必须出现在开头，size 和 family 在后面，而且 size 必须在 family 之前，否则样式将不生效，font 中不能设置颜色，可以单独设置 style weight 和 size，不能单独设置 family
**color** 为设置字体颜色，可以使用十六进制数表示颜色，也可以使用某些特殊的字体颜色：red, green, blue 等，或者使用 rgb(r,g,b) 和 rgba(r,g,b,a) 来设置，其中 r、g、b、a 值为0~255，如果想不显示颜色可以设置值为透明 transparent
**Tip:** 字体颜色用的是 color 属性，没有 font-color 这个属性的

**（2）文字位置**
**padding-left** 为设置文字距离左边边界的距离
**padding-top** 为设置文字距离顶边边界的距离
**padding-right** 为设置文字距离右边边界的距离
**padding-bottom** 为设置文字距离底边边界的距离
**Tip:** 在 qss 中，属性 text-align 对 Label 是不起作用的，只能通过设置 padding 来实现文字的显示位置；一般 padding-left 相当于 x 坐标，padding-top 相当于 y 坐标，设置这两个就可以在任意位置显示了（默认情况下文字是上下左右都居中显示的）

**（3）边框样式**
**border-style** 为设置边框样式，solid 为实线， dashed 为虚线， dotted 为点线， none 为不显示（如果不设置 border-style 的话，默认会设置为 none）
**border-width** 为设置边框宽度，单位为 px 像素
**border-color** 为设置边框颜色，可以使用十六进制数表示颜色，也可以使用某些特殊的字体颜色：red, green, blue 等，或者使用 rgb(r,g,b) 和 rgba(r,g,b,a) 来设置，其中 r、g、b、a 值为0~255，如果想不显示颜色可以设置值为透明 transparent
**border** 为同时设置 border 的 width style color 属性，但值的顺序必须是按照 width style color 来写，不然不会生效！
**border-top-style** 为设置顶部边框样式
**border-top-width** 为设置顶部边框宽度
**border-top-color** 为设置顶部边框颜色
**border-top** 为设置顶部边框 width style color 的属性，原理和 border 一致

**（4）边框半径**
**border-top-left-radius** 为设置左上角圆角半径，单位 px 像素
**border-top-right-radius** 为设置右上角圆角半径，单位 px 像素
**border-bottom-left-radius** 为设置左下角圆角半径，单位 px 像素
**border-bottom-right-radius** 为设置右上角圆角半径，单位 px 像素
**border-radius** 为设置所有边框圆角半径，单位为 px 像素，通过圆角半径可以实现圆形的 Label

**（5）背景样式**
**background-color** 为设置背景颜色，可以使用十六进制数表示颜色，也可以使用某些特殊的字体颜色：red, green, blue 等，或者使用 rgb(r,g,b) 和 rgba(r,g,b,a) 来设置，其中 r、g、b、a 值为0~255，如果想不显示颜色可以设置值为透明 transparent
**background-image** 为设置背景图片，图片路径为 url(image-path)
**background-repeat** 为设置背景图是否重复填充背景，如果背景图片尺寸小于背景实际大小的话，默认会自动重复填充图片，可以设置为 no-repeat 不重复，repeat-x 在x轴重复，repeat-y 在y轴重复
**background-position** 为设置背景图片显示位置，只支持 left right top bottom center；值 left right center 为设置水平位置，值 top bottom center 为设置垂直位置
**background** 为设置背景的所有属性，color image repeat position 这些属性值出现的顺序可以任意

```css
QLabel
{   
    font-family: "Microsoft YaHei";
    font-size: 14px;
    font-style: italic;
    font-weight: bold;
    color: #BDC8E2;
    font: bold italic 18px "Microsoft YaHei";

    padding-left: 10px;
    padding-top: 8px;
    padding-right: 7px;
    padding-bottom: 9px;

    border-style: solid;
    border-width: 2px;
    border-color: red;
    border: 2px solid red;

    border-top-style: solid;
    border-top-width: 2px;
    border-top-color: red;
    border-top: 2px solid red;
  
    border-right-style: solid;
    border-right-width: 3px;
    border-right-color: green;
    border-right: 3px solid green;
  
    border-bottom-style: solid;
    border-bottom-width: 2px;
    border-bottom-color: blue;
    border-bottom: 2px solid blue;
  
    border-left-style: solid;
    border-left-width: 3px;
    border-left-color: aqua;
    border-left: 3px solid aqua;

    border-top-left-radius: 20px;
    border-top-right-radius: 20px;
    border-bottom-left-radius: 20px;
    border-bottom-right-radius: 20px;
    border-radius: 20px;

    background-color: #2E3648;
    background-image: url("./image.png");
    background-repeat: no-repeat; 
    background-position: left center;
    background: url("./image.png") no-repeat left center #2E3648;
}
```

# 十、问题汇总

## 1.设置窗口图标以及软件图标

（1）设置窗口图标，包括任务栏图标
vs+qt下
打开qrc文件，添加前缀/textEditor，添加文件/res/text.png
应用如下：

```c
setWindowIcon(QIcon(":/textEditor/res/text.png"));
```

（2）设置软件图标，包括桌面图标
vs+qt下
添加.rc文件，内容如下

IDI_ICON1 ICON DISCARDABLE "res/AppIcon128.ico"

转化为.ico的软件可使用 格式工厂

ps:图标下载地址：[阿里巴巴矢量图库](https://www.iconfont.cn/)

## 2.设置快捷键

```c
QShortcut* shortcut = new QShortcut(this);
shortcut->setKey(QKeySequence("Ctrl+S"));
shortcut->setContext(Qt::ApplicationShortcut);
connect(shortcut, SIGNAL(activated()), this, SLOT(OnActionSaveFile()));
```

或
action_save->setShortcut(tr("Ctrl+S"));

## 3.设置软件开机启动动画

```c
QApplication a(argc, argv);
//设置开机启动动画
QPixmap pixmap(":/textEditor/res/splash.png");   //设置启动画面
pixmap = pixmap.scaled(500, 500, Qt::KeepAspectRatio);
QSplashScreen splash(pixmap);
splash.show();   //显示此启动图像
a.processEvents();   //使得程序在显示启动画面的同时还能够响应其他事件

Sleep(4000);
Mainwindow w;
w.show();

splash.finish(&w);  //程序启动画面完成

return a.exec();
```

## 4.Qt程序打包发布

（1）发布
将textEditor.exe单独复制到一个临时目录下，在此目录下打开终端，输入
*windeployqt textEditor.exe*
会生成所需要的qt依赖dll，之后把程序依赖的其他dll或者文件夹拷贝进临时目录，可将此临时目录发布。

（2）封装exe
使用封装工具将所有的exe以及dll封装成一个安装exe（以Enigma Virtual Box为例）
下载安装网址https://enigmaprotector.com/en/downloads.html
封装教程：
选择源程序
选择输出程序
添加文件(依赖)
添加文件夹
process

## 5.设置子窗口位于父窗口中心

```c
QPoint globalPos = parentWidget->mapToGlobal(QPoint(0,0));//父窗口绝对坐标
int x = globalPos.x() + (parentWidget->width() - this->width()) / 2;//x坐标
int y = globalPos.y() + (parentWidget->height() - this->height()) / 2;//y坐标
this->move(x, y);//窗口移动
```

## 6.状态栏使用

一般状态信息

```c
QString posLabel = new QLabel(bar_status);
bar_status->addWidget(posLabel);
posLabel->setText(QString("Row:%1 Col:%2\t\t")
    .arg(row).arg(col));
```

永久状态信息

```c
QLabel* perLabel = new QLabel(QStringLiteral("建设美丽祖国"), this);
bar_status->insertPermanentWidget(1, perLabel); //现实永久信息
```

## 7.防止软件二次打开

```c
#include <QMutex>
#include <QSharedMemory>
#inlcude <QMessageBox>
int main(int argc, char *argv[]){
    QApplication a(argc,argv);

    // 设置一个互斥量
    QMutex mutex;
    mutex.lock();// 开启临界区
    // 在临界区内创建SingleApp的共享内存块
    static QSharedMemory *shareMem = new QSharedMemory("SingleApp");
    if(!shareMem.create(1)){
        mutex.unlock();// 关闭临界区
        QMessageBox::information(0, "Tip", "App has been running!");
        return -1;  // 创建失败，说明已有一个程序在运行，退出当前程序
    }
    mutex.unlock();// 关闭临界区
    //继续执行其他代码
    ······
    return a.exec();
}
```

## 8.如何添加背景音乐/音效

vs+qt下
项目属性->Qt Project Settings->Qt Modules中添加multimedia

```c
#include <QSoundEffect>
QSoundEffect* effect = new QSoundEffect(this);
effect->setSource(QUrl::fromLocalFile("..\\res\\Alarm01.wav"));
effect->setLoopCount(QSoundEffect::Infinite);
effect->setVolume(0.25f);
effect->play();
```

## 9.非模态对话框的显示与及时释放(可同时存在多个)

```c
    MyDialog* remindPop = new MyDialog(this);
    remindPop->setAttribute(Qt::WA_DeleteOnClose);// 设置退出自动销毁
    remindPop->show();
```
