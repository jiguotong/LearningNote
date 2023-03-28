# 十、问题汇总

## 1.设置窗口图标
vs+qt下
打开qrc文件，添加前缀/textEditor，添加文件/res/text.png
应用如下：

```c
setWindowIcon(QIcon(":/textEditor/res/text.png"));
```
图标下载地址：[阿里巴巴矢量图库](https://www.iconfont.cn/)

## 2.设置快捷键
```c
QShortcut* shortcut = new QShortcut(this);
shortcut->setKey(QKeySequence("Ctrl+S"));
shortcut->setContext(Qt::ApplicationShortcut);
connect(shortcut, SIGNAL(activated()), this, SLOT(OnActionSaveFile()));
```

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
将textEditor.exe单独复制到一个临时目录下，在此目录下打开终端，输入
*windeployqt textEditor.exe*
会生成所需要的qt依赖dll，之后把程序依赖的其他dll或者文件夹拷贝进临时目录，可将此临时目录发布。

