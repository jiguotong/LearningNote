---
title: ——MainWindow
updated: 2023-02-26T23:15:40
created: 2023-02-21T22:51:52
---

一、菜单栏
\#include \<QMenuBar\>
QMenuBar \*bar_menu= QMainWindow::menuBar();

//添加菜单栏的菜单选项
QMenu\* menu_file = menuBar()-\>addMenu(QStringLiteral("文件"));
QMenu\* menu_help = menuBar()-\>addMenu(QStringLiteral("Help"));

//编辑菜单——文件
QAction\* action_new = menu_file-\>addAction(QStringLiteral("新建"));
QAction\* action_open= menu_file-\>addAction(QStringLiteral("打开"));
QAction\* action_save= menu_file-\>addAction(QStringLiteral("保存"));
QAction\* action_save_as= menu_file-\>addAction(QStringLiteral("另存为"));

//Action设置
const QIcon newIcon = QIcon::fromTheme("document-new", QIcon(".\\images\\filenew.png"));
action_new-\>setIcon(newIcon);
action_new-\>setShortcut(QKeySequence::New);

const QIcon openIcon = QIcon::fromTheme("document-new", QIcon(".\\images\\fileopen.png"));
action_open-\>setIcon(openIcon);
action_open-\>setShortcut(QKeySequence::Open);

二、工具栏
\#include \<QToolBar\>
QToolBar\* bar_tool = QMainWindow::addToolBar(QStringLiteral("tool bar"));
\#工具栏可以跟菜单栏共用同一个action
bar_tool-\>addAction(action_new);
bar_tool-\>addAction(action_open);
bar_tool-\>addAction(action_save);

三、状态栏
\#include \<QStatusBar\>
// create status bar
QStatusBar\* pStatubBar = QMainWindow::statusBar();

// hide size grip
pStatubBar-\>setSizeGripEnabled(false);

// left
QLabel\* pageLb = new QLabel("幻灯片 第 1 张，共 1 张",this);
QLabel\* languageLb = new QLabel("中文(中国)", this);
QPushButton\* infoBtn = new QPushButton("辅助功能:一切就绪", this);
infoBtn-\>setIcon(QIcon(":images/dance.png"));
pStatubBar-\>addWidget(pageLb);
pStatubBar-\>addWidget(languageLb);
pStatubBar-\>addWidget(infoBtn);

// right
QToolButton\* gridBtn = new QToolButton(this);
gridBtn-\>setIcon(QIcon(":images/grid.png"));
QToolButton\* bookBtn = new QToolButton(this);
bookBtn-\>setIcon(QIcon(":images/book.png"));
QToolButton\* drinkBtn = new QToolButton(this);
drinkBtn-\>setIcon(QIcon(":images/drink.png"));

QSlider\* percentSlider = new QSlider(Qt::Horizontal,this);
percentSlider-\>setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
percentSlider-\>setValue(53);
QLabel\* percentLb = new QLabel(" 53%", this);

pStatubBar-\>addPermanentWidget(gridBtn);
pStatubBar-\>addPermanentWidget(bookBtn);
pStatubBar-\>addPermanentWidget(drinkBtn);
pStatubBar-\>addPermanentWidget(percentSlider);
pStatubBar-\>addPermanentWidget(percentLb);

四、停靠栏
\#include \<QDockWidget\>

QDockWidget\* dock = new QDockWidget(QStringLiteral("New Dock"), this);
QLabel\* label = new QLabel(dock);
label-\>setText(QStringLiteral("dock_label"));

dock-\>setWidget(label); //dock添加子控件
dock-\>setFeatures(QDockWidget::AllDockWidgetFeatures); //设置dock属性为可移动、可浮动、可关闭
dock-\>setAllowedAreas(Qt::LeftDockWidgetArea \| Qt::RightDockWidgetArea); //可以停靠的位置
this-\>addDockWidget(Qt::LeftDockWidgetArea, dock); /\* 初始化显示在窗口左侧 \*/

![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image1.png)

五、启动界面
\#include \<QSplashScreen\>

//创建启动画面
QPixmap lodingPix("./images/screen.png"); //创建启动需要显示的图片
lodingPix = lodingPix.scaled(480, 270, Qt::KeepAspectRatio, Qt::SmoothTransformation);
QSplashScreen splash(lodingPix); //利用图片创建一个QSplashScreen对象
splash.show(); //显示此启动图片
splash.showMessage("Loading...", Qt::AlignTop \| Qt::AlignRight, Qt::white); //在图片上显示文本信息，第一个参数是文本内容，第二个是显示的位置，第三个是文本颜色

QDateTime time = QDateTime::currentDateTime();
QDateTime currentTime = QDateTime::currentDateTime(); //记录当前时间
while (time.secsTo(currentTime) \<= 5) //5为需要延时的秒数
currentTime = QDateTime::currentDateTime();
