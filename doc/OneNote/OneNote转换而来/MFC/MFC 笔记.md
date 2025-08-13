---
title: MFC 笔记
updated: 2025-03-03T14:57:07
created: 2021-09-02T17:08:57
---

**1、几种绘图方式**
第一种：利用SDK全局函数实现
//获取设备描述表
*HDC* hdc;
//调用全局函数获得当前窗口的设备描述表，CWnd::m_hWnd根据继承原理，CDrawView继承了CWnd类的数据成员
hdc = ::*GetDC*(*m_hWnd*);
//移动到线条的起点
MoveToEx(hdc, m_pOrigin.*x*, m_pOrigin.*y*, *NULL*);//第四个参数用于保存鼠标移动前的位置，此处不需要，设为NULL
//画线
LineTo(hdc, point.*x*, point.*y*);
//释放设备描述表
::*ReleaseDC*(*m_hWnd*, hdc);

第二种：利用CDC类实现
//说明：CDC类封装了所有与绘图相关的操作
*CDC*\* pDC = *GetDC*();//定义CDC类型的指针，利用CWnd类的成员函数GetDC获得当前窗口的设备描述表对象的指针
pDC-\>*MoveTo*(m_pOrigin);//利用CDC类的成员函数MoveTo和LineTo完成画线功能
pDC-\>LineTo(point);
ReleaseDC(pDC);

第三种：利用MFC的CClientDC类实现
//说明：此类派生于CDC类，在构造时调用GetDC()函数，在析构时调用ReleaseDC()函数，因此无需显示调用这两个函数。
//CClientDC dc(this);//在当前视图窗口画线方法
*CClientDC* dc(*GetParent*());//获取当前视图窗口的父窗口，可以在父窗口画线（框架窗口）
dc.MoveTo(m_pOrigin);
dc.LineTo(point);

第四种：利用MFC的CWindowDC类实现
//CWindowDC dc(this);//只能在视类中画线
*CWindowDC* dc(*GetParent*());//可以在父窗口中画线
dc.MoveTo(m_pOrigin);
dc.LineTo(point);

若想改变颜色啥的，自己添加画笔（以上四种都实用）
*CPen* pen(*PS_SOLID*, 5, *RGB*(255, 255, 0)); //创建画笔对象
*CClientDC* dc(this);
*CPen*\* pOldPen = dc.SelectObject(&pen); //将pen选进设备描述表中，并返回指向先前Oldpen的指针，方便后面进行恢复
dc.MoveTo(m_pOrigin);
dc.LineTo(point);
dc.SelectObject(pOldPen); //恢复设备描述表中的之前的画笔对象

\############画刷操作，不用加进设备描述表中
*CClientDC* dc(this); //创建普通画刷
*CBrush* brush(*RGB*(GetRand(0, 255), GetRand(0, 255), GetRand(0, 255)));
dc.*FillRect*(*CRect*(m_pOrigin, point), &brush);
\############位图画刷
*CBitmap* bitmap; //创建位图对象
bitmap.*LoadBitmap*(IDB_BITMAP1);//加载位图资源
*CBrush* brush(&bitmap); //创建位图画刷
*CClientDC* dc(this); //创建并获得设备描述表
dc.*FillRect*(*CRect*(m_pOrigin, point), &brush);//利用红色画刷填充矩形区域
\############透明画刷
*CClientDC* dc(this); //创建并获得设备描述表
*CBrush*\* pBrush = *CBrush*::*FromHandle*((*HBRUSH*)*GetStockObject*(*NULL_BRUSH*)); //创建空画刷
*CBrush*\* pOldBrush = dc.*SelectObject*(pBrush); //将空画刷选入设备描述表
dc.*Rectangle*(*CRect*(m_pOrigin, point)); //绘制一个矩形
dc.*SelectObject*(pOldBrush);

**2、编码问题**
[**不能从const char \*转换为LPCWSTR --VS经常碰到**](https://www.cnblogs.com/dongsheng/p/3586418.html)
不能从const char \*转换为LPCWSTR
在VC 6.0中编译成功的项目在VS2005 vs2005、vs2008、vs2010中常会出现类型错误。
经常出现的错误是：不能从const char \*转换为LPCWSTR
可行的办法是使用 \_T("TEST")转换，或者TEXT("TEST" )都可以

**3、MFC中标题、图标以及背景更换**
*BOOL* CMainFrame::PreCreateWindow(*CREATESTRUCT*& cs)
{
if( !*CFrameWndEx*::*PreCreateWindow*(cs) )

return *FALSE*;

// TODO: 在此处通过修改

// CREATESTRUCT cs 来修改窗口类或样式

==cs.*style* &= ~*FWS_ADDTOTITLE*;==

==cs.*lpszName* = *\_T*("DRAW PRO");==

return *TRUE*;
}
[**https://blog.csdn.net/shufac/article/details/25659093**](https://blog.csdn.net/shufac/article/details/25659093)
****
****
**4、菜单是否是弹出式菜单 pop-up**
将菜单的按钮的属性中把pop-up按钮关掉
**5、新建弹出对话框的方式**
**①资源中添加资源，选择dialog，摆控件**
**②类向导，添加MFC类，选择基类CDialog**
**③在draw.app中添加消息处理**
**6、新建弹出对话框的方式**
①添加一个dialog的子类，摆控件，加两个成员变量，**关联变量与按钮**（给按钮添加变量）
②CNewButton新建一个新的按钮类，添加一个用于存另一个按钮的地址的指针变量
③添加虚函数中的OnInitDialog初始函数，实现两个成员变量的地址互换
**7、用于随时调试**
加一个static控件
*CString* str;
str.*Format*(*\_T*("x=%d,y=%d"), point.*x*, point.*y*);
*SetDlgItemText*(*IDC_STATIC*, str);
**8、解决改变窗口初始大小的方式**
<https://blog.csdn.net/weixin_30840253/article/details/94993669>
**9、图标的动态变化**
运用OnTimer消息函数
**10、定制应用程序外观**
工具栏、状态栏、进度栏
**11、添加开机启动动画 CWzdSplash作为动画**
<https://www.cnblogs.com/qiwu1314/p/9116557.html>
**12、设置对话框（自己创建对话框）、颜色对话框CColorDialog类、字体对话框CFontDialog类、添加示例**
**13、解决对话框的默认ESC和ENTER退出事件**
<https://www.cnblogs.com/yangjig/p/3913751.html>

14、字符串之间的转换大全
15、显示图片简单代码
CPaintDC dc(this);

CString path = "C:/Users/justin/Desktop/Greed_snake/res/001.jpg";

CImage nImage;

nImage.Load(path);

nImage.Draw(dc,200,100,200,200);

16、字符串分割简单代码
CString str = "abc,def";
int n = str.Find('',");
CString result = str.Left(n);//把n前面的赋给result

17、MessageBox使用
若是在对话框中使用
MessageBox(L"导出成功！", L"提示", MB_OK);

若是在非对话框中使用
MessageBox(NULL, L"导出成功！", L"提示", MB_OK);

**18、读取txt的好用的方法**
CString address = \_T("./OutputData/result.txt");
CStdioFile hFile;
if (hFile.Open(address, CFile::modeRead))
{
//将文件内容逐行导入

CString strData;

while (hFile.ReadString(strData))

{

CString strX, strY, strZ;

//分割字符串，分别接收

AfxExtractSubString(strX, strData, 0, L' ');

AfxExtractSubString(strY, strData, 1, L' ');

AfxExtractSubString(strZ, strData, 2, L' ');

}

hFile.Close();
}
else
{
return;
}

**19、数据转换大全**
double **CString2Double**(CString strData)
{
double i;

i = \_ttof(strData);

return i;
}
CString **Double2CString**(double d)
{
CString strDouble;

strDouble.Format(L"%0.4lf",d);

return strDouble;
}
std::string **ws2s**(const std::wstring& ws)
{
string curLocale = setlocale(LC_ALL, NULL); // curLocale = "C";

setlocale(LC_ALL, "chs");

const wchar_t\* \_Source = ws.c_str();

size_t \_Dsize = 2 \* ws.size() + 1;

char \*\_Dest = new char\[\_Dsize\];

memset(\_Dest, 0, \_Dsize);

wcstombs(\_Dest, \_Source, \_Dsize);

string result = \_Dest;

delete\[\]\_Dest;

setlocale(LC_ALL, curLocale.c_str());

return result;
}
std::wstring **s2ws**(const std::string& s)
{
setlocale(LC_ALL, "chs");

const char\* \_Source = s.c_str();

size_t \_Dsize = s.size() + 1;

wchar_t \*\_Dest = new wchar_t\[\_Dsize\];

wmemset(\_Dest, 0, \_Dsize);

mbstowcs(\_Dest, \_Source, \_Dsize);

wstring result = \_Dest;

delete\[\]\_Dest;

setlocale(LC_ALL, "C");

return result;
}

std::string **CString2string**(CString instr)
{
std::wstring ws(instr);

std::string str;

str.assign(ws.begin(), ws.end());

return str;
}
或者
std::string string **CString2string**(CString &cstr)
{
string str = ws2s(cstr.GetBuffer());

return str;
}

char\* 转CString
char temp\[256\];
CString rockGrade;
rockGrade.Format(\_T("%s"), temp);

**20、MFC中进度条的使用**
CProgressCtrl m_progress;

DDX_Control(pDX, IDC_PROGRESS, m_progress); //控件与变量相关联
m_progress.SetRange32(0,100);
m_progress.SetPos(50);

**//进度条的设计有点奇怪，可能当前SetPos不会立即执行，在下一条语句之后执行。**

21、string转LPCTSTR
//Multi-Byte编码下, string转LPCSTR(LPCTSTR)类型:
stringstr = "hello, I'm string";  
LPCSTR strtmp1 = str.c_str();

//Unicode编码下, string转LPCWSTR(LPCTSTR)类型:
stringstr = "hello, I'm string";  
size_t size = str.length();  
wchar_t\* buffer=newwchar_t\[size+1\];  
MultiByteToWideChar(CP_ACP, 0, str.c_str(), size, buffer, size\*sizeof(wchar_t));  
buffer\[size\] = 0;

delete buffer;//用完删除

22、**AfxMessageBox()实参传递问题：**
**宽字符串wstring如何放到AfxMessageBox()里面：**
wstring aimCloud = pInfo-\>strAimObject; //宽字符串
aimCloud += \_T("点云不在线路参数之内，请检查线路参数是否正确以及点云是否已经进行坐标转换操作！");
AfxMessageBox(aimCloud.c_str());

**字符串string如何放到AfxMessageBox()里面：**
string aimCloud = str；
aimCloud += "点云不在线路参数之内，请检查线路参数是否正确以及点云是否已经进行坐标转换操作!";
AfxMessageBox(aimCloud.c_str());

**int/float/double如何放到AfxMessageBox()里面：%d %f**
CString temp_value = \_T(""); //temp_value用来处理float值
temp_value.Format(\_T("%d"), points_count);//固定格式
AfxMessageBox(temp_value);

23、MFC对话框控件**Check-box Control**用法：控制其他控件是否可用。
auto pbtn = (CButton\*)GetDlgItem(IDC_CHECK_LOCAL_ANALYSE);
if (pbtn-\>GetCheck())
{
m_edit_section_left_start.EnableWindow();

m_edit_section_right_start.EnableWindow();
}
else
{
m_edit_section_left_start.EnableWindow(FALSE);

m_edit_section_right_start.EnableWindow(FALSE);
}
![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image1.png)

**==24、当前窗口重绘==**
CRect rectDlg;
GetClientRect(rectDlg);// 获得窗体的大小
int pointWidth = rectDlg.Width();// 获取窗体宽度
int pointHeight = rectDlg.Height();// 获取窗体高度
RedrawWindow(CRect(0, 0, pointWidth, pointHeight));// 重绘指定区域

25、改变当前对话框的位置为鼠标光标位置
GetCursorPos(&point);
SetWindowPos(NULL, point.x, point.y, length, 60, 0);

26、设置某个Edit为空
GetDlgItem(IDC_CDOUBLEEDIT)-\>SetWindowText(\_T(""));

==27、获得对话框中的某个资源的方法！！！==
**==auto pbtn = (CButton\*)GetDlgItem(IDOK);==**
**==pbtn-\>EnableWindow(true);==**

28、使用类向导的时候出现无法添加消息跟虚函数的情况下：
![](C:\Users\Administrator\AppData\Local\Temp\国同 的笔记本\pandoc/media/image2.jpeg)

解决方法：
1、关闭解决方案
2、删掉解决方案下的.vs文件中的.suo文件
3、重启就好了，注意改一下debug或者release

29、新建文件夹的方式
WCHAR szPath\[MAX_PATH\];
ZeroMemory(szPath, sizeof(szPath));

BROWSEINFO bi;
bi.hwndOwner = GetForegroundWindow();
bi.pidlRoot = NULL;
bi.pszDisplayName = szPath;
bi.lpszTitle = L"New Project：";
bi.ulFlags = BIF_NEWDIALOGSTYLE \| BIF_RETURNONLYFSDIRS \| BIF_EDITBOX \| BIF_RETURNFSANCESTORS;
bi.lpfn = NULL;
bi.lParam = 0;
bi.iImage = 0;
LPITEMIDLIST pidl = NULL;
SHGetSpecialFolderLocation(NULL, CSIDL_DESKTOP, &pidl);
bi.pidlRoot = pidl;

//弹出选择目录对话框
LPITEMIDLIST lp = SHBrowseForFolder(&bi);

if (lp && SHGetPathFromIDList(lp, szPath))
{
if (FALSE == PathIsDirectory(szPath))

{

AfxMessageBox(\_T("You have selected a path that either does not exist or you lack permission to access. "));

return;

}

CSystemConfig::GetInstance()-\>SetCurrentProjectPath(szPath);
}

30、另存为文件时选择存放目录
Cstring filePath = \_T("报表");
CString strFile;
CFileDialog dlg(FALSE, \_T("xlsx"), filePath);//FALSE表示为“另存为”对话框，否则为“打开”对话框
if (dlg.DoModal() == IDOK)
{
strFile = dlg.GetPathName();//获取完整路径
}
else
{
return FALSE;

31、对话框中的tab切换控件使用：  
（1）看看对话框中属性中的control项是否是true；或者style切换一下。
（2）如果是动态创建的edit，则需要加上WS_TABSTOP来保证让edit可以被切换。

32、MFC 关于OnPaint绘图的一些经验
<https://blog.csdn.net/Justin_JGT/article/details/124669343?spm=1001.2014.3001.5502>

33、
