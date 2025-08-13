---
title: VC++深入详解笔记
updated: 2022-03-26T21:15:53
created: 2022-03-22T11:33:16
---

**第十四章 网络编程**
*14.1 计算机网络基本知识*
(1)socket连接应用成语与网络驱动程序
(2)通信具备：IP地址、协议、端口号
(3)OSI(Open System Interconnection)七层参考模型：
物理层→数据链路层→网络层→传输层→会话层→表示层→应用层
(4)应用层使用的协议：文件传输协议FTP、超文本传输协议HTTP、域名服务DNS、简单邮件传输协议SMTP
传输层使用的协议：传输控制协议TCP、用户数据报协议UDP
网络层使用的协议：网际协议IP、Internet互联网控制报文协议ICMP、Internet组管理协议IGMP
(5)数据封装：在数据前面加上特定的协议头部。
(6)TCP/IP模型：应用层、传输层、网络层、网络接口层。
(7)端口号：一个整数型标识符，范围为0~65535。
(8)socket：套接字，存在于通信区域中，支持TCP/IP网络通信的基本操作单元。
(9)网络字节顺序：低位先存/高位先存；**我们常用的PC机，采用的是低位先存**，TCP/IP协议使用高位先存格式。在网络中不同主机间进行通信时，要统一采用**网络字节顺序**(即高位先存，大端字节序)

*14.2 Windows Sockets的实现*
**(1)套接字的类型**
流式套接字(SOCK_STREAM)、数据报式套接字(SOCK_DGRAM)、原始套接字(RAW)

**(2)基于TCP(面向连接)的socket编程：**
**服务器端：**
①创建套接字(socket)
②将套接字绑定到一个本地地址和端口上(bind)
③将套接字设为监听模式，准备接受客户请求(listen)
④等待客户请求到来；当请求到来后，接受连接请求，返回一个新的对应于此次连接的套接字(accept)
⑤用返回的套接字和客户端进行通信(send/recv)
⑥返回，等待另一客户请求
⑦关闭套接字
**客户端：**
①创建套接字(socket)
②向服务器发出连接请求(connect)
③和服务器端进行通信(send/recv)
④关闭套接字

**(3)基于UDP(面向无连接)的socket编程**
**服务器端：**
①创建套接字(socket)
②将套接字绑定到一个本地地址和端口上(bind)
③等待接受数据(recvfrom)
④关闭套接字
**客户端：**
①创建套接字(socket)
②向服务器发送数据(sendto)
③关闭套接字

*14.3 相关函数*
*14.4 基于UDP的网络应用程序的编写*
*依赖ws2_32.lib*
*服务端：*
[UdpSrv.cpp](../../resources/62873421a1fb4f4888e6f1527bf96c9f.cpp)
*客户端：*
[UdpClient.cpp](../../resources/95a6531686e54ba4b9ccaee1f953856d.cpp)

缺点：只能一条一条交互，一方不能连发消息

*14.5 改进版（多线程）*
*可以连发消息 且客户端/服务端同用一个文件*
[UdpChatMultiThread.cpp](../../resources/fa68104447c0451ebd901a9e7bbfa2b6.cpp)

**第十五章 多线程**
*15.1 示例*

//线程函数
DWORD WINAPI RecvMsgThread(LPVOID lpParameter)
{
SOCKET sock = ((RECVPARAM\*)lpParameter)-\>sock;

SOCKADDR_IN addrFrom;

int len = sizeof(SOCKADDR);

char recvBuf\[200\];//存储接受的消息

int retval;

while (TRUE)

{retval = recvfrom(sock, recvBuf, 200, 0, (SOCKADDR\*)&addrFrom, &len);

if (SOCKET_ERROR == retval)

break;

cout \<\< inet_ntoa(addrFrom.sin_addr) \<\< "" \<\< recvBuf \<\< endl;

}

return 0;
}

int mani(){
//新开子线程

HANDLE hRecvThread = CreateThread(NULL, 0, RecvMsgThread, (LPVOID)pRecvParameter, 0, NULL);

CloseHandle(hRecvThread);//关闭接受线程句柄

while(1)

{

}
}

*15.2 示例程序（基于UDP的网络聊天程序）*
*多线程解决阻塞问题。*
[UdpChatMultiThread.cpp](../../resources/fa68104447c0451ebd901a9e7bbfa2b6.cpp)
