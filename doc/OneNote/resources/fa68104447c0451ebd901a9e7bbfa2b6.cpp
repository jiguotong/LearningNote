//#define _CRT_SECURE_NO_WARNINGS
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <Winsock2.h>			//用于网络通信
#include <windows.h>			//用于多线程
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <string>
using namespace std;

typedef struct RECVPARAM
{
	SOCKET sock;		//已创建的套接字
	string ip;			//sendto的ip地址
	//HWND hwnd;			//对话框句柄
};

DWORD WINAPI RecvMsgThread(LPVOID lpParameter);
DWORD WINAPI SendMsgThread(LPVOID lpParameter);
string ip;
int main()
{
	//加载网络库
	WORD wVersionRequested;
	WSADATA wsaData;
	int err;

	wVersionRequested = MAKEWORD(1, 1);

	err = WSAStartup(wVersionRequested, &wsaData);

	if (err != 0) {
		return 0;
	}
	if (LOBYTE(wsaData.wVersion) != 1 || HIBYTE(wsaData.wVersion) != 1)
	{
		WSACleanup();
		return 0;
	}

	//创建套接字
	SOCKET m_socket = socket(AF_INET, SOCK_DGRAM, 0);
	SOCKADDR_IN addrSock;
	addrSock.sin_addr.S_un.S_addr = htonl(INADDR_ANY);//htonl将主机字节序转为网络字节序
	addrSock.sin_family = AF_INET;
	addrSock.sin_port = htons(6000);

	//存储套接字描述符
	int retval;
	//绑定套接字
	retval = bind(m_socket, (SOCKADDR*)&addrSock, sizeof(SOCKADDR));
	if (retval == SOCKET_ERROR)
	{
		closesocket(m_socket);
		cout << "绑定失败" << endl;
		return false;
	}

	RECVPARAM* pRecvParameter = new RECVPARAM;
	pRecvParameter->sock = m_socket;
	cout << "请输入你想要通信的IP地址(格式为：127.0.0.1)：";
	cin >> pRecvParameter->ip;
	HANDLE hRecvThread = CreateThread(NULL, 0, RecvMsgThread, (LPVOID)pRecvParameter, 0, NULL);
	HANDLE hSendThread = CreateThread(NULL, 0, SendMsgThread, (LPVOID)pRecvParameter, 0, NULL);
	CloseHandle(hRecvThread);		//关闭接受线程句柄
	CloseHandle(hSendThread);		//关闭发送线程句柄
	
	while (1)
	{
	}
	//关闭套接字
	delete pRecvParameter;
	closesocket(m_socket);
	WSACleanup();

	system("pause");
	return 0;
}
//线程函数
DWORD WINAPI RecvMsgThread(LPVOID lpParameter)
{
	SOCKET sock = ((RECVPARAM*)lpParameter)->sock;

	SOCKADDR_IN addrFrom;
	int len = sizeof(SOCKADDR);

	char recvBuf[200];		//存储接受的消息
	int retval;

	while (TRUE) 
	{	retval = recvfrom(sock, recvBuf, 200, 0, (SOCKADDR*)&addrFrom, &len);
		if (SOCKET_ERROR == retval)
			break;
		cout << inet_ntoa(addrFrom.sin_addr) << "		" << recvBuf << endl;
	}
	return 0;
}
DWORD WINAPI SendMsgThread(LPVOID lpParameter)
{
	SOCKET sock = ((RECVPARAM*)lpParameter)->sock;

	SOCKADDR_IN addrTo;
	addrTo.sin_addr.S_un.S_addr = inet_addr(((RECVPARAM*)lpParameter)->ip.c_str());//htonl将主机字节序转为网络字节序
	addrTo.sin_family = AF_INET;
	addrTo.sin_port = htons(6000);

	string sendBuf;

	while (TRUE)
	{
		getline(cin, sendBuf);
		if (!sendBuf.size())
			continue;
		sendto(sock, sendBuf.c_str(), strlen(sendBuf.c_str()) + 1, 0, (SOCKADDR*)&addrTo, sizeof(SOCKADDR));
	}
	return 0;
}