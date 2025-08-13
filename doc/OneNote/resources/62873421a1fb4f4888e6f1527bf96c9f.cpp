//#define _CRT_SECURE_NO_WARNINGS
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <Winsock2.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
using namespace std;
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
	SOCKET sockSrv = socket(AF_INET, SOCK_DGRAM, 0);
	SOCKADDR_IN addrSrv;
	addrSrv.sin_addr.S_un.S_addr = htonl(INADDR_ANY);//htonl将主机字节序转为网络字节序
	addrSrv.sin_family = AF_INET;
	addrSrv.sin_port = htons(6000);

	//绑定套接字
	bind(sockSrv, (SOCKADDR*)&addrSrv, sizeof(SOCKADDR));

	//等待并接受数据
	SOCKADDR_IN addrClient;
	int len = sizeof(SOCKADDR);
	char recvBuf[100];
	char sendBuf[100];
	char tempBuf[200];

	while (1)
	{
		recvfrom(sockSrv, recvBuf, 100, 0, (SOCKADDR*)&addrClient, &len);
		cout << inet_ntoa(addrClient.sin_addr) << "		" << recvBuf << endl;
		cin>> sendBuf;
		sendto(sockSrv, sendBuf, strlen(sendBuf) + 1, 0, (SOCKADDR*)&addrClient, sizeof(SOCKADDR));
	}
	
	
	//关闭套接字
	closesocket(sockSrv);
	WSACleanup();

	system("pause");
	return 0;
}
