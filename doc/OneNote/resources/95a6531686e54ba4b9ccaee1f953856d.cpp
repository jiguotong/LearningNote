#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <Winsock2.h>
#include <iostream>
using namespace std;
int main()
{
	//加载套接字库
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
	SOCKET sockClient = socket(AF_INET, SOCK_DGRAM, 0);
	SOCKADDR_IN addrSrv;
	addrSrv.sin_addr.S_un.S_addr = inet_addr("192.168.1.173");
	//addrSrv.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
	addrSrv.sin_family = AF_INET;
	addrSrv.sin_port = htons(6000);

	//发送数据
	char recvBuf[100];
	char sendBuf[100];
	char tempBuf[200];

	int len = sizeof(SOCKADDR);
	while (1)
	{
		cin >> sendBuf;
		sendto(sockClient, sendBuf, strlen(sendBuf) + 1, 0, (SOCKADDR*)&addrSrv, sizeof(SOCKADDR));
		recvfrom(sockClient, recvBuf, 100, 0, (SOCKADDR*)&addrSrv, &len);
		cout << inet_ntoa(addrSrv.sin_addr) << "		" << recvBuf << endl;
	}
	

	//关闭套接字
	closesocket(sockClient);
	WSACleanup();

	system("pause");
	return 0;
}
