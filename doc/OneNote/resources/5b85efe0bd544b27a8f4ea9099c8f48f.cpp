//#define _CRT_SECURE_NO_WARNINGS
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <Winsock2.h>			//��������ͨ��
#include <windows.h>			//���ڶ��߳�
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <string>
using namespace std;

typedef struct RECVPARAM
{
	SOCKET sock;		//�Ѵ������׽���
	string ip;			//sendto��ip��ַ
	//HWND hwnd;			//�Ի�����
};

DWORD WINAPI RecvMsgThread(LPVOID lpParameter);
DWORD WINAPI SendMsgThread(LPVOID lpParameter);
string ip;
int main()
{
	//���������
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

	//�����׽���
	SOCKET m_socket = socket(AF_INET, SOCK_DGRAM, 0);
	SOCKADDR_IN addrSock;
	addrSock.sin_addr.S_un.S_addr = htonl(INADDR_ANY);//htonl�������ֽ���תΪ�����ֽ���
	addrSock.sin_family = AF_INET;
	addrSock.sin_port = htons(6000);

	//�洢�׽���������
	int retval;
	//���׽���
	retval = bind(m_socket, (SOCKADDR*)&addrSock, sizeof(SOCKADDR));
	if (retval == SOCKET_ERROR)
	{
		closesocket(m_socket);
		cout << "��ʧ��" << endl;
		return false;
	}

	RECVPARAM* pRecvParameter = new RECVPARAM;
	pRecvParameter->sock = m_socket;
	cout << "����������Ҫͨ�ŵ�IP��ַ(��ʽΪ��127.0.0.1)��";
	cin >> pRecvParameter->ip;
	HANDLE hRecvThread = CreateThread(NULL, 0, RecvMsgThread, (LPVOID)pRecvParameter, 0, NULL);
	HANDLE hSendThread = CreateThread(NULL, 0, SendMsgThread, (LPVOID)pRecvParameter, 0, NULL);
	CloseHandle(hRecvThread);		//�رս����߳̾��
	CloseHandle(hSendThread);		//�رշ����߳̾��
	
	while (1)
	{
	}
	//�ر��׽���
	delete pRecvParameter;
	closesocket(m_socket);
	WSACleanup();

	system("pause");
	return 0;
}
//�̺߳���
DWORD WINAPI RecvMsgThread(LPVOID lpParameter)
{
	SOCKET sock = ((RECVPARAM*)lpParameter)->sock;

	SOCKADDR_IN addrFrom;
	int len = sizeof(SOCKADDR);

	char recvBuf[200];		//�洢���ܵ���Ϣ
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
	addrTo.sin_addr.S_un.S_addr = inet_addr(((RECVPARAM*)lpParameter)->ip.c_str());//htonl�������ֽ���תΪ�����ֽ���
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