---
title: 'C++ 关于TCP通讯的示例'
date: 2021-08-08
lastmod: 2021-08-08
draft: false
tags: ["C++", "TCP"]
categories: ["Protocol"]
authors: ["chase"]
summary: "C++ 关于TCP通讯的示例"
showToc: true
TocOpen: true
hidemeta: false
comments: false
---

客户端Client：
```cpp
#include<stdio.h>
#include<sys/types.h>
#include<stdlib.h>
#include<string>
#include<sys/socket.h>
#include<netinet/in.h>
#include<arpa/inet.h>
#include<unistd.h>
#include<iostream>

using namespace std;

int main()
{
    // 1.创建一个socket（客户端）
    // socket()函数解释：IP协议族，数据流方式，TCP协议
    int socket_fd = socket(AF_INET, SOCK_STREAM,0);
    if(socket_fd == -1)
    {
        std::cout << "socket 创建失败：" << std::endl;
        exit(-1);
    }
    // 2.建立套接子地址
    // 绑定IP和端口号port
    struct sockaddr_in addr;
    addr.sin_family = AF_INET; //IPv4，代表interner协议族
    addr.sin_port = htons(8888); //设置端口，htons是将u_short型变量从主机字节顺序变换为TCP/IP网络字节顺序
    addr.sin_addr.s_addr = inet_addr("192.168.1.5"); // address 对应服务端的ip
    // 3.连接(连接服务端)
    int res = connect( socket_fd , (struct sockaddr*)&addr , sizeof(addr) );
    if(res == -1){
        std::cout <<  " bind 链接失败：" << std::endl;
        exit(-1);
    }
    else{
        std::cout << "bind 链接成功：" << std::endl;
    }
    // 4.发送
    char data[] = "Call server from client!";
    write(socket_fd ,data , sizeof(data) );

    char buffer[255]={};
    int size = read(socket_fd, buffer, sizeof(buffer));//通过fd与客户端联系在一起,返回接收到的字节数

    std::cout << "接收到字节数为： " << size << std::endl;
    std::cout << "从Server上接收到的内容： " << buffer << std::endl;


    // 4.关闭套接字
    close(socket_fd);
    return 0;
}
```


服务端Server：
```cpp
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/socket.h>
#include<netinet/in.h>
#include<arpa/inet.h>
#include<unistd.h>
#include<iostream>

using namespace std;

int main()
{
    // 1.创建一个socket(服务端)
    int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd == -1)
    {
        std::cout << "socket 创建失败： "<< std::endl;
        exit(1);
    }
    // 2.建立套接子地址
    // 绑定IP和端口号port
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(8888);// 将一个无符号短整型的主机数值转换为网络字节顺序，即大尾顺序(big-endian)
    addr.sin_addr.s_addr = inet_addr("192.168.1.5");// inet_addr方法可以转化字符串，主要用来将一个十进制的数转化为二进制的数，用途多于ipv4的IP转化。

    // 3.bind()绑定
    int res = bind(socket_fd,(struct sockaddr*)&addr,sizeof(addr));
    if (res == -1)
    {
        std::cout << "bind创建失败： " << std::endl;
        exit(-1);
    }
    std::cout << "bind ok 等待客户端的连接" << std::endl;
    // 4.监听客户端listen()函数
    // 参数二：进程上限，一般小于30
    listen(socket_fd,30);
    // 5.等待客户端的连接accept()，返回用于交互的socket描述符
    struct sockaddr_in client;
    socklen_t len = sizeof(client);
    int fd = accept(socket_fd,(struct sockaddr*)&client,&len);
    if (fd == -1)
    {
        cout << "accept错误\n" << endl;
        exit(-1);
    }
    // 6.使用第5步返回socket描述符，进行读写通信。
    char *ip = inet_ntoa(client.sin_addr);
    std::cout << "客户： 【" << ip << "】连接成功" << std::endl;
  
    write(fd, "welcome", 7);

    char buffer[255]={};
    int size = read(fd, buffer, sizeof(buffer));//通过fd与客户端联系在一起,返回接收到的字节数

    std::cout << "接收到字节数为： " << size << std::endl;
    std::cout << "从Client上接收到的内容： " << buffer << std::endl;

    // 7.关闭sockfd
    close(fd);
    close(socket_fd);
    return 0;
}
```
在**Linux**系统上安装**g++**，然后在上述文件对应的文件夹中打开终端输入：

```bash
g++ server.cpp -o server
g++ client.cpp -o client
```
生成可执行程序后，打开两个终端，各自输入以下：
```bash
./server
./client
```
