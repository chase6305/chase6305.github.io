---
title: 'C++ 关于UDP通讯的示例'
date: 2021-08-08
lastmod: 2021-08-08
draft: false
tags: ["C++", "UDP"]
categories: ["Protocol"]
authors: ["chase"]
summary: "C++ 关于UDP通讯的示例"
showToc: true
TocOpen: true
hidemeta: false
comments: false
---


# UDP介绍

UDP是User Datagram Protocol的简称，即用户数据报协议，为一种无连接的传输层协议，提供面向简单不可靠信息传送服务。客户端Client和服务端Server在交互数据之前无需像TCP那样是先建立连接。
在网络质量较差的情况下，UDP协议数据包丢失会比较严重。但由于UDP的特性，其不属于连接型协议，具有资源消耗小，处理速度快的优点，所以通常音频、视频和普通数据在传送时使用UDP较多，丢失一、两个Packet也不会有太多的影响，同时像微信Wechat聊天。

## 客户端

```cpp
#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <string>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <netdb.h>  
using namespace std;

#define DEST_PORT 5001
#define LOCAL_PORT 8080

int main() {
  int sockfd;

  char ipStr[32];
  char name[256];
  gethostname(name, sizeof(name));

  // 1.创建socket
  sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (-1 == sockfd) {
    return false;
    puts("Failed to create socket");
  }

  // 2.设置地址与端口
  struct sockaddr_in addr;
  socklen_t addr_len = sizeof(addr);

  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;         // Use IPV4
  addr.sin_port = htons(LOCAL_PORT); //
  addr.sin_addr.s_addr = htonl(INADDR_ANY); // INADDR_ANY 绑定本地IP

  const int bBroadcast = 1;
  int nb = setsockopt(sockfd, SOL_SOCKET, SO_BROADCAST,
                      (const char *)&bBroadcast, sizeof(bBroadcast));
  if (nb < 0) {
    perror("setsockopt SO_BROADCAST");
    exit(1);
  }

  // 3.绑定获取数据的端口，作为发送方，不绑定也行
  if (bind(sockfd, (struct sockaddr *)&addr, addr_len) == -1) {
    printf("Failed to bind socket on port %d\n", LOCAL_PORT);
    close(sockfd);
    return false;
  }
  char buffer[128];
  memset(buffer, 0, 128);

  int counter = 0;
  while (1) {
    addr.sin_family = AF_INET;
    addr.sin_port = htons(DEST_PORT);
    addr.sin_addr.s_addr = htonl(INADDR_BROADCAST);
    
    char send_buf[20] = "Hello!";

    sendto(sockfd, send_buf, strlen(send_buf), 0, (sockaddr *)&addr, addr_len);

    printf("IP sended %s\n", inet_ntoa(addr.sin_addr));
    printf("Sended %d\n", ++counter);
    sleep(1);

    int sz = recvfrom(sockfd, buffer, 128, 0, (sockaddr *)&addr, &addr_len);
    if (sz > 0) {
      buffer[sz] = 0;
      printf("Get Message %d:\n %s\n", counter++, buffer);
      printf("get IP %s \n", inet_ntoa(addr.sin_addr));
      printf("get Port %d \n\n", ntohs(addr.sin_port));
    } else {
      puts("timeout");
    }
  }

  close(sockfd);
  return 0;
}
```

## 服务端

```cpp
#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#define DEST_PORT 5001

int main() {

  int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (-1 == sockfd) {
    return false;
    puts("Failed to create socket");
  }

  struct sockaddr_in addr;
  socklen_t addr_len = sizeof(addr);

  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;        // Use IPV4
  addr.sin_port = htons(DEST_PORT); //

  const int opt = 1;
  int nb = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt,
                      sizeof(opt));
  if (nb < 0) {
    perror("setsockopt error");
    exit(1);
  }
  // Bind
  // 端口，用来接受之前设定的地址与端口发来的信息,作为接受一方必须bind端口，并且端口号与发送方一致
  if (bind(sockfd, (struct sockaddr *)&addr, addr_len) == -1) {
    printf("Failed to bind socket on port %d\n", DEST_PORT);
    close(sockfd);
    return false;
  }

  char buffer[128];
  memset(buffer, 0, 128);

  int counter = 0;
  while (1) {
    struct sockaddr_in src;
    socklen_t src_len = sizeof(src);
    memset(&src, 0, sizeof(src));

    // 阻塞住接受消息
    int sz = recvfrom(sockfd, buffer, 128, 0, (sockaddr *)&src, &src_len);
    if (sz > 0) {
      buffer[sz] = 0;
      printf("Get Message %d:\n %s\n", counter++, buffer);
      printf("get IP %s \n", inet_ntoa(src.sin_addr));
      printf("get Port %d \n\n", ntohs(src.sin_port));

      struct sockaddr_in sent;
      socklen_t sent_len = sizeof(sent);
      memset(&sent, 0, sizeof(sent));
      sent.sin_family = AF_INET;
      sent.sin_addr = src.sin_addr;
      sent.sin_port = src.sin_port;

      char send_buf[20] = "I am Robot!";

      sendto(sockfd, send_buf, strlen(send_buf), 0, (sockaddr *)&sent,
             sent_len);
    } else {
      puts("timeout");
    }
  }

  close(sockfd);
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
