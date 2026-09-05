---
title: 'C++ 关于TCP通讯的示例'
date: 2021-08-08
lastmod: 2026-09-05
draft: false
tags: ["C++", "TCP", "Network Programming"]
categories: ["系统与工具"]
authors: ["chase"]
summary: "实现带长度前缀的 Linux C++17 TCP 请求响应，处理短读短写、截断消息、长度上限与 socket 生命周期。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "实现带长度前缀的 Linux C++17 TCP 请求响应，处理短读短写、截断消息、长度上限与 socket 生命周期。"
contentLanguage: "zh-CN"
reading_prerequisites: "C++17 与 Linux socket"
reading_focus: "先在回环地址编译验证，完整消息边界和超时策略必须由应用定义。"
related_posts:
  - "/posts/network-protocol/c++_udp"
  - "/posts/queue"
---

## TCP 是字节流，不是消息队列

一次 `send` 不对应一次 `recv`：数据可能被拆开或合并，调用也可能只处理部分字节。应用需要约定消息边界，并区分对端正常关闭、截断消息与系统调用错误。

下面给出 **Linux / C++17 / IPv4** 的单次请求—响应示例：服务端只监听 `127.0.0.1:8888`，客户端发送“4 字节网络字节序长度 + 消息体”，服务端完整接收后原样回复。消息上限为 1 MiB，不是面向公网的生产服务。

## 完整程序：tcp_demo.cpp

同一份程序通过 `server` 或 `client` 参数切换角色，便于保证两端协议一致。

```cpp
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>

struct Socket {
    int fd;
    explicit Socket(int value) : fd(value) {
        if (fd < 0) throw std::system_error(errno, std::generic_category(), "socket");
    }
    ~Socket() { ::close(fd); }
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
};

void check(int result, const char* operation) {
    if (result < 0)
        throw std::system_error(errno, std::generic_category(), operation);
}

void send_all(int fd, const char* data, std::size_t size) {
    while (size > 0) {
        const auto n = ::send(fd, data, size, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        check(static_cast<int>(n), "send");
        if (n == 0) throw std::runtime_error("send made no progress");
        data += n;
        size -= static_cast<std::size_t>(n);
    }
}

void recv_exact(int fd, char* data, std::size_t size) {
    while (size > 0) {
        const auto n = ::recv(fd, data, size, 0);
        if (n < 0 && errno == EINTR) continue;
        check(static_cast<int>(n), "recv");
        if (n == 0) throw std::runtime_error("EOF before the frame was complete");
        data += n;
        size -= static_cast<std::size_t>(n);
    }
}

constexpr std::uint32_t max_size = 1024 * 1024;

void send_frame(int fd, const std::string& body) {
    if (body.size() > max_size) throw std::runtime_error("message too large");
    const std::uint32_t length = htonl(static_cast<std::uint32_t>(body.size()));
    send_all(fd, reinterpret_cast<const char*>(&length), sizeof(length));
    send_all(fd, body.data(), body.size());
}

std::string recv_frame(int fd) {
    std::uint32_t length = 0;
    recv_exact(fd, reinterpret_cast<char*>(&length), sizeof(length));
    length = ntohl(length);
    if (length > max_size) throw std::runtime_error("message too large");
    std::string body(length, '\0');
    recv_exact(fd, body.data(), body.size());
    return body;
}

int main(int argc, char** argv) {
    try {
        if (argc != 2 || (std::string(argv[1]) != "server" &&
                          std::string(argv[1]) != "client"))
            throw std::runtime_error("usage: tcp_demo server|client");

        Socket socket(::socket(AF_INET, SOCK_STREAM, 0));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(8888);
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

        if (std::string(argv[1]) == "server") {
            const int reuse = 1;
            check(::setsockopt(socket.fd, SOL_SOCKET, SO_REUSEADDR,
                              &reuse, sizeof(reuse)), "setsockopt");
            check(::bind(socket.fd, reinterpret_cast<sockaddr*>(&address),
                         sizeof(address)), "bind");
            check(::listen(socket.fd, 1), "listen");
            std::cout << "Listening on 127.0.0.1:8888" << std::endl;
            int accepted;
            do { accepted = ::accept(socket.fd, nullptr, nullptr); }
            while (accepted < 0 && errno == EINTR);
            check(accepted, "accept");
            Socket peer(accepted);
            send_frame(peer.fd, recv_frame(peer.fd));
        } else {
            check(::connect(socket.fd, reinterpret_cast<sockaddr*>(&address),
                            sizeof(address)), "connect");
            const std::string request = "Hello, framed TCP!";
            send_frame(socket.fd, request);
            const auto reply = recv_frame(socket.fd);
            if (reply != request) throw std::runtime_error("reply mismatch");
            std::cout << reply << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
```

## 编译与运行

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic tcp_demo.cpp -o tcp_demo
```

先在一个终端运行：

```bash
./tcp_demo server
```

再在另一个终端运行：

```bash
./tcp_demo client
```

预期客户端输出 `Hello, framed TCP!`，两端正常退出。程序按明确长度处理字符串，消息体也可以包含零字节，不依赖 `buffer[n] = '\0'`。

## 从示例走向实际服务

- `listen` 的 backlog 是待接受连接队列的相关参数，不是“最多允许多少进程”。
- 本例的阻塞读没有截止时间，不能直接部署到不可信网络；实际服务需要连接/读写超时、总消息预算、并发和取消机制。
- 多条消息可以复用同一连接，但要在完整帧边界区分正常 EOF 与半帧截断。
- TCP 提供可靠有序的传输，不提供身份认证、应用级幂等或机密性；按应用需要增加 TLS 与协议校验。

参考：[POSIX recv](https://pubs.opengroup.org/onlinepubs/9799919799/functions/recv.html)、[POSIX send](https://pubs.opengroup.org/onlinepubs/9799919799/functions/send.html)。


## 阅读自测与验收

- 让客户端分多次发送头部与正文，确认服务端仍能解析一帧；一次 recv 返回的数据量不能代表应用消息边界。
- 测试对端提前关闭和超长声明长度，确认不会无界分配或把不完整内容作为完整响应。
