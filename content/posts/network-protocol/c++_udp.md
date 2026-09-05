---
title: 'C++ 关于UDP通讯的示例'
date: 2021-08-08
lastmod: 2026-09-05
draft: false
tags: ["C++", "UDP", "Network Programming"]
categories: ["系统与工具"]
authors: ["chase"]
summary: "实现带接收超时和截断检查的 Linux C++17 UDP 回环通信，说明报文边界、零长度消息与广播限制。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "实现带接收超时和截断检查的 Linux C++17 UDP 回环通信，说明报文边界、零长度消息与广播限制。"
contentLanguage: "zh-CN"
reading_prerequisites: "C++17 与 Linux socket"
reading_focus: "先确认一次完整数据报，再按业务需求设计序列号、去重和重试预算。"
related_posts:
  - "/posts/network-protocol/c++_tcp"
  - "/posts/network-protocol/fixed_IP"
---

## UDP 保留报文边界，但不保证送达

UDP 是无连接的数据报传输协议。它不保证送达、顺序或去重；“没有重传等待”不等于每个场景都更快，更不等于丢包对业务没有影响。

下面使用 **Linux / C++17 / IPv4** 实现一次本机请求—响应，默认 `127.0.0.1:5001`。不向局域网广播，避免示例程序意外干扰其他设备。

## 完整程序：udp_demo.cpp

客户端对 UDP socket 调用 `connect` 只是设置默认对端并过滤接收来源，不会建立 TCP 式握手。服务端使用 `recvfrom` 获取发送者，再回复该地址。

```cpp
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <array>
#include <cerrno>
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

int main(int argc, char** argv) {
    try {
        if (argc != 2 || (std::string(argv[1]) != "server" &&
                          std::string(argv[1]) != "client"))
            throw std::runtime_error("usage: udp_demo server|client");
        const bool server = std::string(argv[1]) == "server";
        Socket socket(::socket(AF_INET, SOCK_DGRAM, 0));

        timeval timeout{5, 0};
        check(::setsockopt(socket.fd, SOL_SOCKET, SO_RCVTIMEO,
                          &timeout, sizeof(timeout)), "setsockopt");
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(5001);
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        const std::string message = "Hello, UDP!";

        if (server) {
            check(::bind(socket.fd, reinterpret_cast<sockaddr*>(&address),
                         sizeof(address)), "bind");
            std::cout << "Listening on 127.0.0.1:5001 (5 s timeout)" << std::endl;
        } else {
            check(::connect(socket.fd, reinterpret_cast<sockaddr*>(&address),
                            sizeof(address)), "connect");
            ssize_t sent;
            do { sent = ::send(socket.fd, message.data(), message.size(), 0); }
            while (sent < 0 && errno == EINTR);
            check(static_cast<int>(sent), "send");
            if (static_cast<std::size_t>(sent) != message.size())
                throw std::runtime_error("incomplete datagram send");
        }

        std::array<char, 1500> buffer{};
        sockaddr_in peer{};
        socklen_t peer_size = sizeof(peer);
        ssize_t received;
        do {
            received = ::recvfrom(socket.fd, buffer.data(), buffer.size(),
                                  MSG_TRUNC, reinterpret_cast<sockaddr*>(&peer),
                                  &peer_size);
        } while (received < 0 && errno == EINTR);
        if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
            throw std::runtime_error("receive timed out after 5 s");
        check(static_cast<int>(received), "recvfrom");
        // Linux MSG_TRUNC 返回原始数据报长度；截断的消息不得继续处理。
        if (static_cast<std::size_t>(received) > buffer.size())
            throw std::runtime_error("datagram exceeds the application limit");

        if (server) {
            ssize_t sent;
            do {
                sent = ::sendto(socket.fd, buffer.data(), received, 0,
                                reinterpret_cast<sockaddr*>(&peer), peer_size);
            } while (sent < 0 && errno == EINTR);
            check(static_cast<int>(sent), "sendto");
            if (sent != received) throw std::runtime_error("incomplete reply");
        } else {
            const std::string reply(buffer.data(), received);
            if (reply != message) throw std::runtime_error("reply mismatch");
            std::cout << reply << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
```

## 编译与验证

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic udp_demo.cpp -o udp_demo
```

两个终端依次执行 `./udp_demo server` 和 `./udp_demo client`，在 5 秒内启动客户端。预期客户端输出 `Hello, UDP!`。若超时，服务端退出后需要重新启动，不会永久等待。

`recvfrom` 返回 0 表示收到零长度数据报，并不等于 TCP 的连接关闭。数组恰好收满时不能写 `buffer[buffer.size()]`；本例始终按长度处理数据，不补写终止符。

## 协议设计边界

- 单个数据报应受应用层长度约束；1500 字节缓冲区只是本例限制，不代表所有网络的安全 UDP 载荷上限。
- 若业务需要可靠性，增加序列号、确认、去重、重试预算与拥塞控制，或选择已有可靠传输协议。
- 广播另需 `SO_BROADCAST` 和正确的子网广播地址；只在明确授权的局域网设备发现流程中使用，并限制发送频率。
- 本例没有认证和加密，不能直接作为机器人运动指令通道。

参考：[Linux udp(7)](https://man7.org/linux/man-pages/man7/udp.7.html)、[Linux recv(2)](https://man7.org/linux/man-pages/man2/recv.2.html)。


## 阅读自测与验收

- 测试零长度、正常长度和超出接收缓冲区的数据报；零长度是合法消息，而截断消息不应继续按完整协议解析。
- 应用需要重传时必须另行设计序号、超时、去重与幂等；一次回显成功不说明 UDP 提供可靠或有序交付。
