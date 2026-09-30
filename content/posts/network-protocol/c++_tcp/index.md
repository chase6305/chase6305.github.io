---
title: 'C++ TCP 通信：消息边界、完整读写与截止时间'
date: 2021-08-08
lastmod: 2026-09-30
draft: false
tags: ["C++", "TCP", "Network Programming"]
categories: ["系统与工具"]
authors: ["chase"]
summary: "实现带长度前缀的 Linux C++17 TCP 通信，区分空消息与截断连接，用整帧截止时间阻止慢速收包无限等待，并澄清机器人指令确认的含义。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "实现带长度前缀的 Linux C++17 TCP 通信，区分空消息与截断连接，用整帧截止时间阻止慢速收包无限等待，并澄清机器人指令确认的含义。"
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

<figure class="article-figure">
{{< post-image src="assets/tcp-framing.webp" alt="两条带四字节长度头的消息，共十三字节，按二、六、五字节分三次接收，再按长度重新组成完整消息" >}}
<figcaption><span class="article-figure__number">图 1</span><span class="article-figure__text">长度字段只统计正文。接收块可以从头部中间切开，也可以跨越两条消息；图中分块只是可能情况，与实际 TCP 报文段边界无须一致。每帧的头部和正文共享一个截止时间。</span></figcaption>
</figure>

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

## 接收超时：限制的是一次等待，还是整条消息

阻塞版本把消息读全，却没有限制等多久。假设每次等待读取都重新获得 100 ms，对方每 40 ms 发来一个字节，每次调用都可能成功，一帧仍能拖很久。给头部和正文分别启动计时，也会把一帧的总预算扩大。

完整实现见 [tcp_deadline.cpp](tcp_deadline.cpp)。它保留相同的 4 字节长度协议，增加 `read_frame_until(fd, deadline)`：在调用前用 `std::chrono::steady_clock` 计算一次绝对截止时间，头部、正文、短读重试和 `EINTR` 重试都使用这个时间点。`poll` 每次只等待剩余预算；收到就绪通知后，再用 `MSG_DONTWAIT` 读取，防止就绪状态变化后重新陷入阻塞。[Linux poll 文档](https://man7.org/linux/man-pages/man2/poll.2.html)、[Linux recv 文档](https://man7.org/linux/man-pages/man2/recv.2.html)

| 接收结果 | 返回或处理 | 应用含义 |
| --- | --- | --- |
| 完整头部，声明正文为 0 字节 | 空字符串 | 有效空帧，仍消耗了 4 字节头部 |
| 新帧头部一个字节都没收到便 EOF | `std::nullopt` | 在帧边界正常结束输入 |
| 头部不满 4 字节便 EOF | 异常 | 头部被截断 |
| 长度有效，正文不足便 EOF | 异常 | 正文被截断，包括正文一个字节都没有的情况 |
| 声明长度超过 1 MiB | 异常 | 分配正文内存之前拒绝 |
| 整帧截止时间到达 | `Timeout` | 丢弃本次不完整消息并关闭连接 |
{.table-readable}

这个实现只有一个读取者，调用期间描述符保持有效。发生超时后，它没有保留已经读走的半帧，因此调用者应关闭连接。直接捕获异常后再调用一次，会把残留正文误当作下一帧的长度；需要跨调用恢复时，应另写保存头部、正文偏移和截止时间的解析状态机。

### 在本地验证边界条件

下载文件后，在 Linux 上运行：

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -pthread tcp_deadline.cpp -o tcp_deadline
./tcp_deadline
```

程序通过本地 `socketpair` 验证：包含零字节的正文、空消息、连续多帧、逐字节发送、边界 EOF、半头部、缺少正文、半正文、超长声明，以及慢速滴入、头部和正文共享预算、已经过期的截止时间。测试只检查字节流和协议逻辑，不评估 TCP 丢包重传、跨主机网络或实时性能。

还要分清两个限制：本例只为**接收一帧**设预算，连接、发送、业务执行各有自己的预算；操作系统调度也可能让线程晚于截止时间才恢复运行。这里保证过期后不再把读到的内容作为成功消息返回，不承诺线程必在某一微秒内返回。`poll` 的时间粒度与调度限制见其官方手册。

## 机器人指令：发出、接收、执行完成是三个状态

`send` 返回成功只表示相应字节被本地系统接受，不能据此认定机器人已经收到或完成动作。即使收到应用层 `ACK`，也要看协议定义：它可能只表示命令通过校验或进入队列。

一个任务协议可以分别定义 `accepted`、`running`、`completed`、`failed`，并携带会话标识和命令序号。连接在响应返回前断开时，动作可能已经执行；带机械副作用的命令不能仅因“没收到 ACK”就盲目重发。需要由服务端按命令标识查询或去重，并写清去重记录的保存范围与过期行为。

状态流和任务流也应区别处理：旧关节状态可能需要丢弃，已接受的顺序任务则不能任意跳过。队列容量与数据年龄的关系可继续读[无锁队列中的背压与新鲜度]({{< relref "/posts/queue" >}})。

## 从示例走向实际服务

- `listen` 的 backlog 是待接受连接队列的相关参数，不是“最多允许多少进程”。
- 第一份程序演示完整读写，第二份补充整帧接收截止时间；实际服务仍需连接与发送预算、并发管理和取消机制。
- 多条消息可以复用同一连接，但要在完整帧边界区分正常 EOF 与半帧截断。
- TCP 提供可靠有序的传输，不提供身份认证、应用级幂等或机密性；按应用需要增加 TLS 与协议校验。

参考：[POSIX recv](https://pubs.opengroup.org/onlinepubs/9799919799/functions/recv.html)、[POSIX send](https://pubs.opengroup.org/onlinepubs/9799919799/functions/send.html)。


## 阅读自测与验收

- 让客户端分多次发送头部与正文，确认服务端仍能解析一帧；一次 recv 返回的数据量不能代表应用消息边界。
- 测试对端提前关闭和超长声明长度，确认不会无界分配或把不完整内容作为完整响应。
