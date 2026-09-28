---
title: 'C++ UDP 通信：报文边界、MTU、序号与数据年龄'
date: 2021-08-08
lastmod: 2026-09-28
draft: false
tags: ["C++", "UDP", "Network Programming"]
categories: ["系统与工具"]
authors: ["chase"]
summary: "实现带超时与截断检查的 Linux C++17 UDP 通信，解释接收缓冲与路径 MTU 的区别，并设计机器人状态流的序号、会话与有效期。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "实现带超时与截断检查的 Linux C++17 UDP 通信，解释接收缓冲与路径 MTU 的区别，并设计机器人状态流的序号、会话与有效期。"
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
            throw std::runtime_error("receive wait timed out (SO_RCVTIMEO=5 s)");
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

两个终端依次执行 `./udp_demo server` 和 `./udp_demo client`，在 5 秒内启动客户端。预期客户端输出 `Hello, UDP!`。若某次接收等待超时，服务端退出后需要重新启动。`SO_RCVTIMEO` 限制一次阻塞接收的等待；遇到 `EINTR` 后重新调用，以及系统调度，都可能延长整个请求的耗时。若要限制请求总预算，应使用单调时钟的绝对截止时间，参见 [TCP 整帧截止时间]({{< relref "/posts/network-protocol/c++_tcp" >}})。

`recvfrom` 返回 0 表示收到零长度数据报，并不等于 TCP 的连接关闭。数组恰好收满时不能写 `buffer[buffer.size()]`；本例始终按长度处理数据，不补写终止符。

## 1500 字节缓冲区为什么不是 1500 字节网络载荷

接收缓冲区大小是应用愿意接受的最大数据报尺寸；路径 MTU 是网络允许的 IP 包尺寸，两者不在同一层。以路径 MTU 恰为 1500 字节、没有 IP 可选字段或扩展头为例：

| 协议 | IP 头 | UDP 头 | 无 IP 分片时的 UDP 载荷预算 |
| --- | ---: | ---: | ---: |
| IPv4 | 20 B | 8 B | 1472 B |
| IPv6 | 40 B | 8 B | 1452 B |

载荷预算还包含应用自己的序号、时间戳、校验与封装字段。隧道、不同链路或扩展头会改变有效预算，不能把这两个数当作所有网络通用上限。回环接口允许的报文尺寸也不能代表真实网卡与路径。

发送超过路径限制的报文，可能触发分片或返回错误，具体取决于协议、系统及路径 MTU 发现配置。即使接收端重组后能一次读到完整 UDP 报文，也不说明途中没有分片；任一分片丢失都可能使整个报文无法交付。UDP 应用应尽量避免依赖 IP 分片，并处理过长报文与拥塞问题。[IETF UDP 使用指南 §3.2](https://www.rfc-editor.org/rfc/rfc8085.html#section-3.2)

## 机器人状态流：序号与时间戳分别解决什么

假设一条消息包含协议版本、会话标识、序号、采集时间和关节状态。接收顺序不一定是采集顺序，应先验证消息长度与版本，再决定新旧关系和数据年龄。

| 字段或检查 | 用途 | 单独不能证明什么 |
| --- | --- | --- |
| 会话标识 | 区分设备重启或新一轮流 | 不等于身份认证 |
| 单调增长序号 | 识别重复、乱序和缺口 | 不能单独给出真实数据年龄 |
| 采集时间戳 | 计算采集到消费的延迟 | 跨设备时仍需时钟对齐 |
| 接收端单调时钟 | 计算本地排队和无消息时长 | 不包含消息到达前已经发生的延迟 |
| 有效期与数据有效标记 | 拒绝过期或不可信输入 | 拒绝之后仍需定义无有效输入时的行为 |

序号绕回不能只用普通 `incoming > previous`。对 32 位无符号序号，先按模 `2^32` 计算差值；在两次可比较记录相距不到半个序号空间的前提下，差值位于 `1…2^31−1` 表示更新，0 表示重复。恰好相差半个空间时关系不确定；其余值通常作为旧数据处理。这个比较规则的前提与边界见 [RFC 1982 序号算术](https://www.rfc-editor.org/rfc/rfc1982.html#section-3.2)。

例如从 `4294967295` 到 `0` 的模差为 1，应接受为更新。设备重启却不能简单按绕回处理：新会话应重新建立序号状态，并确认时间基准与标定状态。序号连续但采集缓慢的数据仍可能过期；刚到达的数据也可能已经在发送端积压很久。

对于关节状态，可能允许跳过旧报文并统计丢失；对于“执行抓取”等带副作用的命令，需要另外定义确认、幂等和重试范围。不要把状态流的“保留最新”直接套到顺序任务上。关于队列内进一步积压的影响，可继续读[队列与数据新鲜度]({{< relref "/posts/queue" >}})。

## 协议设计边界

- 单个数据报应受应用层长度约束；1500 字节缓冲区只是本例限制，不代表所有网络的安全 UDP 载荷上限。
- 若业务需要可靠性，增加序列号、确认、去重、重试预算与拥塞控制，或选择已有可靠传输协议。
- 广播另需 `SO_BROADCAST` 和正确的子网广播地址；只在明确授权的局域网设备发现流程中使用，并限制发送频率。
- 本例没有认证和加密，不能直接作为机器人运动指令通道。

参考：[Linux socket 超时选项](https://man7.org/linux/man-pages/man7/socket.7.html)、[Linux udp(7)](https://man7.org/linux/man-pages/man7/udp.7.html)、[Linux recv(2)](https://man7.org/linux/man-pages/man2/recv.2.html)。


## 阅读自测与验收

- 测试零长度、正常长度和超出接收缓冲区的数据报；零长度是合法消息，而截断消息不应继续按完整协议解析。
- 应用需要重传时必须另行设计序号、超时、去重与幂等；一次回显成功不说明 UDP 提供可靠或有序交付。
