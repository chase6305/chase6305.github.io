---
title: 'Linux固定网口IP的方法'
date: 2025-02-21
lastmod: 2026-09-28
draft: false
tags: ["Linux Networking", "IP Configuration"]
categories: ["系统与工具"]
authors: ["chase"]
summary: "配置 Ubuntu 静态 IP 时核对网络接口、renderer、路由与 DNS，使用 Netplan 检查和回滚机制降低断连风险。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "配置 Ubuntu 静态 IP 时核对网络接口、renderer、路由与 DNS，使用 Netplan 检查和回滚机制降低断连风险。"
contentLanguage: "zh-CN"
reading_prerequisites: "Linux 网络接口与路由"
reading_focus: "先区分外网网卡和机器人直连网卡，远程修改前保留回退通道。"
related_posts:
  - "/posts/network-protocol/c++_udp"
  - "/posts/network-protocol/c++_tcp"
---

Ubuntu 固定 IP 配置应先确认网卡名称与当前网络管理方式。下面使用 Netplan；IP、网关和 DNS 均为示例，实际值需要与局域网规划一致，避免与 DHCP 地址池或其他设备冲突。

## 1. 查看当前网络

```bash
ip -br address
ip route
ls /etc/netplan
sudo netplan get
```

编辑当前实际生效的 YAML，避免新增多个文件重复配置同一接口。桌面系统可能由 NetworkManager 管理；只连接机器人设备的网口通常不需要默认网关。

## 2. 配置示例

### 机器人直连：配置地址即可，不添加默认网关

假设电脑的 Wi-Fi 已负责上网，机器人使用固定地址 `192.168.50.10/24`，有线接口 `enp3s0` 连接机器人。可将该有线接口设为 `192.168.50.2/24`：

```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    enp3s0:
      dhcp4: false
      addresses:
        - 192.168.50.2/24
```

这个示例只描述机器人网口，不应覆盖现有 Wi-Fi 或其他接口的配置。`renderer` 仍以现有管理方式为准。两端都在 `192.168.50.0/24`，同网段通信不需要经过路由器，也不需要为了连接这个 IP 配置 DNS。

| 接口职责 | 地址示例 | 默认路由 |
| --- | --- | --- |
| 电脑机器人网口 | `192.168.50.2/24` | 不添加 |
| 机器人 | `192.168.50.10/24` | 此直连通信不依赖网关 |
| 电脑联网网口或 Wi-Fi | 保留当前网络分配 | 保留联网网络提供的路由 |

机器人子网应避开 Wi-Fi、VPN 和容器网络使用的网段。两块网卡都配置到同一个 `192.168.1.0/24`，往往比“没设 DNS”更容易造成机器人流量走错接口；先核对实际路由，不要只改网关优先级。

### 该有线接口需要经路由器上网时

假设接口为 `enp3s0`，使用 systemd-networkd 管理。保留现有 renderer，只有明确要切换网络管理方式时才修改它：

```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    enp3s0:
      dhcp4: false
      addresses:
        - 192.168.1.100/24
      routes:
        - to: default
          via: 192.168.1.1
      nameservers:
        addresses:
          - 192.168.1.1
```

这里使用 `routes` 表达默认路由。它与上面的机器人直连配置是两种用途，不要在不同 YAML 文件里同时给同一接口套用两份示例。

## 3. 检查并试用配置

```bash
sudo netplan generate
sudo netplan try
```

`generate` 检查并生成配置；`try` 提供限时确认与回退机制，但回退仍需复核。通过 SSH 修改网络时，应保留备用访问手段，确认远程连接可用后再接受配置。[Netplan 官方示例](https://netplan.readthedocs.io/en/stable/examples/)

## 4. 分层验证

```bash
ip -br address
ip route
ping -c 3 192.168.1.1
resolvectl status
```

接口地址正确、能到达网关、DNS 正常是三个不同检查。目标设备不响应 ping 时，也应结合其 ICMP 设置和实际服务端口判断。

对前述机器人直连场景，更有用的是先询问内核将如何访问机器人：

```bash
ip route get 192.168.50.10
ip neigh show dev enp3s0
```

第一条输出应包含预期的 `dev enp3s0` 和 `src 192.168.50.2`。它执行的是路由查询，不会证明网线和目标设备已经连通。[`ip route get` 的定义](https://github.com/iproute2/iproute2/blob/main/man/man8/ip-route.8.in)与简单列出主路由表不同，存在 VPN 或策略路由时尤其值得检查。

若访问过目标后，邻居表仍是 `INCOMPLETE` 或 `FAILED`，优先排查网线、接口、地址冲突、子网和设备供电；若邻居已解析、应用仍连接失败，再核对服务监听地址、端口与防火墙。没有发起过通信时邻居表为空是正常现象。

使用 Netplan 的系统不需要把 `systemctl restart networking` 当作通用收尾步骤；服务名称和管理方式取决于实际后端。


## 阅读自测与验收

- 修改前记录接口名、地址、路由和 DNS，修改后分别验证同网段、网关和域名解析，而不是只 ping 一个地址。
- 远程修改必须保留可用的回退通道；配置语法通过不保证新地址无冲突或管理连接仍可达。
