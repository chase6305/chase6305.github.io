---
title: 'Linux固定网口IP的方法'
date: 2025-02-21
lastmod: 2026-09-05
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

这里使用 `routes` 表达默认路由。若该网口只是与机器人直连，去掉 `routes` 和 `nameservers`，以免抢占联网网卡的默认路由。

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

使用 Netplan 的系统不需要把 `systemctl restart networking` 当作通用收尾步骤；服务名称和管理方式取决于实际后端。


## 阅读自测与验收

- 修改前记录接口名、地址、路由和 DNS，修改后分别验证同网段、网关和域名解析，而不是只 ping 一个地址。
- 远程修改必须保留可用的回退通道；配置语法通过不保证新地址无冲突或管理连接仍可达。
