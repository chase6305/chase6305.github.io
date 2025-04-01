---
title: 'Linux固定网口IP的方法'
date: 2025-02-21
lastmod: 2025-02-21
draft: false
tags: ["IP"]
categories: ["Protocol"]
authors: ["chase"]
summary: "Linux固定网口IP的方法"
showToc: true
TocOpen: true
hidemeta: false
comments: false
---


以下是关于在 Ubuntu 上配置固定 IP 地址的步骤：

## 1. 打开网络配置文件

使用文本编辑器打开网络配置文件：

```bash
sudo nano /etc/netplan/01-netcfg.yaml
```

## 2. 编辑网络配置文件

在文件中添加或修改以下内容：
```bash
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: no
      addresses:
        - 192.168.1.100/24
      gateway4: 192.168.1.1
      nameservers:
        addresses:
          - 8.8.8.8
          - 8.8.4.4
 ```
请根据实际情况替换以下内容：

eth0：你的网络接口名称，可以使用 ip a 命令查看。
192.168.1.100/24：你想要设置的固定 IP 地址和子网掩码。
192.168.1.1：你的网关地址。
8.8.8.8 和 8.8.4.4：DNS 服务器地址。


## 3. 应用配置
保存文件并退出编辑器，然后运行以下命令应用配置：

```bash
sudo netplan apply
```
## 4. 验证配置
使用以下命令验证 IP 地址是否已正确配置：
```bash
ip a
```
你应该会看到你的网络接口已经配置了你设置的固定 IP 地址。

## 5. 重启网络服务（可选）
如果配置没有立即生效，可以尝试重启网络服务：
```bash
sudo systemctl restart networking
```
这样，你的 Ubuntu 系统就会使用固定 IP 地址进行网络连接。
