---
title: '关于 nvidia-smi: no devices were found 解决方案'
date: 2025-11-15
lastmod: 2025-11-15
draft: false
tags: ["nvidia", "bug"]
categories: ["编程技术"]
authors: ["chase"]
summary: "关于 nvidia-smi: no devices were found 解决方案"
showToc: true
TocOpen: true
hidemeta: false
comments: false
---

# 解决Ubuntu 22.04下RTX 5070显卡驱动安装的曲折历程

## 问题背景

最近在Ubuntu 22.04系统上安装NVIDIA RTX 5070显卡驱动时，遇到了一系列挑战。尽管按照常规方法安装了官方推荐的驱动，但`nvidia-smi`始终显示"No devices were found"。经过几天的摸索和多次重装，暂时找到了解决方案。

## 系统环境

- **操作系统**: Ubuntu 22.04 LTS
- **内核版本**: 6.8.0-87-generic
- **显卡**: NVIDIA RTX 5070 Ti
- **多显卡配置**: 系统同时配备了NVIDIA RTX 5070和AMD集成显卡

## 问题排查过程

### 1. 初始硬件检测

```bash
# 检查系统内核版本
chase@chase:~$ uname -r
6.8.0-87-generic

# 查看显卡硬件信息
chase@chase:~$ lspci | grep -i vga
01:00.0 VGA compatible controller: NVIDIA Corporation Device 2c05 (rev a1)
79:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Device 13c0 (rev c9)
```

从输出可以看到，系统正确识别到了NVIDIA RTX 5070（设备ID: 2c05）和AMD集成显卡。

### 2. 可用驱动检测

```bash
chase@chase:~$ ubuntu-drivers devices
== /sys/devices/pci0000:00/0000:00:01.1/0000:01:00.0 ==
modalias : pci:v000010DEd00002C05sv00001043sd000089F4bc03sc00i00
vendor   : NVIDIA Corporation
driver   : nvidia-driver-580-server-open - distro non-free
driver   : nvidia-driver-570-server-open - distro non-free
driver   : nvidia-driver-570 - distro non-free
driver   : nvidia-driver-570-open - distro non-free
driver   : nvidia-driver-580 - distro non-free recommended
driver   : nvidia-driver-570-server - distro non-free
driver   : nvidia-driver-580-open - distro non-free
driver   : nvidia-driver-580-server - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builtin
```

系统推荐安装`nvidia-driver-580`版本驱动。

### 3. 初次安装尝试

按照官方推荐安装580版本驱动：

```bash
sudo apt-get install nvidia-driver-580
```

安装完成后，检查驱动状态：

```bash
nvidia-smi
```

结果令人失望：
```
No devices were found
```

## 问题分析与解决方案

### 根本原因

经过多次尝试和排查，发现问题可能源于以下几个方面：

1. **新硬件兼容性问题**: RTX 5070是相对较新的显卡，标准闭源驱动可能存在兼容性问题
2. **内核模块加载失败**: 闭源驱动可能无法正确加载内核模块
3. **Secure Boot冲突**: 在某些情况下，Secure Boot可能会阻止专有驱动加载

### 最终解决方案

使用开源版本的580驱动成功解决问题：

```bash
# 卸载之前安装的驱动（如有）
sudo apt-get remove nvidia-driver-580

# 安装开源版本的580驱动
sudo apt-get install nvidia-driver-580-open
```

安装完成后，重启系统并验证：

## 成功验证

```bash
chase@chase:~$ nvidia-smi
Sat Nov 15 15:27:11 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5070 Ti     Off |   00000000:01:00.0  On |                  N/A |
|  0%   35C    P8             20W /  300W |     921MiB /  16303MiB |      6%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2044      G   /usr/lib/xorg/Xorg                      392MiB |
|    0   N/A  N/A            2191      G   /usr/bin/gnome-shell                     80MiB |
|    0   N/A  N/A            4977      G   ...rack-uuid=3190708988185955192        386MiB |
+-----------------------------------------------------------------------------------------+
```

## 经验总结

1. **新硬件优先尝试开源驱动**: 对于像RTX 5070这样的新发布硬件，开源版本驱动往往具有更好的兼容性
2. **不要盲目相信"recommended"**: 虽然系统推荐闭源驱动，但实际兼容性可能不如开源版本
3. **多显卡环境复杂性**: 在NVIDIA和AMD显卡共存的环境中，驱动冲突的可能性更高
4. **版本选择很重要**: 580版本驱动相比570版本对新硬件支持更好

## 后续优化建议

```bash
# 安装CUDA工具包（如需要）
sudo apt install nvidia-cuda-toolkit

# 这里我是安装cuda 12.8
sudo sh cuda_12.8.0_570.86.10_linux.run

# 验证CUDA安装
nvcc --version

```

```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Jan_15_19:20:09_PST_2025
Cuda compilation tools, release 12.8, V12.8.61
Build cuda_12.8.r12.8/compiler.35404655_0
```

## 结论

在Ubuntu 22.04上安装RTX 5070显卡驱动时，如果遇到`nvidia-smi`显示"No devices were found"的问题，尝试使用开源版本的驱动（如`nvidia-driver-580-open`）暂时能够解决问题。


