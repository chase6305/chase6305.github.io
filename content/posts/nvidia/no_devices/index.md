---
title: '关于 nvidia-smi: no devices were found 解决方案'
date: 2025-11-15
lastmod: 2026-09-05
draft: false
tags: ["NVIDIA", "Linux", "Troubleshooting"]
categories: ["系统与工具"]
authors: ["chase"]
summary: "结合 RTX 5070 Ti 历史排障记录，说明 Blackwell 开放内核模块要求，并按 PCI、模块、签名和固件检查设备识别。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "结合 RTX 5070 Ti 历史排障记录，说明 Blackwell 开放内核模块要求，并按 PCI、模块、签名和固件检查设备识别。"
contentLanguage: "zh-CN"
reading_prerequisites: "Ubuntu 驱动与内核日志"
reading_focus: "历史版本不等于通用安装清单，先核对架构要求和实际加载模块。"
related_posts:
  - "/posts/cuda/gcc"
  - "/posts/vscode/gdm"
---

## 结论先行：先核对架构与内核模块类型

这是一份 2025 年 11 月的 RTX 5070 Ti 排障记录，下面的内核、驱动版本和输出均属于当时环境，不是今天所有机器的安装清单。Blackwell GPU 要求使用 NVIDIA 的开放内核模块，不能把 `-open` 仅理解为可选性能优化；它也不等于 Nouveau，用户态驱动仍需匹配。

先只读检查，再决定安装方案：

```bash
lspci -nnk -d 10de:
uname -r
modinfo -F license nvidia
cat /proc/driver/nvidia/version
journalctl -k -b | rg -i 'nvidia|nvrm|nouveau|verification|firmware'
```

如果 PCI 设备不可见，先查硬件、虚拟机直通或 BIOS；如果内核日志报告模块签名失败，检查 Secure Boot 与签名流程；如果日志明确要求 open kernel modules，再修正模块类型。不要仅凭一句 `No devices were found` 判断所有机器的根因相同。

参考：[NVIDIA 内核模块类型与硬件要求](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/latest/kernel-modules.html)。

## Ubuntu 22.04 下的历史排障记录

### 问题背景

最近在Ubuntu 22.04系统上安装NVIDIA RTX 5070 Ti 显卡驱动时，遇到了一系列挑战。尽管按照常规方法安装了官方推荐的驱动，但`nvidia-smi`始终显示"No devices were found"。经过几天的摸索和多次重装，暂时找到了解决方案。

### 系统环境

- **操作系统**: Ubuntu 22.04 LTS
- **内核版本**: 6.8.0-87-generic
- **显卡**: NVIDIA RTX 5070 Ti
- **多显卡配置**: 系统同时配备了NVIDIA RTX 5070 Ti 和 AMD集成显卡

### 问题排查过程

#### 1. 初始硬件检测

```text
# 检查系统内核版本
chase@chase:~$ uname -r
6.8.0-87-generic

# 查看显卡硬件信息
chase@chase:~$ lspci | grep -i vga
01:00.0 VGA compatible controller: NVIDIA Corporation Device 2c05 (rev a1)
79:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Device 13c0 (rev c9)
```

从输出可以看到，系统正确识别到了NVIDIA GPU（设备 ID: 2c05）和AMD集成显卡。

#### 2. 可用驱动检测

```text
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

#### 3. 初次安装尝试

按照官方推荐安装580版本驱动：

```bash
sudo apt-get install nvidia-driver-580
```

安装完成后，检查驱动状态：

```bash
nvidia-smi
```

结果令人失望：

```text
No devices were found
```

### 问题分析与解决方案

#### 根本原因

经过多次尝试和排查，发现问题可能源于以下几个方面：

1. **模块类型不匹配**：Blackwell 需要开放内核模块，不能使用专有内核模块。
2. **加载与固件错误**：需要以内核日志确认，不从安装成功推断驱动已经接管设备。
3. **Secure Boot 与签名**：无论开放还是专有模块，签名策略都可能影响加载；没有日志证据时不能认定它是本次原因。

#### 最终解决方案

使用开源版本的580驱动成功解决问题：

```bash
# 卸载之前安装的驱动（如有）
sudo apt-get remove nvidia-driver-580

# 安装开源版本的580驱动
sudo apt-get install nvidia-driver-580-open
```

安装完成后，重启系统并验证：

### 成功验证

```text
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

### 经验总结

1. **先核对架构要求**：Blackwell 对开放内核模块的要求不是一般性的“新卡优先尝试”。
2. **核对发行版推荐结果**：软件源元数据与硬件要求不一致时，以设备支持表和内核日志继续定位。
3. **多显卡不等于冲突**：分别确认设备绑定的驱动和显示/计算用途。
4. **记录完整版本组合**：本例证明当时的 580.95.05-open 组合可用，不能由此推断所有 570 或 580 包的表现。

### CUDA Toolkit 是另一个安装问题

不要同时照抄发行版 Toolkit 包和 `.run` 安装器两条路线。选择一种与项目要求匹配的安装方式；若使用包含驱动的安装器，避免覆盖已经验证的系统驱动。`nvidia-smi` 里的 CUDA Version 不是本地 `nvcc` 版本。下面保留当时独立安装 CUDA 12.8 后的版本记录。

```bash
# 仅在需要本地 CUDA 编译时检查 Toolkit
nvcc --version

```

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Jan_15_19:20:09_PST_2025
Cuda compilation tools, release 12.8, V12.8.61
Build cuda_12.8.r12.8/compiler.35404655_0
```

### 结论

本例通过匹配 Blackwell 所需的开放内核模块恢复设备识别。其他机器仍应按 PCI 枚举、模块加载、固件、签名和用户态库逐层检查；修复后除 `nvidia-smi` 外，还应运行实际应用的最小 GPU 测试。


## 阅读自测与验收

- 把 PCI 识别、内核模块加载、模块签名和 nvidia-smi 分开检查，保留原始日志以及驱动安装来源。
- 修复后在目标内核和重启后的会话中复测；安装命令成功与驱动实际绑定硬件是不同的验收条件。
