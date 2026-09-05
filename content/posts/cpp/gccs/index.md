---
title: 在 Ubuntu 上安装和切换多个 GCC 版本
date: 2025-03-07
lastmod: 2026-09-05
draft: false
tags: ["C++", "GCC"]
categories: ["编程开发"]
authors: ["chase"]
summary: "用项目级 CMake 与 CC/CXX 管理多套 GCC，解释 PATH、编译缓存和 alternatives 的边界，避免改坏系统工具链。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "用项目级 CMake 与 CC/CXX 管理多套 GCC，解释 PATH、编译缓存和 alternatives 的边界，避免改坏系统工具链。"
contentLanguage: "zh-CN"
reading_prerequisites: "Linux shell 与 C/C++ 构建"
reading_focus: "每套编译器使用独立构建目录，同时核对 C 与 C++ 编译器。"
related_posts:
  - "/posts/cmake/ExternalProject_Add"
  - "/posts/cpp/gcc"
---

多个 GCC 版本可以并存。对 CMake 项目，显式指定 C 与 C++ 编译器最容易复现，也便于确认构建缓存是否仍指向旧版本。

## 1. 查看发行版提供的版本

以下以 GCC 11 为例，不代表所有 Ubuntu 版本都提供同一组软件包：

```bash
apt-cache policy gcc-11 g++-11
sudo apt update
sudo apt install gcc-11 g++-11
gcc-11 --version
g++-11 --version
```

GCC 与 G++ 应选择同一套工具链。编译器版本和程序运行时加载的 `libstdc++.so.6` 还需分别验证。

## 2. 为一个项目指定编译器

使用新的构建目录，让 CMake 从首次配置时就记录正确的编译器：

```bash
cmake -S . -B build-gcc11 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-11 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-11
cmake --build build-gcc11
```

也可以在首次配置时传入环境变量：

```bash
CC=gcc-11 CXX=g++-11 cmake -S . -B build-gcc11-env
```

`CC` 和 `CXX` 不会覆盖已经写入 `CMakeCache.txt` 的编译器选择。对应行为见 [CMake CXX 环境变量文档](https://cmake.org/cmake/help/latest/envvar/CXX.html)。

## 3. 系统默认命令与 alternatives

确实需要维护全局默认工具链时，先查看已有配置：

```bash
update-alternatives --query gcc
update-alternatives --query g++
```

在尚未独立管理 `g++` 的系统上，可以把它作为 `gcc` 的从属链接，让两者成套切换：

```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
  --slave /usr/bin/g++ g++ /usr/bin/g++-11
sudo update-alternatives --config gcc
```

如果 `g++` 已经作为独立链接组注册，这条命令会发生冲突。先查明现有管理方式；已有两个独立组的系统应分别选择相同版本，或继续采用项目级配置。

## 4. 不要把可执行文件写进 PATH

`PATH` 包含目录，而不是 `/usr/bin/gcc-11` 这样的文件名。下面两种方式各有明确用途：

- 直接执行 `gcc-11` / `g++-11`，选择一次编译所用的命令。
- 设置 `CC` / `CXX`，让支持这些变量的构建系统选择工具链。

在 `PATH` 中追加编译器文件本身不会切换 `gcc` 命令。

## 5. 验证最终结果

```bash
command -v gcc
command -v g++
gcc -dumpfullversion
g++ -dumpfullversion
rg 'CMAKE_(C|CXX)_COMPILER:' build-gcc11/CMakeCache.txt
```

如果错误是 `GLIBCXX_* not found`，还要检查程序实际加载的运行时库；仅切换编译器并不会自动修复已有二进制的依赖。


## 阅读自测与验收

- 记录配置阶段选中的 CMAKE_CXX_COMPILER 或实际编译命令；修改 PATH 后复用旧 CMakeCache.txt，不一定切换了编译器。
- 用一个最小程序同时验证编译、链接和运行，并保留原工具链；编译成功不能排除运行时 ABI 不匹配。
