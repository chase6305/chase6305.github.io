---
title: 在 Ubuntu 上安装和切换多个 GCC 版本
date: 2025-03-07
lastmod: 2025-03-07
draft: false
tags: ["C++", "GCC"]
categories: ["编程技术"]
authors: ["chase"]
summary: 在 Ubuntu 上安装和切换多个 GCC 版本
showToc: true
TocOpen: true
hidemeta: false
comments: false
---


在软件开发过程中，有时需要在不同的 GCC 版本之间进行切换。本文将介绍如何在 Ubuntu 上安装多个 GCC 版本，并使用 `update-alternatives` 工具进行版本切换。

#### 1. 安装所需的 GCC 版本

首先，更新包列表并安装所需的 GCC 版本。本文将安装 `gcc-7`, `gcc-9` 和 `gcc-11`。

```sh
sudo apt update
sudo apt install gcc-7 g++-7 gcc-9 g++-9 gcc-11 g++-11
```

#### 2. 使用 `update-alternatives` 工具配置 GCC 和 G++

使用 `update-alternatives` 工具来管理和切换 GCC 和 G++ 版本。

```sh
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 70
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
```

#### 3. 切换 GCC 和 G++ 版本

运行以下命令来选择默认的 GCC 和 G++ 版本：

```sh
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```

系统会显示一个可用 GCC 和 G++ 版本的列表，输入相应的数字选择所需的版本（例如 `gcc-7` 和 `g++-7`）。

#### 4. 验证 GCC 和 G++ 版本

切换完成后，可以通过以下命令验证当前使用的 GCC 和 G++ 版本：

```sh
gcc --version
g++ --version
```

#### 5. 手动设置环境变量（如果需要）

如果 `update-alternatives` 设置正确，但仍然显示旧版本，可以尝试手动设置环境变量：

1. **编辑 `.bashrc` 或 `.zshrc` 文件**：

    ```sh
    nano ~/.bashrc
    ```

    或者：

    ```sh
    nano ~/.zshrc
    ```

2. **添加以下行来设置 GCC 和 G++ 的路径**：

    ```sh
    export PATH=/usr/bin/gcc-7:$PATH
    export PATH=/usr/bin/g++-7:$PATH
    ```

3. **保存并退出编辑器**。

4. **重新加载配置文件**：

    ```sh
    source ~/.bashrc
    ```

    或者：

    ```sh
    source ~/.zshrc
    ```

#### 6. 验证版本

再次验证 GCC 和 G++ 版本：

```sh
gcc --version
g++ --version
```

### 总结

通过上述步骤，你可以在 Ubuntu 上安装和切换多个 GCC 版本。