---
title: 'Ubuntu切换默认python版本'
date: 2025-03-13
lastmod: 2025-03-13
draft: false
tags: ["Ubuntu"]
categories: ["Ubuntu"]
authors: ["chase"]
summary: "Ubuntu切换默认python版本"
showToc: true
TocOpen: true
hidemeta: false
comments: false
---

`update-alternatives` 是一个用于管理符号链接的工具，它允许你在系统中安装多个版本的同一个程序，并方便地在它们之间切换。以下是详细的原理介绍：

### 原理

1. **符号链接管理**：
   `update-alternatives` 通过创建和管理符号链接来实现不同版本程序的切换。符号链接是指向另一个文件或目录的指针。

2. **优先级机制**：
   每个可选项都有一个优先级，优先级数值越高，优先级越高。当你使用 `auto mode` 时，系统会自动选择优先级最高的版本。

3. **手动模式和自动模式**：
   - **手动模式**：用户明确选择某个版本，系统将一直使用该版本，直到用户再次更改。
   - **自动模式**：系统根据优先级自动选择版本，当有新的版本添加且优先级更高时，系统会自动切换到新版本。

### 操作步骤

1. **安装新版本**：
   使用 `update-alternatives --install` 命令将新版本添加到管理系统中。例如：
   ```sh
   sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
   sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
   ```
   这两条命令分别将 Python 3.10 和 Python 3.8 添加到 `update-alternatives` 系统中，并设置它们的优先级。

2. **配置版本**：
   使用 `update-alternatives --config` 命令来选择你想要的版本。例如：
   ```sh
   sudo update-alternatives --config python3
   ```
   运行该命令后，系统会列出所有可用的版本及其优先级，并提示你选择一个版本。

3. **查看当前配置**：
   你可以使用 `update-alternatives --display` 命令查看当前的配置。例如：
   ```sh
   sudo update-alternatives --display python3
   ```
   这条命令会显示当前 `python3` 的所有可选项及其状态。

### 示例

假设你已经安装了 Python 3.8 和 Python 3.10，并希望在它们之间切换：

1. 添加 Python 3.8 和 Python 3.10 到 `update-alternatives` 系统中：
   ```sh
   sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
   sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
   ```

2. 配置 Python 版本：
   ```sh
   sudo update-alternatives --config python3
   ```
   选择你想要的版本对应的数字，然后按回车键。

3. 查看当前配置：
   ```sh
   sudo update-alternatives --display python3
   ```

通过这些步骤，你可以方便地在不同版本的 Python 之间切换，而无需手动更改符号链接。