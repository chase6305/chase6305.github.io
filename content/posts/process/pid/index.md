---
title: 'Ubuntu 下查看进程 PID 和终止进程方法'
date: 2025-02-26
lastmod: 2025-02-26
draft: false
tags: ["Ubuntu"]
categories: ["Ubuntu"]
authors: ["chase"]
summary: "Ubuntu 下查看进程 PID 和终止进程方法"
showToc: true
TocOpen: true
hidemeta: false
comments: false
---

#### 查看进程 PID

1. **使用 `ps` 命令**:
   ```sh
   ps aux | grep <process_name>
   ```
   例如，查看名为 `python` 的进程：
   ```sh
   ps aux | grep python
   ```

2. **使用 `pgrep` 命令**:
   ```sh
   pgrep <process_name>
   ```
   例如，查看名为 `python` 的进程：
   ```sh
   pgrep python
   ```

3. **使用 `top` 命令**:
   ```sh
   top
   ```
   在 `top` 界面中，可以按 `Shift + M` 按内存使用排序，按 `Shift + P` 按 CPU 使用排序。按 `q` 退出。

4. **使用 `htop` 命令**（需要先安装 `htop`）:
   ```sh
   sudo apt-get install htop
   htop
   ```
   在 `htop` 界面中，可以使用上下箭头键选择进程，按 `F3` 搜索进程，按 `F9` 终止进程。按 `q` 退出。

#### 终止进程

1. **使用 `kill` 命令**:
   ```sh
   kill <PID>
   ```
   例如，终止 PID 为 1234 的进程：
   ```sh
   kill 1234
   ```

2. **使用 `killall` 命令**:
   ```sh
   killall <process_name>
   ```
   例如，终止所有名为 `python` 的进程：
   ```sh
   killall python
   ```

3. **使用 `pkill` 命令**:
   ```sh
   pkill <process_name>
   ```
   例如，终止所有名为 `python` 的进程：
   ```sh
   pkill python
   ```

4. **使用 `xkill` 命令**（用于图形界面）:
   ```sh
   xkill
   ```
   然后点击要终止的窗口。

#### 强制终止进程

如果进程无法正常终止，可以使用 `-9` 信号强制终止：

1. **使用 `kill -9` 命令**:
   ```sh
   kill -9 <PID>
   ```
   例如，强制终止 PID 为 1234 的进程：
   ```sh
   kill -9 1234
   ```

2. **使用 `killall -9` 命令**:
   ```sh
   killall -9 <process_name>
   ```
   例如，强制终止所有名为 `python` 的进程：
   ```sh
   killall -9 python
   ```

3. **使用 `pkill -9` 命令**:
   ```sh
   pkill -9 <process_name>
   ```
   例如，强制终止所有名为 `python` 的进程：
   ```sh
   pkill -9 python
   ```