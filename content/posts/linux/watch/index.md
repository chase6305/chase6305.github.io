---
title: "使用 watch 命令监控系统和进程状态"
date: 2025-03-07
lastmod: 2025-03-07
draft: false
tags: ["Ubuntu", "Watch"]
categories: ["Ubuntu"]
authors: ["chase"]
summary: "watch 是一个功能强大的 Linux 命令行工具，用于周期性执行命令并实时显示输出结果。它特别适合监控系统状态、进程运行和资源使用情况。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
---



## 基本介绍

`watch` 是一个功能强大的 Linux 命令行工具，用于周期性执行命令并实时显示输出结果。它特别适合监控系统状态、进程运行和资源使用情况。

## 基本语法

```bash
watch [选项] 命令
```

### 常用选项
- `-n N`: 设置更新间隔(默认2秒)
- `-d`: 高亮显示变化的部分
- `-t`: 不显示标题栏
- `-g`: 当输出变化时退出
- `-e`: 当出错时退出
- `-c`: 启用彩色输出

## 常用监控场景

### 1. 系统资源监控

```bash
# 监控内存使用
watch -n 1 'free -h'

# 监控CPU使用
watch -n 1 'top -b -n 1 | head -n 20'

# 监控系统负载
watch -n 1 'cat /proc/loadavg'

# 监控磁盘使用
watch -n 5 'df -h'
```

### 2. 进程监控

```bash
# 监控特定进程
watch -n 1 -d 'ps -o pid,%cpu,%mem,rss,cmd -p $(pgrep -f process_name)'

# 按内存排序显示前10个进程
watch -n 1 'ps aux --sort=-%mem | head -n 11'
```

### 3. EtherCAT通信监控

```bash
# 监控从站状态
watch -n 1 'ethercat slaves'

# 监控主站状态
watch -n 1 'ethercat master'

# 监控特定从站状态字
watch -n 1 'ethercat upload --type uint16 0x6041 0'
```

### 4. 文件系统监控

```bash
# 监控目录大小
watch -n 5 'du -sh /path/to/directory'

# 监控文件变化
watch -n 1 'ls -l /path/to/file'
```

## 高级使用技巧

### 1. 组合监控脚本

```bash
#!/bin/bash
# monitor_system.sh
echo "=== 系统状态 ==="
echo "内存使用:"
free -h
echo
echo "CPU使用:"
top -bn1 | head -n 3
echo
echo "EtherCAT状态:"
ethercat slaves
```

使用方法：
```bash
chmod +x monitor_system.sh
watch -n 1 -d ./monitor_system.sh
```

### 2. TMUX分屏监控

```bash
#!/bin/bash
# monitor_tmux.sh
tmux new-session -d -s monitor
tmux split-window -h
tmux split-window -v
tmux select-pane -t 0
tmux send-keys "watch -n 1 'free -h'" C-m
tmux select-pane -t 1
tmux send-keys "watch -n 1 'top -bn1'" C-m
tmux select-pane -t 2
tmux send-keys "watch -n 1 'ethercat slaves'" C-m
tmux attach-session -t monitor
```

使用方法：
```bash
chmod +x monitor_tmux.sh
./monitor_tmux.sh
```

## 注意事项

1. 合理设置更新间隔，避免系统负载过高
2. EtherCAT相关命令需要root权限
3. 使用 `-d` 选项可以更容易发现变化
4. 按 `Ctrl+C` 可以退出监控
5. TMUX中使用 `Ctrl+B` 然后按方向键切换窗格

## 实用组合示例

### 监控机器人程序运行状态
```bash
watch -n 1 '
echo "=== 程序状态 ===";
ps aux | grep robot_program;
echo;
echo "=== 内存使用 ===";
free -h;
echo;
echo "=== EtherCAT状态 ===";
ethercat slaves;
'
```

这种组合监控可以帮助及时发现程序异常、资源不足或通信问题。

## 小结

`watch` 命令是 Linux 系统管理和调试的重要工具，合理使用可以大大提高工作效率。通过组合不同的命令和选项，我们可以构建强大的监控方案，满足各种应用场景的需求。