---
title: "使用 watch 命令监控系统和进程状态"
date: 2025-03-07
lastmod: 2026-09-05
draft: false
tags: ["Linux", "Process Monitoring", "CLI"]
categories: ["系统与工具"]
authors: ["chase"]
summary: "用 watch 周期性查看资源、进程和 EtherCAT 状态，讲清刷新间隔、命令引用与监控副作用。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "用 watch 周期性查看资源、进程和 EtherCAT 状态，讲清刷新间隔、命令引用与监控副作用。"
contentLanguage: "zh-CN"
reading_prerequisites: "Linux shell 与只读诊断命令"
reading_focus: "监控命令必须可安全重复；瞬时故障和短窗口 CPU 需要专门采样工具。"
related_posts:
  - "/posts/process/pid"
  - "/posts/dialout/dh"
---

`watch` 周期性执行一个命令，并用新结果替换终端画面。适合观察内存、磁盘和进程快照；它既不是日志记录器，也不保证捕获两次刷新之间的瞬时故障。以下以 Linux 的 procps-ng 实现为例。

## 1. 先单次检查，再放进循环

先确认命令的目标、权限、输出单位与副作用，再加 `watch`。`Ctrl+C` 停止刷新。

```bash
free -h
watch -n 1 -d 'free -h'

# 只读磁盘快照，降低刷新频率。
watch -n 5 'df -h'

# 按内存占用排序；表头加十个进程。
watch -n 2 'ps aux --sort=-%mem | head -n 11'
```

| 选项 | 含义与边界 |
| --- | --- |
| `-n 1` | 两次更新间隔设为 1 秒；执行本身耗时也会影响刷新 |
| `-d` | 标出相邻画面的变化，不会保留完整变化历史 |
| `-t` | 隐藏标题和间隔信息 |
| `-g` | 检测到可见输出变化时退出，不能替代业务成功条件 |
| `-e` | 命令返回非零时冻结更新，等待按键后退出，不适合当作无人值守重试器 |
| `-c` | 解释 ANSI 颜色和样式控制序列，仅用于可信输出 |

默认模式由 shell 解释命令。管道整体用单引号包住，避免外层 shell 提前执行管道或变量替换：

```bash
# date 会在每次刷新时执行，而不是启动 watch 时只执行一次。
watch -n 2 'date; cat /proc/loadavg'

# 将模式换成自己的程序名；方括号避免匹配携带原模式的 shell 命令行。
watch -n 2 'pgrep -af "[r]obot_program" || true'
```

最后的 `|| true` 有意把“未匹配到进程”变为空输出；它也会掩盖 `pgrep` 自身的错误。需要区分“没有进程”与“查询失败”的自动化，不应直接复用这个展示命令。

## 2. 不把快照当成区间采样

`ps %CPU` 通常是进程运行以来累计的 CPU 时间比例；反复启动 `top -b -n 1` 也不等同于保持同一个采样器连续运行。想观察最近一秒的变化，可以直接运行：

```bash
top -d 1

# pidstat 来自 sysstat；将 12345 替换为已核实的目标 PID。
pidstat -p 12345 1
```

这些程序自己维持采样间隔，无需再包一层 `watch`。磁盘较大时反复 `du` 会产生明显 I/O；远程查询、SDO 读取也会占用相应资源，刷新越快并不一定越有用。

## 3. 用 tmux 同时看三组指标

下面保存为 `monitor_tmux.sh`。所有拆分和布局操作都显式指定新建会话或 pane ID，不依赖当前活动窗格，也不会向已有会话发送按键。

```bash
#!/usr/bin/env bash
set -euo pipefail

for required in tmux watch free df; do
  command -v "$required" >/dev/null || {
    printf '缺少命令：%s\n' "$required" >&2
    exit 1
  }
done

monitor_session=blog-monitor
if tmux has-session -t "=$monitor_session" 2>/dev/null; then
  printf '会话 %s 已存在；请先查看它，脚本不会覆盖。\n' "$monitor_session" >&2
  exit 1
fi

first_pane=$(tmux new-session -d -P -F '#{pane_id}' \
  -s "$monitor_session" -n metrics -x 120 -y 30 'watch -n 1 free -h')
tmux split-window -d -h -t "$first_pane" 'watch -n 5 df -h'
tmux split-window -d -v -t "$first_pane" 'watch -n 2 cat /proc/loadavg'
tmux select-layout -t "$monitor_session:metrics" tiled
printf '查看新会话：tmux attach-session -t %s\n' "$monitor_session"
```

```bash
bash monitor_tmux.sh
tmux attach-session -t blog-monitor
```

`Ctrl+B` 后按方向键切换窗格；`Ctrl+B` 后按 `D` 只会分离客户端，监控仍在后台运行。若要结束，在各窗格中用 `Ctrl+C` 停止 `watch`、再输入 `exit`；最后一个窗格关闭后会话随之结束。创建过程若中途失败，脚本不自动删除已建会话，先人工检查。

## 4. EtherCAT：只在确认现场环境后查询

以下不是通用电脑的必需命令。先确认已安装并配置相应主站工具、设备节点权限和现场拓扑，再单次查询：

```bash
ethercat slaves
ethercat master

# 确认上面查询安全且轮询负载可接受后，再按需运行这一条。
watch -n 2 'ethercat slaves'
```

读取某个从站的对象字典前，还需核实从站位置、对象索引、类型及设备手册；不同设备不一定提供相同对象。不要把写寄存器、状态切换、重启服务或清理文件放进刷新循环，也不要为只读监控直接扩大系统权限。

参考：[watch 手册](https://man7.org/linux/man-pages/man1/watch.1.html)、[tmux 的目标会话与窗格](https://github.com/tmux/tmux/wiki/Advanced-Use)。

## 阅读自测与验收

- 先单次执行被 watch 包裹的命令，核对筛选范围和输出单位；重复执行的频率要低于设备或系统可承受的轮询负载。
- 区分快照、累计量和区间采样值；watch 刷新更快不代表测量值具有更高的时间精度。
