---
title: 'Ubuntu 下查看进程 PID 和终止进程方法'
date: 2025-02-26
lastmod: 2026-09-05
draft: false
tags: ["Linux", "Process Management"]
categories: ["系统与工具"]
authors: ["chase"]
summary: "在 Ubuntu 上核对 PID、用户和启动信息，再选择 SIGTERM 或 SIGKILL，说明进程回收与服务自动重启。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "在 Ubuntu 上核对 PID、用户和启动信息，再选择 SIGTERM 或 SIGKILL，说明进程回收与服务自动重启。"
contentLanguage: "zh-CN"
reading_prerequisites: "Linux 进程与信号"
reading_focus: "优先使用应用或服务管理器的停止入口，避免按 python 名称批量终止任务。"
related_posts:
  - "/posts/linux/watch"
  - "/posts/vscode/debug"
---

## 先识别进程，再发送信号

PID 是进程标识，不是控制算法 PID。终止进程可能丢失未保存数据或中断硬件通信；先核对用户、启动时间、命令行和父进程，不要看到 `python` 就批量结束。

## 查找与确认

```bash
pgrep -a -u "$USER" -f 'your_application.py'
ps -p 1234 -o pid,ppid,user,lstart,stat,args
```

这里的脚本名和 `1234` 都是占位示例，需要替换。`pgrep -f` 匹配整个命令行，是正则匹配，可能命中不止一个进程。第二步必须检查精确 PID，尤其是在长时间等待后，PID 可能已被系统复用。

交互检查也可使用 `top` 或 `htop`；`top` 中通常用 `Shift+M` 按内存排序、`Shift+P` 按 CPU 排序，`q` 退出。

## 优先使用应用自己的停止方式

前台终端通常先按 `Ctrl+C`，让应用处理 SIGINT 并清理资源。由 systemd、容器或任务调度器管理的服务，应使用对应的停止入口，否则直接杀子进程后可能立即被拉起。

对已确认属于自己的单个进程，发送正常终止信号：

```bash
kill -TERM 1234
```

SIGTERM 可以被应用处理，便于保存状态、关闭文件和断开连接。等待应用正常退出，再用 `ps -p 1234` 核实。

## 何时才考虑 SIGKILL

只有在明确目标、接受无法清理的后果且正常停止无效时，才考虑：

```bash
kill -KILL 1234
```

SIGKILL 无法被捕获，不能保证文件和共享状态完整；不可中断内核等待中的进程也可能不会立即消失。对于机器人，软件进程消失更不等于驱动器已经安全停机，硬件安全链应独立存在。

不要把 `killall python`、`pkill -9 python` 作为默认排障命令，它们可能结束同一用户的其他训练、服务或编辑器任务。

## 常见现象

- `No such process`：目标已退出或 PID 不正确，重新查询，不继续猜 PID。
- `Operation not permitted`：核对所有者和权限，先确认是否有操作该服务的授权。
- 状态 `Z`：僵尸进程已经退出，需要父进程回收，重复 kill 通常无效。
- 停止后重新出现：检查服务管理器或父进程的重启策略，不循环强杀。


## 阅读自测与验收

- 发送信号前再次核对 PID、命令行、用户和启动时间，防止误认同名进程或复用的 PID。
- 先观察正常退出是否完成资源清理，仍未退出时再分析原因；停止控制进程不等于机器人已进入安全状态。
