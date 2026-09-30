---
title: "VS Code Python 调试：解释器、断点与附加调试"
date: 2026-01-27
lastmod: 2026-09-30
draft: false
tags: ["VS Code", "Python", "Debugging"]
categories: ["系统与工具"]
authors: ["chase"]
summary: "从最小 debugpy 配置讲解 VS Code 断点、模块入口、参数与附加调试，补充解释器核对及调试端口保护。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "从最小 debugpy 配置讲解 VS Code 断点、模块入口、参数与附加调试，补充解释器核对及调试端口保护。"
contentLanguage: "zh-CN"
reading_prerequisites: "Python 执行入口与 VS Code"
reading_focus: "先跟随最小循环观察断点前后的变量，再连接解释器、模块入口与附加调试。"
related_posts:
  - "/posts/vscode/two-env"
  - "/posts/process/pid"
---

## 先用最小调试配置定位问题

调试器让程序在断点暂停，查看变量和调用栈。先确认所选 Python 解释器、工作目录和启动参数与正常运行一致，再加入远程、容器或多进程配置。

本文使用 VS Code Python Debugger（debugpy）。`launch.json` 支持注释，因此代码块标为 JSONC；其中的调试设置要由具体适配器支持，不能混用 Node.js 或 Docker 扩展的字段。

## 1. 当前文件调试

在项目的 `.vscode/launch.json` 中保存：

```jsonc
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: 当前文件",
      "type": "debugpy",
      "request": "launch",
      "program": "${file}",
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal",
      "justMyCode": true
    }
  ]
}
```

上面路径使用 VS Code 变量，按项目需要指定固定入口。运行前通过“Python: Select Interpreter”选择解释器；在代码中打印 `sys.executable` 可确认调试实际使用的环境。

设置断点后按 `F5`。常见默认快捷键如下，操作系统和自定义键位可能不同：

| 操作 | 默认快捷键 |
| --- | --- |
| 启动 / 继续 | F5 |
| 无调试运行 | Ctrl+F5 |
| 暂停 | F6 |
| 单步跳过 | F10 |
| 单步进入 / 跳出 | F11 / Shift+F11 |
| 切换断点 | F9 |
| 停止 | Shift+F5 |
| 重启 | Ctrl+Shift+F5 |

### 第一次单步，应该观察到什么

把下面的独立例子保存为 `debug_demo.py`，在 `updated = ...` 行设置断点，然后使用上面的“当前文件”配置启动：

```python
def apply_offset(raw, offset):
    updated = raw + offset
    return updated


samples = [10, 20, 30]
corrected = [apply_offset(value, 2) for value in samples]
print(corrected)  # [12, 22, 32]
```

第一次停下时，`raw=10`、`offset=2`；断点所在行尚未执行，所以当前调用里还没有 `updated`。按 F10 执行这一行后，应看到 `updated=12`。调用栈能说明这是从列表推导式进入的函数；继续运行会在后两次调用停下，最后输出三个结果。

如果想只观察第二个样本，可把这个断点的条件设为 `raw == 20`。这个实验分别验证了入口、断点、局部变量、单步和条件表达式；它比一开始就在多进程训练入口排查空心断点更容易定位配置问题。调试控制台里修改 `raw` 会影响后续计算，观察值与修改值应有意识地区分。

## 2. 模块、参数与环境变量

模块入口用 `module`，不要同时配置 `program`。数组中的每个元素是一个命令行参数，不要把所有参数拼成一条字符串：

```jsonc
{
  "name": "Python: 模块入口",
  "type": "debugpy",
  "request": "launch",
  "module": "my_package.train",
  "args": ["--config", "configs/debug.yaml", "--steps", "10"],
  "cwd": "${workspaceFolder}",
  "console": "integratedTerminal",
  "env": {
    "PYTHONUNBUFFERED": "1"
  },
  "justMyCode": false
}
```

这是一项 configuration，放入完整文件的 `configurations` 数组。`my_package.train` 和配置路径都是项目占位示例。凭据不要提交到 `launch.json`，环境文件也需按项目规则排除敏感内容。

## 3. 条件断点、日志点与异常断点

右键断点可设置条件，例如 `step >= 100 and loss > 10`。条件在目标进程中求值，应避免写文件、发网络请求或修改状态的表达式。

日志点适合观察循环变量而不频繁暂停；异常断点适合定位最初的抛错位置。调试控制台执行的是程序上下文中的表达式，同样可能改变对象状态，不是纯只读窗口。

`justMyCode: false` 允许进入库代码，但 Python 调试不能用 Node 的 `skipFiles` 规则作为通用性能开关。

## 4. 本机与 SSH 附加调试

在目标机器的正确环境运行：

```bash
python -m debugpy --listen 127.0.0.1:5678 --wait-for-client your_script.py
```

本机添加附加配置：

```jsonc
{
  "name": "Python: Attach",
  "type": "debugpy",
  "request": "attach",
  "connect": {
    "host": "127.0.0.1",
    "port": 5678
  },
  "justMyCode": false
}
```

远程机器可通过 `ssh -L 5678:127.0.0.1:5678 user@server` 转发。若本地和远程源码路径不同，按实际位置配置 `pathMappings`；源码版本也必须一致。

调试端口具有执行代码的能力，不应无保护监听公网地址。暂停机器人控制线程、生产服务或分布式训练 rank 可能触发超时或失控，优先在隔离环境复现。

## 5. 异步、多进程与容器

`asyncio` 程序通常按普通 Python 入口调试，不存在通用的 `"asyncio": true` debugpy 开关。`gevent` 配置针对 gevent，不是所有协程库。

多进程应用按已安装 debugpy 版本评估 `subProcess`，并关注启动方法。容器使用 Dev Containers / Remote 环境或受保护的 attach，不能随意添加未定义的 `dockerOptions` 就期望启动容器。

## 6. 断点不生效时的检查表

- 断点为空心：检查源码映射、实际执行文件和模块是否加载。
- 终端能运行、调试失败：比较解释器、cwd、args、环境变量与依赖版本。
- 暂停后卡住：检查其他线程、进程和外部服务是否等待当前线程持有的锁。
- 问题只在无调试模式出现：考虑时序、超时和竞态；调试器会改变执行速度。

参考：[VS Code Python 调试](https://code.visualstudio.com/docs/python/debugging)、[debugpy 使用说明](https://github.com/microsoft/debugpy)。


## 阅读自测与验收

- 从调试控制台打印解释器或程序路径，并核对启动目录、参数和环境变量，避免断点落在另一份源文件上。
- 先验证断点、单步和异常中断，再配置远程 attach；调试端口只绑定本机，通过受控隧道访问。
