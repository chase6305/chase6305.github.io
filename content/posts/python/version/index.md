---
title: 'Ubuntu 为项目选择 Python 版本：venv、Conda 与解释器排查'
date: 2025-03-13
lastmod: 2026-09-30
draft: false
tags: ["Python", "Ubuntu", "Environment Management"]
categories: ["编程开发"]
authors: ["chase"]
summary: "使用 venv、Conda 和显式解释器管理 Python 版本，核对 pip 与运行路径，并说明系统 Python 的保护边界。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "使用 venv、Conda 和显式解释器管理 Python 版本，核对 pip 与运行路径，并说明系统 Python 的保护边界。"
contentLanguage: "zh-CN"
reading_prerequisites: "Linux shell 与 Python 包安装"
reading_focus: "解释器选择、包安装和终端激活分别验证，不修改系统 python3 软链接。"
related_posts:
  - "/posts/python/py39"
  - "/posts/vscode/two-env"
---

在 Ubuntu 上为项目选择 Python 版本，优先使用指定版本解释器创建虚拟环境。系统工具可能依赖发行版自带的 `/usr/bin/python3`，修改这个入口会让系统工具和项目依赖混在一起。

## 1. 确认当前解释器

```bash
command -v python3
python3 --version
python3 -c "import sys; print(sys.executable); print(sys.prefix)"
```

终端里的环境名只是提示。判断程序实际使用哪个环境，应查看 `sys.executable`，并使用同一解释器运行 `python -m pip`。

## 2. 用指定版本创建项目环境

以下以机器上已经安装的 Python 3.10 为例。可用版本与安装方式取决于 Ubuntu 版本；`venv` 不会替你安装新的 Python 解释器。

```bash
python3.10 --version
python3.10 -m venv .venv
source .venv/bin/activate
python --version
python -m pip --version
```

如果提示缺少 `ensurepip`，检查当前发行版是否提供对应的 `python3.10-venv` 包。激活只改变当前 shell 的命令查找路径；也可以直接运行 `.venv/bin/python app.py`。更多行为见 [Python venv 文档](https://docs.python.org/3/library/venv.html)。

## 3. 使用 Conda 管理解释器与依赖

已经使用 Conda 的项目可以创建独立环境：

```bash
conda create -n project-py310 python=3.10
conda activate project-py310
python -c "import sys; print(sys.executable)"
python -m pip --version
```

项目内选择一种环境管理方式即可。VS Code 还需要通过 **Python: Select Interpreter** 选择同一个解释器，已有终端可能需要重新打开。

## 4. update-alternatives 适用在哪里

`update-alternatives` 管理的是符号链接组：自动模式选择最高优先级候选项，手动模式固定用户选择。它不负责 Python 包隔离，也不保证系统脚本兼容另一个 Python 版本。

因此，不把替换 `/usr/bin/python3` 作为项目版本切换步骤。需要恢复之前修改过的系统入口时，先确认发行版原本提供的 Python 版本和包归属，再通过对应包恢复，不能照抄其他 Ubuntu 版本的链接路径。

## 5. 验证与退出

```bash
python -c "import sys; print(sys.version); print(sys.executable)"
python -m pip check
```

`venv` 使用 `deactivate` 退出，Conda 使用 `conda deactivate` 退出。检查实际解释器和依赖是否一致，比仅观察命令提示符更可靠。

## 6. 已安装却导入失败：沿四个入口排查

同样显示 Python 3.10，不代表两个进程共享同一套包。把解释器、包安装目标、模块来源和实际接口分开检查，通常比反复重装更快。

```bash
python -c "import sys; print('executable:', sys.executable); print('prefix:', sys.prefix); print('base_prefix:', sys.base_prefix)"
python -m pip --version
python -c "import importlib.util; s=importlib.util.find_spec('numpy'); print(None if s is None else s.origin)"
python -c "import numpy; print(numpy.__version__); print(numpy.__file__)"
```

这里用 NumPy 作例子；更换为正在排查的顶层模块名即可。第三条定位导入目标，第四条才真正加载模块，因而还能暴露共享库或二进制兼容问题。[`find_spec` 文档](https://docs.python.org/3/library/importlib.html#importlib.util.find_spec)也说明：查询带点的子模块名时可能先导入父包，因此不要把它当作适用于所有名称的“绝不执行代码”检查。

| 现象 | 优先看什么 | 典型处理 |
| --- | --- | --- |
| `ModuleNotFoundError` | 当前解释器与 `python -m pip` 的路径 | 在实际运行环境安装，而不是另一个同版本环境 |
| 模块来源指向项目里的 `numpy.py` 或 `cv2.py` | 文件名遮蔽 | 重命名自己的冲突模块，再重启 Python 进程 |
| `ImportError` / 缺少 `.so` / 未定义符号 | 实际模块文件、包版本与动态库依赖 | 在独立环境复现，定位二进制依赖，不先改系统软链接 |
| 导入成功但缺少某个函数 | `__version__`、`__file__`、`hasattr` | 核对该发行包的接口，版本号本身不能证明功能存在 |
{.table-readable}

`sys.prefix != sys.base_prefix` 可用于识别普通 `venv`，但不能据此否定一个 Conda 环境；两者的管理机制不同。`pip check` 也只检查声明的包依赖，无法保证所有动态库都能加载或每个 API 都存在。手眼标定中的 [OpenCV 接口检查]({{< relref "/posts/calibration/model" >}})给出了后一种情况的具体例子。


## 阅读自测与验收

- 把 python、python -m pip 与 IDE 使用的解释器路径放在一起比较；不要仅依赖 python --version 的主次版本号。
- 建立独立环境复现最小导入，保留系统 Python；切换项目解释器不需要改系统命令的默认链接。
