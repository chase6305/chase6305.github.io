---
title: 'JetsonNV 上解决 PyQt5 “Could not load the Qt platform plugin ‘xcb‘“ 错误'
date: 2025-03-06 
lastmod: 2025-03-06 
draft: false
tags: ["GUI", "pyqt5", "Jetson"]
categories: ["GUI"]
authors: ["chase"]
summary: "JetsonNV 上解决 PyQt5 “Could not load the Qt platform plugin ‘xcb‘“ 错误"
showToc: true
TocOpen: true
hidemeta: false
comments: false
---


在 Jetson NV 上运行 PyQt5 应用程序时，可能会遇到以下错误：

```
qt.qpa.xcb: could not connect to display 
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, xcb.

Aborted (core dumped)
```

这是由于缺少显示环境或 Qt 平台插件的问题。以下是解决该问题的详细步骤：

### 1. 设置 **DISPLAY** 环境变量

确保你有一个有效的显示环境。如果你在没有显示环境的情况下运行（例如通过 SSH），你可以尝试设置 `DISPLAY` 环境变量：

```sh
export DISPLAY=:0
```

### 2. 安装必要的库

确保你已经安装了所有必要的 Qt 库：

```sh
sudo apt-get install libxcb-xinerama0
```

### 3. 使用 `xhost` 允许连接

如果你通过 SSH 连接到 Jetson NV，可以使用 `xhost` 允许连接：

```sh
xhost +
```

### 4. 使用 `offscreen` 平台插件

如果你没有显示环境，可以尝试使用 `offscreen` 平台插件：

```sh
QT_QPA_PLATFORM=offscreen python lcg_qt_control_20_ZeroErr_drives.py
```

### 5. 检查 Qt 安装

确保你的 Qt 安装没有问题，可以尝试重新安装 Qt：

```sh
sudo apt-get install --reinstall qt5-default
```

### 6. 使用 VNC 或 X11 转发

如果你通过 SSH 连接，可以使用 VNC 或 X11 转发来提供显示环境：

#### 使用 X11 转发

在 SSH 连接时使用 `-X` 或 `-Y` 选项：

```sh
ssh -X user@jetson-nv
```

#### 使用 VNC

安装并配置 VNC 服务器，然后通过 VNC 客户端连接。

### 7. 检查 Python 环境

确保你在正确的 Python 环境中运行代码，并且已经正确安装了 PyQt5。

### 示例代码

以下是一个简单的 PyQt5 示例代码，创建一个基本的窗口：

```python
from PyQt5 import QtWidgets
import sys

app = QtWidgets.QApplication(sys.argv)

window = QtWidgets.QWidget()
window.setWindowTitle('PyQt5 example')
window.setGeometry(100, 100, 280, 80)
window.show()

sys.exit(app.exec_())
```

### 运行示例代码

1. **保存上述代码到一个文件**，例如 `example.py`。
2. **在终端中运行该文件**：
    ```sh
    python example.py
    ```

通过以上步骤，你应该能够在 Jetson NV 上成功运行 PyQt5 应用程序。如果遇到任何问题，请检查错误信息并确保所有依赖项都已正确安装。