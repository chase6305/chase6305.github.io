---
title: 'symbol version Qt_5_PRIVATE_API not defined in libQt5Gui.so.5 解决方法'
date: 2025-02-28
lastmod: 2025-02-28
draft: false
tags: ["QT"]
categories: ["QT"]
authors: ["chase"]
summary: 'symbol version Qt_5_PRIVATE_API not defined in libQt5Gui.so.5 解决方法'
showToc: true
TocOpen: true
hidemeta: false
comments: false
---


使用 PyQt5 时遇到了库版本不兼容的问题。以下是解决该问题的详细步骤：

### 问题描述
在链接时出现以下错误：
- `symbol version Qt_5_PRIVATE_API not defined in libQt5Gui.so.5`
- `QT_5 not define in file libQt5Core.so.5 with link time reference`

### 解决方案
1. **卸载 PyQt5**：
   - 使用 `pip uninstall pyqt5` 命令卸载当前安装的 PyQt5。

2. **检查已安装的 PyQt5 相关包**：
   - 使用 `pip list` 查看已安装的 PyQt5 相关包，确保只保留 `PyQt5-sip`。

3. **重新安装兼容版本的 PyQt5**：
   - 根据 `PyQt5-sip` 的版本，重新安装兼容的 PyQt5 版本。例如，如果 `PyQt5-sip` 版本是 `12.7.0`，可以尝试安装 `PyQt5==5.13.2`。

### 示例命令
```bash
# 卸载当前的 PyQt5
pip uninstall pyqt5

# 检查已安装的 PyQt5 相关包
pip list

# 安装兼容版本的 PyQt5
pip install pyqt5==5.13.2
```

### 结论
通过卸载不兼容的 PyQt5 版本并重新安装与 `PyQt5-sip` 兼容的版本，可以解决链接时出现的符号未定义问题。确保环境中各个库的版本兼容性是解决此类问题的关键。