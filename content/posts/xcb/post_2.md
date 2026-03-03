---
title: 'Linux系统解决Qt platform plugin "xcb"缺失问题'
date: 2021-08-07
lastmod: 2021-08-07
draft: false
tags: ["Qt", "Linux", "Debug"]
categories: ["编程技术"]
authors: ["chase"]
summary: "解决Linux系统下Qt应用程序常见的xcb插件缺失问题"
showToc: true
TocOpen: true
hidemeta: false
comments: false
---

之前在**Debian10.0**系统中安装图形库（如QT）相关时出现xcb缺失、xinerama缺失的问题。

```bash {.wrap}
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
libxcb-xinerama.so.0: cannot open shared object file: No such file or directory
```

在`~/.bashrc`中添加`QT_DEBUG_PLUGINS`在编译过程中可列出详细错误，亦可直接`export`：

```bash {.wrap}
export QT_DEBUG_PLUGINS=1
```

有装vscode的可在主目录终端输入`code .`
打开`.bashrc`进行相关的修改。

此处，主要是动态链接库的问题，在加载**libqxcb.so**库的时候，还需要加载**libxcb-xinerama**库。那么不存在**libxcb-xinerama.so.0**库，就安装这个库。

```bash {.wrap}
sudo apt-get install libxcb-xinerama0
```
如果安装完**libxcb-xinerama.so.0**仍不能解决问题，可能是因为图形库开发人员的系统环境不一样，只依赖了**libxcb-util.so.1**，然而在Debian系统中对应**libxcb-util.so.1**的库名称为**libxcb-util.so.0**，因而可将进行符号链接，将**libxcb-util.so.1**软链接到**libxcb-util.so.0**。
```bash {.wrap}
cd /usr/lib/x86_64-linux-gnu/

sudo ln -s libxcb-util.so.0  libxcb-util.so.1
```
亦可：
```bash {.wrap}
sudo ln -s /usr/lib/x86_64-linux-gnu/libxcb-util.so.0  /usr/lib/x86_64-linux-gnu/libxcb-util.so.1
```
在创建后即可解决问题，后续可将`QT_DEBUG_PLUGINS`关掉或删掉。
