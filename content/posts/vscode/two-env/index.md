---
title: VScode终端出现显示两个环境名问题的解决方案
date: 2025-02-07
lastmod: 2025-02-07
draft: false
tags: ["Vscode", "Conda", "Bug"]
categories: ["编程技术"]
authors: ["chase"]
summary: VScode终端出现显示两个环境名问题的解决方案
showToc: true
TocOpen: true
hidemeta: false
comments: false
---


在使用 Visual Studio Code (VS Code) 进行 Python 开发时，有时会遇到终端中显示两个环境名的问题。这可能会导致混淆，并影响开发体验。本文将介绍如何通过修改 VS Code 的设置文件来解决这个问题。

```bash
(base) (myenv) user@hostname:~/project$
```

## 问题描述
在 VS Code 中，当你打开终端并激活 Python 环境时，可能会看到类似以下的输出：

这里显示了两个环境名 **(base)** 和 **(myenv)** ，这可能会让人困惑，不知道当前到底使用的是哪个环境。

## 解决方案
我们可以通过修改 VS Code 的设置文件 **settings.json** 来解决这个问题。具体步骤如下：

打开 VS Code。

安装并启用 Python 插件（如果尚未安装）。

打开 VS Code 的设置文件 **settings.json**。

你可以通过以下步骤打开 **settings.json** 文件：

点击左下角的齿轮图标，然后选择“**设置**”。
在设置界面右上角点击打开 **JSON** 设置文件的图标。
在 **settings.json** 文件中添加以下配置：

```yaml
"python.terminal.activateEnvironment": false,
"terminal.integrated.shellIntegration.enabled": false,
```

这两行配置的作用如下：

**"python.terminal.activateEnvironment": false** 禁用 Python 插件自动激活环境的功能。
**"terminal.integrated.shellIntegration.enabled": false** 禁用终端集成的 shell 环境激活功能。
保存 **settings.json** 文件。

## 示例
以下是一个完整的 settings.json 文件示例：

```yaml
{
    // 其他配置项...
    "python.terminal.activateEnvironment": false,
    "terminal.integrated.shellIntegration.enabled": false
}
```

## 效果
完成上述配置后，当你在 VS Code 中打开终端并激活 Python 环境时，应该只会显示一个环境名。例如：

```bash
(myenv) user@hostname:~/project$
```

这样可以避免显示两个环境名的问题，提升开发体验。

## 为了避免自动激活环境
你可以在~/.bashrc文件中在初始化conda环境后, 添加:

```bash
conda config --set auto_activate_base False
```
即
```bash
# 其他配置
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/chase/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/chase/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/chase/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/chase/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate py39 # 这里你可以自动选用你默认使用的环境
conda config --set auto_activate_base false 
```
