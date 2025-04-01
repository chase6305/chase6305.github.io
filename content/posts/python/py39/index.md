---
title: 'libpython3.9.so.1.0: cannot open shared object file: No such file or directory 解决方法'
date: 2025-02-28
lastmod: 2025-02-28
draft: false
tags: ["Ubuntu", "Python"]
categories: ["Ubuntu"]
authors: ["chase"]
summary: "libpython3.9.so.1.0: cannot open shared object file: No such file or directory 解决方法"
showToc: true
TocOpen: true
hidemeta: false
comments: false
---

这个错误通常是由于系统找不到 `libpython3.9.so.1.0` 共享库文件。以下是解决该问题的几种方法：

### 方法一：安装缺失的库文件
确保已安装 Python 3.9 及其开发包。

在基于 Debian 的系统（如 Ubuntu）上，可以使用以下命令：
```bash
sudo apt-get update
sudo apt-get install python3.9 python3.9-dev
```

在基于 Red Hat 的系统（如 CentOS）上，可以使用以下命令：
```bash
sudo yum install python39 python39-devel
```

### 方法二：创建符号链接
如果库文件已安装但系统找不到，可以手动创建符号链接。

首先，查找 `libpython3.9.so.1.0` 的实际位置：
```bash
find /usr -name "libpython3.9.so.1.0"
```

假设找到的路径是 `/usr/local/lib/libpython3.9.so.1.0`，可以创建符号链接：
```bash
sudo ln -s /usr/local/lib/libpython3.9.so.1.0 /usr/lib/libpython3.9.so.1.0
```

### 方法三：更新 `LD_LIBRARY_PATH`
将库文件所在目录添加到 `LD_LIBRARY_PATH` 环境变量中。

假设库文件位于 `/usr/local/lib`，可以使用以下命令：
```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

为了永久生效，可以将上述命令添加到 `~/.bashrc` 或 `~/.profile` 文件中。

### 方法四：使用 `ldconfig` 更新库缓存
将库文件所在目录添加到 `/etc/ld.so.conf.d/` 中，并运行 `ldconfig` 更新库缓存。

创建一个新的配置文件，例如 `/etc/ld.so.conf.d/python3.9.conf`，并添加库文件所在目录：
```bash
echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/python3.9.conf
```

然后运行 `ldconfig` 更新库缓存：
```bash
sudo ldconfig
```

以下是在Ubuntu 22.04上安装Python 3.9的完整教程：

### PS: 在Ubuntu 22.04上安装Python 3.9

1. 首先更新系统包列表：
```bash
sudo apt update
sudo apt upgrade -y
```

2. 安装依赖包：
```bash
sudo apt install -y software-properties-common
```

3. 添加deadsnakes PPA源：
```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
```

4. 再次更新包列表：
```bash
sudo apt update
```

5. 安装Python 3.9：
```bash
sudo apt install -y python3.9
```

6. 安装pip和其他开发工具：
```bash
sudo apt install -y python3.9-dev python3.9-venv python3.9-distutils
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.9 get-pip.py
```

7. 验证安装：
```bash
python3.9 --version
pip3.9 --version
```

### 可选步骤

#### 设置Python 3.9为默认Python版本
```bash
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
```

#### 创建虚拟环境
```bash
# 创建新的虚拟环境
python3.9 -m venv ~/venvs/py39env

# 激活虚拟环境
source ~/venvs/py39env/bin/activate

# 退出虚拟环境
deactivate
```

### 常见问题解决

1. 如果遇到添加PPA源时的GPG错误：
```bash
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776
```

2. 如果遇到无法定位软件包的错误：
```bash
sudo apt install -y apt-transport-https ca-certificates
```

3. 如果pip安装失败：
```bash
# 手动下载get-pip.py
wget https://bootstrap.pypa.io/get-pip.py
# 使用Python 3.9安装
sudo python3.9 get-pip.py
```

### 注意事项

1. Ubuntu 22.04默认带有Python 3.10，安装3.9不会影响系统默认Python
2. 建议使用虚拟环境来管理不同的Python版本和包
3. 使用`deadsnakes` PPA是安装其他Python版本的推荐方式

### 确认安装成功
```bash
# 检查Python版本
python3.9 --version

# 检查pip版本
pip3.9 --version

# 测试Python环境
python3.9 -c "print('Hello from Python 3.9!')"
```

这样就完成了Python 3.9的安装。对于开发项目，建议创建虚拟环境来管理依赖。

### 结论
通过安装缺失的库文件、创建符号链接、更新 `LD_LIBRARY_PATH` 或使用 `ldconfig` 更新库缓存，可以解决 `libpython3.9.so.1.0: cannot open shared object file: No such file or directory` 错误。选择适合你系统环境的方法进行操作即可。