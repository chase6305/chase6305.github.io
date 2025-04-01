---
title: libstdc++.so.6 -> version `GLIBCXX_3.4.30‘ not found 解决方案
date: 2025-02-08
lastmod: 2025-02-08
draft: false
tags: ["C++", "GCC"]
categories: ["编程技术"]
authors: ["chase"]
summary: "libstdc++.so.6 -> version `GLIBCXX_3.4.30‘ not found 解决方案"
showToc: true
TocOpen: true
hidemeta: false
comments: false
---

这个错误表明你的 libstdc++.so.6 库版本不满足 libRLIA.so 的要求。具体来说，缺少 GLIBCXX_3.4.30 版本。以下是一些可能的解决方案：
# 解决方案 1：更新 libstdc++
你可以尝试更新 libstdc++ 库，以确保它包含所需的 GLIBCXX_3.4.30 版本。

对于 Ubuntu 或 Debian 系统
```bash
sudo apt-get update
sudo apt-get install libstdc++6
```

# 解决方案 2：使用 Anaconda 安装兼容的 libstdc++
你可以通过 Anaconda 安装兼容的 libstdc++ 库：
```bash
conda install -c conda-forge libstdcxx-ng
```
以下是一个实际操作的示例：
```bash
(py310) chase:~/dexforce/rlia (dev)$ conda install -c conda-forge libstdcxx-ng
Channels:
 - conda-forge
 - defaults
Platform: linux-64
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /home/chase/anaconda3/envs/py310

  added / updated specs:
    - libstdcxx-ng


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    ca-certificates-2025.1.31  |       hbcca054_0         154 KB  conda-forge
    openssl-3.4.0              |       h7b32b05_1         2.8 MB  conda-forge
    ------------------------------------------------------------
                                           Total:         3.0 MB

The following NEW packages will be INSTALLED:

  libgcc             conda-forge/linux-64::libgcc-14.2.0-h77fa898_1 
  libstdcxx          conda-forge/linux-64::libstdcxx-14.2.0-hc0a3c3a_1 

The following packages will be UPDATED:

  ca-certificates    pkgs/main::ca-certificates-2024.12.31~ --> conda-forge::ca-certificates-2025.1.31-hbcca054_0 
  libgcc-ng          pkgs/main::libgcc-ng-11.2.0-h1234567_1 --> conda-forge::libgcc-ng-14.2.0-h69a702a_1 
  libgomp              pkgs/main::libgomp-11.2.0-h1234567_1 --> conda-forge::libgomp-14.2.0-h77fa898_1 
  libstdcxx-ng       pkgs/main::libstdcxx-ng-11.2.0-h12345~ --> conda-forge::libstdcxx-ng-14.2.0-h4852527_1 
  openssl              pkgs/main::openssl-3.0.15-h5eee18b_0 --> conda-forge::openssl-3.4.0-h7b32b05_1 

The following packages will be SUPERSEDED by a higher-priority channel:

  _libgcc_mutex           pkgs/main::_libgcc_mutex-0.1-main --> conda-forge::_libgcc_mutex-0.1-conda_forge 
  _openmp_mutex          pkgs/main::_openmp_mutex-5.1-1_gnu --> conda-forge::_openmp_mutex-4.5-2_gnu 


Proceed ([y]/n)? y


Downloading and Extracting Packages:
                                                                                                                                                                                                                                        
Preparing transaction: done                                                                                                                                                                                                             
Verifying transaction: done
Executing transaction: done

```
