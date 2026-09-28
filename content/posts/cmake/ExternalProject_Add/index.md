---
title: 'ExternalProject_Add 使用手册与文档详解'
date: 2025-04-08
lastmod: 2026-09-28
draft: false
tags: ["CMake", "ExternalProject", "Build Systems"]
categories: ["编程开发"]
authors: ["chase"]
summary: "讲解 ExternalProject 的构建期行为、依赖与产物声明，用离线示例验证首次构建、源码修改和增量链接，并梳理 ABI 与安装路径问题。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "讲解 ExternalProject 的构建期行为、依赖与产物声明，用离线示例验证首次构建、源码修改和增量链接，并梳理 ABI 与安装路径问题。"
contentLanguage: "zh-CN"
reading_prerequisites: "CMake target、编译与链接"
reading_focus: "先完成全新目录构建，确认消费者依赖外部产物，再接入真实第三方库。"
related_posts:
  - "/posts/cpp/gccs"
  - "/posts/cpp/gcc"
---

## 关键区别：在构建期运行外部项目

`ExternalProject_Add` 来自 `ExternalProject` 模块，先 `include(ExternalProject)` 才能使用。它为下载、配置、构建、安装等步骤创建目标，但不会像 `add_subdirectory` 那样把外部库的 CMake target 自动带进当前作用域。

因此，“调用 ExternalProject 后立刻 find_package”通常会在第一次配置时失败：那时依赖还没有安装。需要选择 superbuild、显式 imported target，或更适合配置期集成的 FetchContent。

## 目录与步骤速查

| 选项 | 作用 | 易错点 |
| --- | --- | --- |
| `PREFIX` | 组织外部项目的工作目录 | 不代表必然有 `prefix/install` 子目录 |
| `SOURCE_DIR` | 外部源码目录 | 不把已有工作树当下载器的可清理目录 |
| `BINARY_DIR` | 独立构建目录 | 优先 out-of-source 构建 |
| `INSTALL_DIR` | 提供安装目录占位符 | 仍需传给外部项目的 `CMAKE_INSTALL_PREFIX` |
| `DEPENDS` | 指定其他目标依赖 | 链接消费者也需正确依赖外部构建产物 |
| `BUILD_BYPRODUCTS` | 声明构建阶段生成物 | Ninja 等生成器需要知道谁生成库文件 |
| `BUILD_ALWAYS` | 每次进入外部项目的构建步骤 | 本地源码可编辑时，让外部构建系统重新检查文件依赖 |
| `LOG_*` | 保存步骤输出 | 下载、配置与编译错误分开定位 |

## 可在本地复现的最小示例

以下是 Linux、单配置 Ninja/Makefiles 下的四个文件，不需要下载第三方仓库。演示库使用自己的源代码，实际外部项目需核对安装布局和编译选项。

### 主项目 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(external_demo LANGUAGES CXX)
include(ExternalProject)

set(HELLO_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/external/hello")
set(HELLO_BUILD "${CMAKE_CURRENT_BINARY_DIR}/hello-build")

ExternalProject_Add(hello_ep
  SOURCE_DIR "${HELLO_SOURCE}"
  BINARY_DIR "${HELLO_BUILD}"
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  BUILD_ALWAYS TRUE
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=Release
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS "${HELLO_BUILD}/libhello.a"
  LOG_CONFIGURE TRUE
  LOG_BUILD TRUE
  LOG_OUTPUT_ON_FAILURE TRUE
)

add_library(hello_imported STATIC IMPORTED GLOBAL)
set_target_properties(hello_imported PROPERTIES
  IMPORTED_LOCATION "${HELLO_BUILD}/libhello.a"
)
add_dependencies(hello_imported hello_ep)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE hello_imported)
```

### external/hello/CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(hello LANGUAGES CXX)
add_library(hello STATIC hello.cpp)
```

### 两个 C++ 源文件

`external/hello/hello.cpp`：

```cpp
int hello() { return 42; }
```

`main.cpp`：

```cpp
#include <iostream>
int hello();
int main() {
    const int value = hello();
    std::cout << value << '\n';
    return 0;
}
```

构建与运行：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
./build/app
```

预期输出 `42`。多配置生成器、Windows 库名、Debug 后缀和共享库运行路径不在此最小示例的范围内。

### 修改了源码，为什么还在运行旧库 {#editable-source-rebuild}

`ExternalProject` 用步骤及其完成标记管理构建；外部工程中每一个 `.cpp` 的依赖关系，并不会自动展开到主工程。对于这个可手动修改的 `SOURCE_DIR`，如果省略 `BUILD_ALWAYS TRUE`，首次构建后再编辑 `hello.cpp`，主工程可能认为外部步骤已经完成，继续链接旧的 `libhello.a`。

`BUILD_ALWAYS TRUE` 让每次主构建都调用一次外部工程的构建工具。**它不等于每次全量编译**：外部 Make/Ninja 仍负责判断哪个源文件需要重编。`BUILD_BYPRODUCTS` 解决的是“谁生成这个库文件”，也不能单独承担“监视外部源码变化”的职责。

首次得到 `42` 后，进行一次真正改变输入的检查：

```bash
python3 - <<'PY'
from pathlib import Path
source = Path("external/hello/hello.cpp")
text = source.read_text()
assert "return 42;" in text
source.write_text(text.replace("return 42;", "return 43;"))
PY
cmake --build build --parallel 2
./build/app
```

此时应输出 `43`，主程序的源文件没有变化。再运行一次构建，外部构建步骤仍会执行，但库和主程序通常不需要重新编译或链接。这里的 `app` 用退出码表示正常执行；输出是否等于预期值，由检查脚本单独判断。

在 CMake 3.22.1、GNU Make 的本地复现中，省略该选项时输出保持 `42`，启用后更新为 `43`。固定版本、不可编辑的下载依赖可以采用其他更新策略；不能把这个本地开发配置直接理解为所有外部依赖都必须反复重建。补上配置后的全新构建、无改动构建、修改源码和删除库产物四项检查，也在 Ninja 1.13.2 上通过。该选项的适用场景也见 [CMake 的 BUILD_ALWAYS 说明](https://cmake.org/cmake/help/latest/module/ExternalProject.html#build-step-options)。

## 下载第三方依赖时

Git 源尽量固定提交；使用 `GIT_SHALLOW TRUE` 时不能任意指定历史提交哈希，需遵循该选项对分支/标签的限制。URL 下载提供完整 SHA256，`3308f84...` 之类省略值不能运行。

`INACTIVITY_TIMEOUT` 针对支持它的下载行为，不是任意编译进程的通用超时。不要用 `BUILD_COMMAND make` 写死生成器；普通 CMake 外部项目可使用默认构建步骤。

离线复现还需要保存源包和依赖版本。不要声称加上某个选项就自动解决所有网络、ABI 或工具链差异。

## FCL / libccd / Eigen 集成检查

原工程中 FCL 依赖 libccd、Eigen 等组件，集成时需明确：

1. 外部依赖的先后构建顺序及消费者 target 的依赖。
2. 每个库的头文件、库文件与 CMake package 安装位置。
3. 静态/动态链接、位置无关代码和运行时搜索路径。
4. 同一二进制边界上的 C++ 标准库 ABI 与编译选项。

`_GLIBCXX_USE_CXX11_ABI=0` 不是“兼容旧 GCC”的万能开关；只有链接边界确实要求旧 libstdc++ dual ABI 时才采用，并确保相关二进制一致。不要因为头文件库 Eigen 出现在依赖列表里，就把所有 ABI 问题归结为它。

参考：[CMake ExternalProject 官方文档](https://cmake.org/cmake/help/latest/module/ExternalProject.html)、[GCC dual ABI 说明](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html)。


## 阅读自测与验收

- 使用全新构建目录和第二次增量构建分别运行 app；只有增量构建成功时，应检查是否依赖了旧库产物。再将外部源码中的 `42` 改为 `43`，确认主程序输出也随之变化。
- 切换编译器、构建类型或生成器时使用独立目录，核对导入库路径和 BUILD_BYPRODUCTS，不能只看头文件是否可找到。
