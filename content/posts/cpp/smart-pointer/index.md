---
title: C++ 智能指针学习总结
date: 2025-01-27
lastmod: 2026-09-28
draft: false
tags: ["C++"]
categories: ["编程开发"]
authors: ["chase"]
summary: "通过 RAII、unique_ptr、shared_ptr 和 weak_ptr 理解资源所有权，再用 C++20 不可变状态快照说明原子发布、对象保活和实时性边界。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "通过 RAII、unique_ptr、shared_ptr 和 weak_ptr 理解资源所有权，再用 C++20 不可变状态快照说明原子发布、对象保活和实时性边界。"
contentLanguage: "zh-CN"
reading_prerequisites: "C++ 类、析构与值语义"
reading_focus: "先决定谁拥有资源，再选择指针类型；引用计数安全不等于对象访问安全。"
related_posts:
  - "/posts/cpp/std"
  - "/posts/queue"
---

## 1. 先回答“谁负责释放资源”

RAII 把资源释放绑定到拥有者的析构，而不是要求每条返回路径都手动 `delete`。普通栈展开会析构已经构造完成的对象，但循环拥有、`std::terminate`、进程被强制终止等情形不能靠“用了智能指针”获得统一保证。

| 需求 | 表达方式 | 生命周期含义 |
| --- | --- | --- |
| 函数内普通对象 | 局部变量、容器 | 通常不必动态分配 |
| 一个拥有者 | `std::unique_ptr<T>` | 可移动，不可复制 |
| 多个共同拥有者 | `std::shared_ptr<T>` | 最后一个强拥有者释放时销毁对象 |
| 观察共享对象 | `std::weak_ptr<T>` | 不延长对象寿命，使用前 lock |
| 临时借用对象 | `T&` / `const T&` / 非拥有原始指针 | 调用方保证生命周期 |

选择 `shared_ptr` 的理由应当是共享所有权，而不是“复制起来方便”。对象的线程安全与所有权管理是两个问题。

## 2. 一份完整的所有权实验

代码要求 C++14+；以下命令采用 C++17。每个断言对应一个生命周期事实，不依赖析构日志的打印顺序。

```cpp
#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>

struct Object {
    static int alive;
    Object() { ++alive; }
    ~Object() { --alive; }
    Object(const Object&) = delete;
    Object& operator=(const Object&) = delete;
};
int Object::alive = 0;

int main() {
    {
        auto first = std::make_unique<Object>();
        auto second = std::move(first);
        assert(!first && second);
        assert(Object::alive == 1);
        // second.get() 只借出指针，不转移所有权。
    }
    assert(Object::alive == 0);

    std::weak_ptr<Object> observer;
    {
        auto owner = std::make_shared<Object>();
        observer = owner;
        auto another = owner;
        assert(owner.use_count() == 2);
        owner.reset();
        assert(Object::alive == 1);
        if (auto temporary_owner = observer.lock()) {
            assert(temporary_owner.get() == another.get());
        }
    }
    assert(observer.expired());
    assert(!observer.lock());
    assert(Object::alive == 0);

    try {
        auto resource = std::make_unique<Object>();
        throw std::runtime_error("demonstrate stack unwinding");
    } catch (const std::runtime_error&) {
        assert(Object::alive == 0);
    }
    std::cout << "ownership checks passed\n";
}
```

```bash
g++ -std=c++17 -Wall -Wextra -Wpedantic ownership.cpp -o ownership
./ownership
```

`std::move` 本身只是转换表达式的值类别；这里真正转移资源的是 `unique_ptr` 的移动构造函数。移动后的 `first` 为空，不应继续解引用。

`weak_ptr` 可以默认构造成空，也可以从兼容的 shared/weak 指针构造，并非必须立即绑定对象。`lock()` 原子地尝试取得强所有权；不要先 `expired()`，再假定对象在下一步仍然存在。[C++ 工作草案中的 weak_ptr 定义](https://eel.is/c++draft/util.smartptr.weak)

## 3. 为什么循环引用仍会泄漏

假设父节点强拥有子节点，子节点也强拥有父节点。即使外部变量都销毁，两边的强引用计数仍可能不为零。一个常见的树结构是：

```cpp
#include <memory>
#include <vector>

struct Node {
    std::vector<std::shared_ptr<Node>> children; // 父节点拥有子节点。
    std::weak_ptr<Node> parent;                 // 子节点只观察父节点。
};
```

这只是所有权模型，不是通用图算法。任意图或异步任务中，是否存在强引用环仍需单独检查。回调捕获 `shared_ptr` 也可能闭合一个环。

## 4. get、release 和 reset 不同

| 操作 | 结果 | 注意事项 |
| --- | --- | --- |
| `get()` | 返回借用的原始指针 | 不要 delete，也不要据此新建独立拥有者 |
| `unique_ptr::release()` | 交出指针并放弃拥有 | 接收方必须接管释放责任，否则泄漏 |
| `reset()` | 释放当前所有权，可接管新对象 | shared_ptr 的其他共同拥有者仍可能存活 |
| `weak_ptr::lock()` | 返回有效 shared_ptr 或空指针 | 成功时在该 shared_ptr 生命周期内保活 |

绝不能把同一个裸指针分别交给两个独立 `shared_ptr` 控制块，否则可能重复释放。需要共享时复制原来的 `shared_ptr`。

若 C API 返回文件、设备等资源，应使用与其分配方式配套的删除器，而不是默认 `delete`。用多态基类管理派生类时，还要确认相应销毁路径是否需要虚析构。

## 5. 并发边界与异常安全

不同 `shared_ptr` 实例可以共同维护同一控制块，但这不自动保护对象字段。多个线程修改同一个普通 `shared_ptr` 变量也需要同步；C++20 提供 `std::atomic<std::shared_ptr<T>>`，但它仍不替对象内部加锁。

`use_count()` 适合本例单线程说明，不适合作为并发环境中“只有我在用”的判定。真实代码通常优先使用容器和默认析构，减少自己持有裸资源和手写复制/移动操作的机会。

构造函数若获取裸资源后又抛异常，尚未完成构造的整个对象不会执行其析构函数。因此应让成员本身使用 RAII，不能依赖稍后才会完成构造的外层析构来补救。


## 6. 机器人状态快照：一致性与所有权一起处理

如果采集线程发布一组相互关联的状态，而消费线程每次只需要最近的一份，可以发布**不可变快照**：先完整构造新对象，再原子地替换共享句柄；读取方取得自己的 `shared_ptr` 副本后，在使用完之前一直保留它。

[C++20 完整例子 immutable_snapshot.cpp](immutable_snapshot.cpp) 使用 `std::atomic<std::shared_ptr<const Snapshot>>`，快照中包含序号和两个必须互相匹配的数值。它先验证旧对象的保活关系，再由两个线程发布、读取 100,000 次更新，检查字段一致、已见序号不倒退，并等待最终状态被观察到。

```bash
g++-12 -std=c++20 -O2 -Wall -Wextra -Wpedantic -pthread immutable_snapshot.cpp -o immutable_snapshot
./immutable_snapshot
```

命令中的 `g++-12` 指本次验证的 GCC 12.3 / libstdc++ 12；其他工具链要同时支持语言模式和标准库特性。仅加 `-std=c++20` 并不保证旧库已实现所有 C++20 接口：本机 GCC 11 的库不能实例化这个原子特化，GCC 12 可以。源码用 `__cpp_lib_atomic_shared_ptr >= 201711L` 做明确检查，不需要修改系统默认编译器。[GCC 的实现记录](https://gcc.gnu.org/pipermail/libstdc%2B%2B-cvs/2022q1/037419.html)

<figure class="article-figure">
{{< post-image src="assets/immutable-snapshot.webp" alt="原子句柄切换到新快照 B 后，读取线程自己的强引用仍保留旧快照 A，直到最后一个强引用释放才销毁 A" >}}
<figcaption><span class="article-figure__number">图 1</span><span class="article-figure__text">图中只考虑画出的强拥有者。发布 B 改变共享句柄的指向，读取者手中的 A 仍然有效；对象销毁取决于最后一个强拥有者释放，而不是发布者是否已经换成新版本。</span></figcaption>
</figure>

### 6.1 为什么这次读取不会混入两个版本

写线程把完整对象通过 `store(..., memory_order_release)` 发布；读线程用 `load(memory_order_acquire)` 得到该句柄时，能看到发布之前完成的初始化。读取期间，对象保持不可变，读线程自己的强引用负责保活。这里分别解决了**初始化可见性、字段一致性和生命周期**三个问题。[C++ 标准草案中的原子 shared_ptr](https://eel.is/c++draft/util.smartptr.atomic.shared)

不要只写 `auto raw = latest.load().get()`，然后在后续语句中使用 `raw`：临时 `shared_ptr` 在该语句结束时就销毁了，发布者替换旧版本后，原始指针可能悬空。也不要以为 `shared_ptr<const T>` 会阻止其他已有可变别名修改对象；本例直接构造 const 对象，没有保留那种别名。

这个通道可以跳过中间版本，读取同一版本多次也合法。因此不能要求“读取次数等于发布次数”，脚本打印的不同版本数会随调度变化。它适合最新状态语义，不等价于保存每条任务的 FIFO；每条任务都必须执行时，应使用有明确背压和收尾协议的[队列]({{< relref "/posts/queue" >}})。

### 6.2 原子共享指针为什么仍不等于硬实时

`std::atomic<std::shared_ptr<T>>` 是否无锁由实现决定，本次环境的 `is_lock_free()` 返回 `false`。即使换一种实现返回 `true`，分配新对象、引用计数、调度，以及最后一个引用释放后的析构与内存回收，仍可能带来不受本例约束的耗时。

读取者可能恰好成为最后一个强拥有者，于是昂贵析构在读取线程发生。硬实时控制循环需要进一步设计预分配、回收线程和严格的等待预算；不能仅凭原子类型或一次吞吐测试认定达标。快照还要携带实际采集时间和有效状态，读取一致的旧数据依然可能超过控制器允许的数据年龄。

## 阅读自测与验收

- 移动 unique_ptr 后，原拥有者应为空；让最后一个 shared_ptr 离开作用域后，weak_ptr::lock() 应返回空。
- 检查回调捕获和父子反向引用是否形成强引用环；引用计数正确与对象成员并发访问正确需要分别验证。
