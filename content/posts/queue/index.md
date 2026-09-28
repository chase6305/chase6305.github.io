---
title: '无锁队列：SPSC 内存序、背压与机器人数据新鲜度'
date: 2025-04-01
lastmod: 2026-09-28
draft: false
tags: ["Lock-Free Queue", "Concurrency", "C++"]
categories: ["编程开发"]
authors: ["chase"]
summary: "区分线程安全、无锁与实时性，解释 SPSC 的 acquire/release 发布关系，再用离散事件实验比较积压、丢弃策略和机器人状态的数据年龄。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "区分线程安全、无锁与实时性，解释 SPSC 的 acquire/release 发布关系，再用离散事件实验比较积压、丢弃策略和机器人状态的数据年龄。"
contentLanguage: "zh-CN"
reading_prerequisites: "线程、原子操作与生产者消费者模型"
reading_focus: "确认单生产者单消费者前提，先验证顺序与收尾，再分析内存序和吞吐。"
related_posts:
  - "/posts/cpp/smart-pointer"
  - "/posts/dialout/dh"
---

线程安全、无锁（lock-free）和无等待（wait-free）描述的是不同性质。选择队列前，先确定生产者和消费者数量、是否允许阻塞，以及队列满时如何处理。

## 1. 先明确并发契约

| 方案 | 并发契约 | 适合的用途 |
| --- | --- | --- |
| Python `queue.Queue` | 多生产者、多消费者，内部使用锁 | 任务分发、背压、线程间通信 |
| 有界 SPSC 环形队列 | 恰好一个生产者和一个消费者 | 采集线程向处理线程传递固定大小数据 |
| 无锁 MPMC 队列 | 多生产者、多消费者 | 需要经过验证的算法和内存回收机制 |

无锁保证系统整体持续取得进展，不保证每个线程都能在固定时间内完成操作，也不保证比互斥锁更快。CAS 只是原子操作；把头尾指针改成原子变量，仍不能解决节点生命周期问题。

![SPSC 队列中生产者发布数据、消费者读取数据并释放槽位的顺序](assets/spsc-publication.webp "生产者先写数据再发布尾索引；消费者读完数据再发布头索引。图中的 acquire/release 分别建立数据可见性和槽位复用关系。")

## 2. Python：使用标准库线程安全队列

`queue.Queue` 内部使用锁，不能称为无锁队列。也不要先调用 `empty()` 再 `get()`：两次调用之间，队列状态可能被另一个线程改变。需要非阻塞取值时，直接调用 `get_nowait()` 并捕获 `queue.Empty`。[Python 官方文档](https://docs.python.org/3/library/queue.html)

下面用有界队列和结束标记实现完整的生产者—消费者生命周期：

```python
from queue import Queue
from threading import Thread

tasks = Queue(maxsize=8)
STOP = object()


def producer():
    for value in range(10):
        tasks.put(value)
    tasks.put(STOP)


def consumer():
    while True:
        value = tasks.get()
        try:
            if value is STOP:
                return
            print(f"Consumed: {value}")
        finally:
            tasks.task_done()


if __name__ == "__main__":
    writer = Thread(target=producer)
    reader = Thread(target=consumer)
    reader.start()
    writer.start()
    writer.join()
    tasks.join()
    reader.join()
```

预期按顺序输出 `0` 到 `9`，随后两个线程正常退出。`maxsize` 限制积压任务数；队列满时，生产者阻塞形成背压。多个消费者通常需要分别接收到结束标记。

## 3. C++17：有界 SPSC 环形队列

下面的实现限定为一个生产者、一个消费者，预分配所有槽位，不涉及链表节点回收。`Slots` 个槽位保留一个空位，用于区分满与空，因此有效容量为 `Slots - 1`。

```cpp
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <thread>
#include <type_traits>

template <typename T, std::size_t Slots>
class SpscQueue {
    static_assert(Slots >= 2);
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(std::is_nothrow_copy_assignable_v<T>);
    static_assert(std::atomic<std::size_t>::is_always_lock_free,
                  "This example requires lock-free index atomics.");

    std::array<T, Slots> data_{};
    std::atomic<std::size_t> head_{0};  // Only the consumer writes.
    std::atomic<std::size_t> tail_{0};  // Only the producer writes.

public:
    bool try_push(const T& value) noexcept {
        const auto tail = tail_.load(std::memory_order_relaxed);
        const auto next = (tail + 1) % Slots;
        if (next == head_.load(std::memory_order_acquire)) {
            return false;
        }
        data_[tail] = value;
        tail_.store(next, std::memory_order_release);
        return true;
    }

    bool try_pop(T& value) noexcept {
        const auto head = head_.load(std::memory_order_relaxed);
        if (head == tail_.load(std::memory_order_acquire)) {
            return false;
        }
        value = data_[head];
        head_.store((head + 1) % Slots, std::memory_order_release);
        return true;
    }
};

int main() {
    SpscQueue<int, 64> queue;
    constexpr int count = 100000;
    long long sum = 0;

    std::thread writer([&] {
        for (int i = 0; i < count; ++i) {
            while (!queue.try_push(i)) {
                std::this_thread::yield();
            }
        }
    });
    std::thread reader([&] {
        for (int i = 0; i < count; ++i) {
            int value;
            while (!queue.try_pop(value)) {
                std::this_thread::yield();
            }
            assert(value == i);  // Check order and detect missing values.
            sum += value;
        }
    });

    writer.join();
    reader.join();
    assert(sum == 1LL * count * (count - 1) / 2);
    std::cout << "Consumed " << count << " values; sum = " << sum << '\n';
}
```

保存为 `spsc_queue.cpp`：

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pthread spsc_queue.cpp -o spsc_queue
./spsc_queue
```

预期输出 `Consumed 100000 values; sum = 4999950000`。示例用自旋加 `yield` 演示重试；真实应用应根据延迟、CPU 占用和丢帧策略设计等待方式。

## 4. 为什么需要 acquire/release

生产者写入普通数组后，通过 `tail.store(..., release)` 发布；消费者的 `tail.load(acquire)` 观察到该发布后，才能读取相应槽位。反向的 `head` 同步保证生产者不会覆盖消费者尚未读完的数据。各线程读取自己独占写入的索引时使用 `relaxed`，不承担跨线程发布职责。

队列析构之前，两个线程必须已经退出。不要把本例直接扩展成多生产者或多消费者：多个写者竞争同一索引会破坏所有权约定。

## 5. 链表无锁队列的难点

出队成功后立即 `delete old_head` 是不安全的：其他线程可能仍持有该节点地址，即使随后会检查 CAS 是否成功，也可能已经读取了释放的内存。此类算法通常还需要 hazard pointers、epoch reclamation 等回收机制，并处理 ABA 问题。

需要通用 MPMC 队列时，应选择具有明确并发契约和测试的实现。可对照 [Boost.Lockfree 文档](https://www.boost.org/doc/libs/latest/doc/html/lockfree.html)理解进展保证与内存分配限制。


## 6. 队列不溢出，不代表数据足够新

控制线程读取关节状态时，关心的是“这个测量距现在有多旧”，而不只是队列是否线程安全。一次入队操作很快，也可能把样本放进一条很长的等待队伍。

假设一个 FIFO 中已有 255 条待处理记录，消费者稳定每 5 ms 取一条，且此后不停顿：最后一条要在首次消费之后约 1.27 s 才轮到。这是**排队顺序造成的等待**，还没有算传感器采集、传输和计算延迟；如果消费者可能任意暂停，有限容量也不能提供年龄上界。

应记录采集时间、接收时间、消费时间和序号，分别统计队列等待、端到端数据年龄、丢弃次数及最长无有效输入区间。同一进程内的间隔可使用单调时钟；不同设备上的时间戳必须先有可解释的时钟同步与误差界，不能直接相减两个互不相关的时钟值。

### 6.1 满队列时丢谁，决定了之后读到什么

| 策略 | 适用语义 | 主要代价或条件 |
| --- | --- | --- |
| 阻塞生产者 | 每项必须处理的任务或日志 | 背压可能传回采集线程，必须允许生产者等待 |
| 满时拒绝新样本 | 保留已经接受的顺序记录 | 消费者恢复后仍可能先读到很旧的状态 |
| 满时替换最旧样本 | 最新状态比完整历史更重要 | 需要支持该所有权协议的容器，不能随手修改 SPSC 索引 |
| 消费者批量取出，只使用最新一条 | 可跳过中间测量的状态通道 | 限制每周期取出次数；“当前队列里最新”仍可能已经过期 |
| 单独的最新状态快照 | 每次只需一份一致的当前状态 | 需要正确的发布、读取和对象生命周期协议 |

前文的 SPSC 中，`head` 只能由消费者写。生产者发现队列满后自行递增 `head`，可能覆盖消费者正在读取的普通数组，破坏原来的同步证明。想实现“替换最旧”，应使用明确支持覆盖的设计或带锁容器；不能把下面的单线程策略模型当成这一改法的并发正确性证明。

### 6.2 同一采样条件下比较三种策略

[queue_freshness.py](queue_freshness.py) 使用整数毫秒驱动离散事件：生产者 200 Hz，消费者 50 Hz，容量 8 条，消费者在 `[400,600)` ms 暂停。仿真覆盖 0 到 1000 ms，两端都采样；同一时间先到达、再消费。第三种策略最多取出容量数量的记录，只保留最新一条，并拒绝年龄超过 20 ms 的记录。

```bash
python -m pip install numpy matplotlib
python -B queue_freshness.py --output-dir results
```

| 策略 | 交付条数 | 已交付样本最大年龄 | 溢出丢弃 | 消费者主动跳过 | 过期拒绝 | 结束时仍待处理 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 拒绝新样本 + FIFO | 41 | 355 ms | 153 | 0 | 0 | 7 |
| 替换最旧样本 + FIFO | 41 | 35 ms | 153 | 0 | 0 | 7 |
| 拒绝新样本 + 批量取最新 + 有效期 | 40 | 0 ms | 36 | 124 | 1 | 0 |

第三行的 0 ms 来自本例理想的事件对齐：正常消费时刚好已有同时间到达的样本，且没有网络与计算耗时，**不能解释成实际系统零延迟**。暂停结束的 600 ms 时，队列里最新的一条仍过期，于是没有交付；到 620 ms 才恢复有效交付。有效期约束应和“连续多久没有有效输入”一起验收，不能只看成功交付样本的年龄。

<figure class="article-figure">
{{< post-image src="assets/queue-freshness.png" alt="相同采样时序下三种队列策略的交付数据年龄和待处理数量，灰色区域为消费者暂停区间" >}}
<figcaption><span class="article-figure__number">图 1</span><span class="article-figure__text">离散事件模型的结果，点表示实际交付，空白段没有有效交付。前两种策略的积压条数相同，交付年龄却不同；批量取最新的策略仍需显式拒绝过期样本。该实验不测量并发吞吐或操作系统调度延迟。</span></figcaption>
</figure>

脚本还检查序号递增、容量上限，以及 `201 = 交付 + 各类丢弃 + 待处理` 的样本守恒；[完整事件与统计记录](assets/queue-freshness-results.json)可用于核对表格。实际线程中不要无上限地“读到队列为空”：生产者持续写入时，这个循环可能耗尽控制周期。用固定读取预算，并在处理前后重新检查时间戳。

### 6.3 和 ROS 2 QoS 的关系

ROS 2 的 `history/depth` 控制历史样本数量，`lifespan` 描述消息有效期，`deadline` 描述预期消息间隔；三者分别对应不同问题。`reliable` 不等于新鲜度保证，`deadline` 事件也不是强制终止超时控制回调的调度器。选择时还要检查发布端和订阅端的 QoS 兼容性。[ROS 2 官方 QoS 说明](https://github.com/ros2/ros2_documentation/blob/rolling/source/ROS-Framework/interfaces/topics/About-Quality-of-Service-Settings.rst)

关节状态、图像观测和可跳过的目标更新，可以按业务语义设计丢弃策略；抓取、放置、工位切换等任务命令通常需要顺序、确认和错误恢复。两类信息共用一个“保留最新”的通道，会在负载升高时悄悄改变任务含义。通信层的确认语义见 [TCP 请求与任务完成]({{< relref "/posts/network-protocol/c++_tcp" >}})。

## 阅读自测与验收

- 用很小的环形容量强制频繁绕回，检查满/空状态、顺序和总数；大容量下偶然成功不足以覆盖索引复用。
- 严格保持单生产者、单消费者约束，测试结束时等待线程退出；无锁进展条件不是实时截止时间保证。
