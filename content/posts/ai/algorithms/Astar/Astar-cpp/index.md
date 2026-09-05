---
title: '基于三维栅格空间的A*算法流程C++实现'
date: 2022-06-29
lastmod: 2026-09-05
draft: false
tags: ["A*", "Path Planning", "C++"]
categories: ["人工智能"]
authors: ["chase"]
summary: "给出可独立编译的三维栅格 A*，统一 6/26 邻接、欧氏边权、穿角检查和失败返回，并与 Dijkstra 验证路径代价。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "给出可独立编译的三维栅格 A*，统一 6/26 邻接、欧氏边权、穿角检查和失败返回，并与 Dijkstra 验证路径代价。"
contentLanguage: "zh-CN"
reading_prerequisites: "C++17、优先队列与栅格地图"
reading_focus: "先读搜索不变量，再运行自测；修改邻接或代价时同步调整启发函数。"
related_posts:
  - "/posts/ai/algorithms/Astar/Astar-introduction"
  - "/posts/planner/to_mpc_wbc"
math: true
---

## 实现范围与不变量

本文给出可独立编译的 C++17 三维栅格 A*，支持 6 邻接和保守禁止穿角的 26 邻接。输入是静态占用栅格，输出包含起点和终点；无路可走返回空数组，越界参数抛异常。

原实现中的三维邻域索引、对角边代价和 16 位累计代价存在边界问题。这里使用标准容器管理节点生命周期，以一致的欧氏边权和启发函数说明搜索本身，不再依赖外部工程头文件或直接 include `.cpp`。

$$
f(n)=g(n)+h(n),\qquad h(n)=\|n-\mathrm{goal}\|_2.
$$

轴向、面对角和体对角边分别取 $1,\sqrt2,\sqrt3$，对应单位栅格尺寸。若真实分辨率为 $s$，几何长度乘以 $s$；各轴分辨率不同则需同时调整边权与启发函数。

## 完整代码：astar_demo.cpp

```cpp
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

using Cell = std::array<int, 3>;

class Grid {
public:
    std::array<int, 3> dimensions;
    std::vector<unsigned char> occupied;

    explicit Grid(std::array<int, 3> size) : dimensions(size) {
        std::size_t count = 1;
        for (int value : size) {
            if (value <= 0 || count > 10000000u / static_cast<unsigned>(value))
                throw std::invalid_argument("Invalid or excessive grid size");
            count *= static_cast<std::size_t>(value);
        }
        occupied.resize(count, 0);
    }

    bool inside(const Cell& p) const {
        for (int axis = 0; axis < 3; ++axis)
            if (p[axis] < 0 || p[axis] >= dimensions[axis]) return false;
        return true;
    }

    std::size_t id(const Cell& p) const {
        return (static_cast<std::size_t>(p[2]) * dimensions[1] + p[1])
               * dimensions[0] + p[0];
    }

    Cell cell(std::size_t index) const {
        Cell p{};
        p[0] = static_cast<int>(index % dimensions[0]);
        index /= dimensions[0];
        p[1] = static_cast<int>(index % dimensions[1]);
        p[2] = static_cast<int>(index / dimensions[1]);
        return p;
    }

    bool free(const Cell& p) const {
        return inside(p) && occupied[id(p)] == 0;
    }

    void block(const Cell& p) {
        if (!inside(p)) throw std::out_of_range("Obstacle outside grid");
        occupied[id(p)] = 1;
    }
};

double distance(const Cell& a, const Cell& b) {
    double squared = 0;
    for (int axis = 0; axis < 3; ++axis) {
        const double delta = static_cast<double>(a[axis]) - b[axis];
        squared += delta * delta;
    }
    return std::sqrt(squared);
}

bool clear_step(const Grid& grid, const Cell& current, const Cell& delta) {
    // 检查组合移动涉及的所有轴向/面对角中间格，保守禁止穿边穿角。
    for (unsigned mask = 1; mask < 8; ++mask) {
        Cell touched = current;
        for (unsigned axis = 0; axis < 3; ++axis)
            if (mask & (1u << axis)) touched[axis] += delta[axis];
        if (!grid.free(touched)) return false;
    }
    return true;
}

std::vector<Cell> astar(const Grid& grid, const Cell& start, const Cell& goal,
                        bool diagonal = false, bool use_heuristic = true) {
    if (!grid.inside(start) || !grid.inside(goal))
        throw std::invalid_argument("Endpoint outside grid");
    if (!grid.free(start) || !grid.free(goal)) return {};

    struct Entry { double f, g; std::size_t id; };
    struct Greater {
        bool operator()(const Entry& a, const Entry& b) const {
            return a.f > b.f;
        }
    };
    const auto count = grid.occupied.size();
    std::vector<double> cost(count, std::numeric_limits<double>::infinity());
    std::vector<std::size_t> parent(count, count);
    std::priority_queue<Entry, std::vector<Entry>, Greater> open;
    const auto source = grid.id(start);
    const auto target = grid.id(goal);
    const auto heuristic = [&](const Cell& p) {
        return use_heuristic ? distance(p, goal) : 0.0;
    };
    cost[source] = 0;
    open.push({heuristic(start), 0, source});

    while (!open.empty()) {
        const auto current = open.top();
        open.pop();
        if (current.g != cost[current.id]) continue; // 丢弃过期队列条目
        if (current.id == target) {
            std::vector<Cell> path;
            for (auto index = target; index != count; index = parent[index])
                path.push_back(grid.cell(index));
            std::reverse(path.begin(), path.end());
            return path;
        }
        const Cell point = grid.cell(current.id);
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int axes = (dx != 0) + (dy != 0) + (dz != 0);
                    if (axes == 0 || (!diagonal && axes != 1)) continue;
                    const Cell delta{dx, dy, dz};
                    const Cell next{point[0] + dx, point[1] + dy, point[2] + dz};
                    if (!clear_step(grid, point, delta)) continue;
                    const auto next_id = grid.id(next);
                    const double candidate = current.g + std::sqrt(axes);
                    if (candidate >= cost[next_id]) continue;
                    cost[next_id] = candidate;
                    parent[next_id] = current.id;
                    open.push({candidate + heuristic(next), candidate, next_id});
                }
    }
    return {};
}

double path_length(const std::vector<Cell>& path) {
    double result = 0;
    for (std::size_t i = 1; i < path.size(); ++i)
        result += distance(path[i - 1], path[i]);
    return result;
}

int main() {
    Grid grid({5, 5, 3});
    const Cell start{0, 0, 0}, goal{4, 4, 2};
    const auto axial = astar(grid, start, goal);
    assert(axial.front() == start && axial.back() == goal);
    assert(std::abs(path_length(axial) - 10.0) < 1e-10);
    assert(astar(grid, start, start).size() == 1);

    grid.block({2, 2, 1});
    const auto guided = astar(grid, start, goal, true);
    const auto dijkstra = astar(grid, start, goal, true, false);
    assert(!guided.empty());
    assert(std::abs(path_length(guided) - path_length(dijkstra)) < 1e-10);

    Grid corner({2, 2, 1});
    corner.block({1, 0, 0});
    corner.block({0, 1, 0});
    assert(astar(corner, {0, 0, 0}, {1, 1, 0}, true).empty());
    std::cout << "A* checks passed; path length = " << path_length(guided) << '\n';
}
```

## 编译与基本验证

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic astar_demo.cpp -o astar_demo
./astar_demo
```

示例断言覆盖轴向距离、起终点相同、障碍绕行、与 Dijkstra 代价对照，以及不允许穿过被障碍封住的对角角隙。测试时不要加 `-DNDEBUG`，否则 assert 会被移除。

## 为什么这些处理重要

- 优先队列没有 decrease-key 时，可推入新条目，在弹出时跳过旧代价；不能只改数组里的 g 而不修复队列顺序。
- 目标弹出才结束，而不是目标第一次被发现就结束。
- 三维展开索引使用 `size_t`，并在分配前限制网格总量；不是把每个坐标都改成无符号类型。
- 26 邻接的体对角长度不等于轴向长度，曼哈顿距离也不是允许对角移动时的合适下界。
- 本例禁止穿角的规则较保守；换用其他碰撞语义时，必须连同测试一起调整。

## 工程扩展边界

机器人有体积时，应使用配置空间障碍膨胀或实际几何的碰撞检查。此实现不包含动态障碍、姿态搜索、关节限位或时间参数化。

大地图可评估稀疏节点存储、堆索引与缓存布局，但先保持与 Dijkstra 的结果一致，再比较扩展节点数、峰值内存与总耗时。路径平滑后也必须重新检查整段碰撞。


## 阅读自测与验收

- 把启发函数设为零后，路径形状可以不同，但同一邻接和代价模型下的最短路径代价应一致。随机障碍地图比只测空地图更容易暴露父节点和队列更新错误。
- 人为封住起点、终点或对角中间格，确认返回失败；改变机器人半径后应重新膨胀障碍，点机器人通过并不代表实体机器人能通过。
