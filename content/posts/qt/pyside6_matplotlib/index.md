---
title: Matplotlib与PySide6兼容性问题及解决方案
date: 2025-08-04
lastmod: 2026-09-05
draft: false
tags: ["PySide6", "Matplotlib"]
categories: ["编程开发"]
authors: ["chase"]
summary: "区分 Matplotlib Agg 与 QtAgg，给出 PySide6 嵌入图表的完整窗口，并说明后端选择、线程与事件循环。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "区分 Matplotlib Agg 与 QtAgg，给出 PySide6 嵌入图表的完整窗口，并说明后端选择、线程与事件循环。"
contentLanguage: "zh-CN"
reading_prerequisites: "PySide6 信号槽与 Matplotlib"
reading_focus: "先验证最小窗口，再加入后台计算；界面与画布更新留在主线程。"
related_posts:
  - "/posts/gui/qt/pyside6"
  - "/posts/gui/qt/qt5"
---

Matplotlib 与 PySide6 集成时，先区分“生成静态图片”和“把可交互画布嵌入 Qt 窗口”。当前 Qt 后端支持 PySide6；发生异常时应核对实际版本、导入顺序和 Qt 绑定，不能直接假定必须降级到旧版 PySide6。

## 1. 记录当前环境

```bash
python -m pip show matplotlib PySide6 PyQt5 PyQt6 toppra
python -m pip check
python -c "import sys, matplotlib; print(sys.executable); print(matplotlib.__version__)"
```

不要仅根据某篇旧教程推断第三方库要求 `matplotlib<3.0`。应查看正在使用的包版本与安装元数据。Matplotlib 2.2.3 与 PySide6 6.2 也不是本文建议的组合。

## 2. 仅保存图片：Agg

必须在导入 `pyplot` 之前选择后端：

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
ax.set(xlabel="x", ylabel="y", title="Headless rendering")
fig.savefig("output.png", dpi=150, bbox_inches="tight")
plt.close(fig)
```

Agg 不创建交互窗口，适合服务器批量出图；它不负责消除其他依赖问题。

## 3. 嵌入 Qt：使用 backend_qtagg

先导入所需的 Qt 绑定，再创建画布，避免环境中其他绑定先被选中：

```python
import sys
import math
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMainWindow
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class PlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Matplotlib + PySide6")
        self.canvas = FigureCanvasQTAgg(Figure(figsize=(6, 4)))
        self.setCentralWidget(self.canvas)
        self.axes = self.canvas.figure.subplots()
        self.line, = self.axes.plot([], [])
        self.axes.set(xlabel="x", ylabel="sin(x + phase)")
        self.phase = 0.0
        self.update_plot()
        self.timer = QTimer(self)  # 随窗口销毁，不使用无父对象的临时定时器。
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def update_plot(self):
        xs = [i * 0.05 for i in range(126)]
        ys = [math.sin(x + self.phase) for x in xs]
        self.line.set_data(xs, ys)  # 复用同一条曲线，不在每次刷新时继续 plot。
        self.axes.relim()
        self.axes.autoscale_view()
        self.canvas.draw_idle()
        self.phase += 0.1

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlotWindow()
    window.show()
    sys.exit(app.exec())
```

保存为 `plot_window.py`，在具有桌面显示权限的环境执行 `python plot_window.py`。需要缩放和平移工具栏时，可添加同一后端中的 `NavigationToolbar2QT`。[Matplotlib Qt 后端文档](https://matplotlib.org/stable/api/backend_qt_api.html)

曲线每 100 ms 请求一次重绘；`draw_idle()` 会等待 GUI 事件循环，必要时合并重绘，并不保证精确 10 FPS。关闭窗口会停止定时器；本示例不复用已经关闭的实例，需要再次打开时创建新窗口。持续自动缩放会影响手动缩放体验；实现交互浏览时应允许暂停刷新或固定坐标范围。

验收时连续更新几十次，检查 `len(window.axes.lines)` 仍为 `1`，关闭后 `window.timer.isActive()` 为 `False`。无桌面环境可用 offscreen 平台做生命周期测试，但仍需在目标桌面验证实际显示。

## 4. 按症状排查

| 症状 | 优先检查 |
| --- | --- |
| 导入时出现 Qt 枚举或符号错误 | 实际导入的 Qt 绑定、版本及包来源 |
| 找不到 xcb 或无法连接 display | 显示会话与平台插件依赖 |
| 能保存图片但没有窗口 | 是否选择了 Agg、是否进入 Qt 事件循环 |
| 更新曲线时卡顿或崩溃 | 是否在主线程操作 GUI、是否重复创建 QApplication |

数据计算可以放在工作线程，界面和画布更新通过 Qt 信号交给主线程。修改依赖后重启 Python 进程，以免旧 Qt 动态库仍留在进程中。


## 阅读自测与验收

- 重复打开、更新和关闭绘图窗口，确认 QApplication、Canvas、定时器与窗口生命周期一致。
- 从工作线程获取数据时通过信号回到 GUI 线程更新界面；刷新频率应与数据速率和绘制成本匹配。
