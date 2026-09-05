---
title: Python中5个提升效率的f-string高阶技巧
date: 2025-05-03
lastmod: 2026-09-05
draft: false
tags: ["Python", "f-string"]
categories: ["编程开发"]
authors: ["chase"]
summary: "演示 f-string 分隔符、对齐、日期和调试表达式，区分数值精度、显示格式、字符宽度与时区。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "演示 f-string 分隔符、对齐、日期和调试表达式，区分数值精度、显示格式、字符宽度与时区。"
contentLanguage: "zh-CN"
reading_prerequisites: "Python 字符串与基本类型"
reading_focus: "全部示例需要 Python 3.8+；格式化改善展示，不替代精确计算。"
related_posts:
  - "/posts/unicode"
  - "/posts/python/tqdm"
---

## 格式化输出，不是改变数值

f-string 自 Python 3.6 提供；本文的调试表达式 `{value=}` 需要 Python 3.8+。以下五组例子都可独立运行，并用断言检查真实输出，避免把示意注释当成运行结果。

## 一、数字分组与小数位

```python
count = 1425775850
assert f"{count:_}" == "1_425_775_850"
assert f"{count:,}" == "1,425,775,850"
value = 9876543210.5
assert f"{value:,.2f}" == "9,876,543,210.50"
assert f"{-1234.5:,.2f}" == "-1,234.50"
assert value == 9876543210.5  # 格式化没有改写原数值
```

`,`、`_` 是明确的分隔规则，不会自动按当前地区选择千分位或小数符。`.2f` 控制显示到两位小数；不要把格式化字符串继续当作原始高精度数值使用。

## 二、宽度、填充与对齐

```python
item = "APPLE"
assert f"{item:_^15}" == "_____APPLE_____"
label = "价格："
assert f"{label:#>15}" == "#" * 12 + label  # label 是 3 个码点
assert f"{9.99:<8.2f}" == "9.99    "        # 保留末尾 4 个空格
assert f"{'long label':>3}" == "long label"  # 宽度是最小值，不截断
assert f"{'abcdef':.3}" == "abc"            # 字符串精度才会截断
```

`>`、`<`、`^` 分别表示右对齐、左对齐、居中；字段宽度按字符串长度工作，不等于终端显示列数。中文、组合字符和 emoji 的视觉宽度需单独处理，详见 [Unicode 与显示宽度]({{< relref "/posts/unicode" >}})。

## 三、带时区的时间显示

为了可复现，不用 `datetime.now()` 搭配固定的“预期当前时间”注释。下面显式创建 UTC 时间并转换到固定 +08:00 偏移：

```python
from datetime import datetime, timedelta, timezone

utc_time = datetime(2025, 5, 3, 6, 30, 45, tzinfo=timezone.utc)
local_time = utc_time.astimezone(timezone(timedelta(hours=8)))
assert f"{local_time:%Y-%m-%d %H:%M:%S %z}" == "2025-05-03 14:30:45 +0800"
assert local_time.isoformat() == "2025-05-03T14:30:45+08:00"
assert utc_time == local_time  # 同一时刻，不同显示偏移
```

格式中的 `%p`、`%A`、`%B` 受 locale 影响，不能假定所有机器都输出英文。没有时区的 `datetime` 不会因套用格式字符串就自动变成 UTC；固定偏移也不能替代需要夏令时规则的地区时区。

## 四、百分比与动态格式参数

```python
ratio = 0.35782
assert f"{ratio:.1%}" == "35.8%"
assert f"{-0.025:.1%}" == "-2.5%"

width, precision = 10, 3
value = 12.3456
assert f"{value:>{width}.{precision}f}" == "    12.346"
assert f"{{value}} = {value:.2f}" == "{value} = 12.35"
```

`%` 格式会把值乘以 100 再加百分号，输入应是比率，不是已经乘过 100 的数。动态宽度和精度来自不可信输入时应限制范围，避免要求生成极长字符串。

需要十进制精确运算时，应先选择合适的数值类型与舍入规则，再格式化。`f"{2.675:.2f}"` 可能与直觉不同，是因为二进制浮点表示和舍入规则，而不是 f-string 提高了或降低了原值精度。

## 五、调试表达式与 repr

```python
x, y = 5, 3
assert f"{x + y=}" == "x + y=8"
name = "chase"
assert f"{name=}" == "name='chase'"
text = "line1\nline2"
assert f"{text!r}" == "'line1\\nline2'"
assert f"{text!s}" == "line1\nline2"
```

`!r` 有助于看清换行与空格，`!s` 更接近日常显示。调试表达式会包含变量名和值，不应用来记录令牌、口令或大批隐私数据。

f-string 会立即求值；在高频日志中，`logger.debug(f"...{expensive()}...")` 即使最终不输出日志，也已运行表达式。占位式日志能延迟格式化，但函数参数本身仍会先求值，昂贵计算需要先检查日志等级。f-string 也不是 SQL 参数化或 shell 参数转义机制。

## 参考

[Python 格式规范](https://docs.python.org/3/library/string.html#formatspec)、[datetime 文档](https://docs.python.org/3/library/datetime.html)。

## 阅读自测与验收

- 使用正数、负数、很小的数和中文字段比较格式输出；小数位格式只控制显示，不提升底层浮点精度。
- 在日志中区分用于显示的字符串与用于继续计算的数值，避免把格式化后的数据再当作高精度原始值。
