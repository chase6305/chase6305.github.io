---
title: Python优化控制大寰PGC夹具的串口通信程序
date: 2025-03-01
lastmod: 2026-09-05
draft: false
tags: ["Serial Communication", "Python", "Robot Gripper"]
categories: ["系统与工具"]
authors: ["chase"]
summary: "把 Modbus RTU 请求与响应封装为串口事务，增加 CRC、异常帧、超时和并发检查，并以线程桥接异步调用。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
description: "把 Modbus RTU 请求与响应封装为串口事务，增加 CRC、异常帧、超时和并发检查，并以线程桥接异步调用。"
contentLanguage: "zh-CN"
reading_prerequisites: "Python、串口与 Modbus 基础"
reading_focus: "先用假串口测试协议，再按设备手册接入；示例不自动初始化或移动夹爪。"
related_posts:
  - "/posts/dialout/udev"
  - "/posts/queue"
---

## 先把一次请求—响应作为不可分割的事务

半双工串口同一时刻只应有一个在途事务。仅分别给 `write` 和 `read` 加锁，仍可能出现 A 写、B 写、A 读到 B 响应的串包。

`async def` 不会把 pySerial 的阻塞调用自动变成异步。这里采用“同步事务 + 工作线程”的结构：整个发包、收包和校验都在同一个线程锁内完成，异步调用方通过 `asyncio.to_thread` 等待，不阻塞事件循环。

本文针对 Modbus RTU 单寄存器读写框架，不代替对应 PGC 型号的厂商手册。串口参数、设备 ID、寄存器地址和夹持力范围必须逐项核对；示例不自动执行夹爪初始化或运动。

## 环境与协议约定

Python 3.9+ 自带 `asyncio`，不需要从 pip 安装同名包：

```bash
python -m pip install pyserial
```

使用 `0x03` 读取一个保持寄存器、`0x06` 写一个寄存器。数据字段按大端编码，Modbus RTU CRC 的低字节先发送。必须验证设备地址、功能码、长度、CRC，以及写响应是否回显原请求；收到 8 字节不等于写入成功。

## 同步事务实现

保存为 `gripper_bus.py`。构造函数接收已配置的串口对象，使协议逻辑可以用假串口测试。每个物理总线只能共享一个该对象，不要让多个进程同时打开设备。

```python
import asyncio
import threading
import time


def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for value in data:
        crc ^= value
        for _ in range(8):
            crc = (crc >> 1) ^ (0xA001 if crc & 1 else 0)
    return crc


def with_crc(body: bytes) -> bytes:
    return body + crc16(body).to_bytes(2, "little")


class ModbusRegisterBus:
    def __init__(self, port, unit=1, timeout=0.5, frame_gap=0.004):
        if not 1 <= unit <= 247 or timeout <= 0 or frame_gap <= 0:
            raise ValueError("Invalid unit, timeout, or frame gap")
        self.port = port
        self.unit = unit
        self.timeout = timeout
        self.frame_gap = frame_gap
        self.lock = threading.Lock()
        self.failed = False
        self.last_end = 0.0
        # 限制单次 read/write 阻塞时间；总接收时间另由 deadline 控制。
        self.port.timeout = min(timeout, 0.05)
        self.port.write_timeout = timeout

    def _read_exact(self, length, deadline):
        data = bytearray()
        while len(data) < length:
            if time.monotonic() >= deadline:
                raise TimeoutError("Incomplete Modbus response")
            data.extend(self.port.read(length - len(data)))
        return bytes(data)

    def _exchange(self, function, register, value):
        if function not in (0x03, 0x06):
            raise ValueError("Only single-register 0x03/0x06 are implemented")
        if not 0 <= register <= 0xFFFF or not 0 <= value <= 0xFFFF:
            raise ValueError("Register and value must fit uint16")
        request = with_crc(
            bytes([self.unit, function])
            + register.to_bytes(2, "big")
            + value.to_bytes(2, "big")
        )
        with self.lock:
            if self.failed:
                raise RuntimeError("Bus faulted; inspect and resynchronize before reuse")
            try:
                delay = self.frame_gap - (time.monotonic() - self.last_end)
                if delay > 0:
                    time.sleep(delay)
                if self.port.write(request) != len(request):
                    raise IOError("Incomplete serial write")
                deadline = time.monotonic() + self.timeout
                header = self._read_exact(3, deadline)
                if header[0] != self.unit:
                    raise ValueError("Unexpected device address")
                if header[1] == (function | 0x80):
                    response = header + self._read_exact(2, deadline)
                elif header[1] == function:
                    if function == 0x03:
                        if header[2] != 2:
                            raise ValueError("Expected exactly one register")
                        response = header + self._read_exact(4, deadline)
                    else:
                        response = header + self._read_exact(5, deadline)
                else:
                    raise ValueError("Unexpected function code")
                if crc16(response[:-2]) != int.from_bytes(response[-2:], "little"):
                    raise ValueError("CRC mismatch")
                if header[1] & 0x80:
                    raise RuntimeError(f"Device exception: {header[2]:#04x}")
                if function == 0x06 and response != request:
                    raise ValueError("Write echo mismatch")
                return response
            except Exception:
                # 超时后晚到的旧响应不能当作下一次响应，停止自动发送。
                self.failed = True
                raise
            finally:
                self.last_end = time.monotonic()

    def read_register(self, register):
        response = self._exchange(0x03, register, 1)
        return int.from_bytes(response[3:5], "big")

    def write_register(self, register, value):
        self._exchange(0x06, register, value)

    async def read_register_async(self, register):
        return await asyncio.to_thread(self.read_register, register)

    async def write_register_async(self, register, value):
        await asyncio.to_thread(self.write_register, register, value)

    def close(self):
        with self.lock:
            self.failed = True
            self.port.close()
```

`frame_gap` 是事务间的保守等待值，不是通用波特率配置；应按设备手册、字符时间、RS-485 方向切换和适配器行为调整。这里没有实现完整的 RTU 帧间隔接收器或自动重新同步，不能直接替代经过验证的工业协议栈。

## 不接硬件也能先测 CRC

```python
from gripper_bus import crc16, with_crc

# Modbus 常用测试向量：01 03 00 00 00 0A，CRC 低字节先传
request = bytes.fromhex("01 03 00 00 00 0A")
assert crc16(request) == 0xCDC5
assert with_crc(request).hex(" ") == "01 03 00 00 00 0a c5 cd"
print("CRC test passed")
```

还应模拟短帧、CRC 错误、异常响应、错误设备 ID、超时和两个并发读请求，检查事务不会交错。

## 接入 PGC 时的检查顺序

1. 用稳定串口别名识别设备，确认波特率、校验位、停止位和从站地址。
2. 查对应型号/固件手册，先读取明确标为只读的状态或版本寄存器。
3. 记录请求与响应原始十六进制帧，先验证状态读取，再考虑写入。
4. 初始化、目标位置、速度和夹持力属于设备动作，应增加范围校验、状态确认、急停和安全空间检查。

原笔记中的 `0x0100`、`0x0103` 等地址及 `0–1000` 数值是设备相关配置，不是通用角度单位；没有手册确认时不要发送。

取消等待 `asyncio.to_thread` 的协程不会杀死工作线程。关闭总线、重新打开或恢复发送之前，必须确认在途事务已经结束；超时也不能证明设备未执行上一条命令，所以本例不自动重试运动写入。

参考：[pySerial API 与超时语义](https://pyserial.readthedocs.io/en/latest/pyserial_api.html)、[Modbus 官方规范入口](https://www.modbus.org/modbus-specifications)。


## 阅读自测与验收

- 在假串口中依次注入短帧、错误 CRC、异常码和错误地址；失败后应禁止继续自动发送，尤其不能自动重试运动写入。
- 两个并发调用必须共享同一个总线对象，检查请求和响应不交错；取消协程后仍需等待正在运行的工作线程结束。
