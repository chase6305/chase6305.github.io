---
title: Python优化控制大寰PGC夹具的串口通信程序
date: 2025-03-01
lastmod: 2025-03-01
draft: false
tags: ["Linux", "Python", "Gripper"]
categories: ["Gripper"]
authors: ["chase"]
summary: Python优化控制大寰PGC夹具的串口通信程序
showToc: true
TocOpen: true
hidemeta: false
comments: false
---



在工业自动化中，夹具的控制是一个非常重要的环节。本文将介绍如何使用Python通过串口控制大寰PGC夹具。我们将使用异步IO和事件锁来实现半双工通信，以提高通讯效率和鲁棒性。

# 1. 环境准备
首先，我们需要安装pyserial库来处理串口通信。可以使用以下命令安装：
```bash
pip install pyserial asyncio
```
# 2. 异步IO
异步IO（Asynchronous I/O）在处理并发任务时具有显著的优势，尤其是在I/O密集型操作中。以下是异步IO的主要优势和处理逻辑：

## 2.1优势

1. **高效利用资源**：
   异步IO允许程序在等待I/O操作完成时执行其他任务，从而更高效地利用CPU资源。

2. **更好的响应性**：
   异步IO可以提高应用程序的响应性，因为它不会因为等待I/O操作而阻塞整个程序。

3. **简化并发编程**：
   使用异步IO可以避免传统多线程编程中的锁竞争和死锁问题，从而简化并发编程。

4. **可扩展性**：
   异步IO使得应用程序更容易扩展，因为它可以处理大量并发连接而不需要为每个连接创建一个线程。
## 2.2 处理逻辑

异步IO的处理逻辑通常包括以下几个步骤：

1. **定义异步函数**：
   使用`async def`定义异步函数，这些函数可以使用`await`关键字来等待异步操作完成。

2. **创建事件循环**：
   使用`asyncio`库创建和管理事件循环，事件循环负责调度和执行异步任务。

3. **执行异步任务**：
   使用`await`关键字等待异步任务完成，或者使用`asyncio.run()`来运行顶层异步函数。






# 3. 实现串口控制类
我们将创建一个名为`dh_device`的类来处理串口连接和数据读写。以下是`PGC_device.py`的实现：

`PGC_device.py`
```python
import asyncio
import serial
import threading
import logging

logging.basicConfig(level=logging.INFO)


class dh_device(object):
    def __init__(self):
        r"""Initialize the dh_device class with a threading lock."""
        self.lock = threading.Lock()
        self.serialPort = serial.Serial(timeout=1.0)

    def connect_device(self, portname, Baudrate):
        r"""Connect to the device via the specified serial port and baud rate.

        Args:
            portname (str): The name of the serial port to connect to.
            Baudrate (int): The baud rate for the serial communication.

        Returns:
            int: 0 if the connection is successful, -1 otherwise.
        """
        with self.lock:
            try:
                self.serialPort.port = portname
                self.serialPort.baudrate = Baudrate
                self.serialPort.bytesize = 8
                self.serialPort.parity = "N"
                self.serialPort.stopbits = 1
                self.serialPort.set_output_flow_control = False
                self.serialPort.set_input_flow_control = False

                self.serialPort.open()
                if self.serialPort.isOpen():
                    logging.info("Serial Open Success")
                    return 0
                else:
                    logging.error("Serial Open Error")
                    return -1
            except serial.SerialException as e:
                logging.error(f"Serial Open Exception: {e}")
                return -1

    def disconnect_device(self):
        r"""Disconnect the device by closing the serial port."""
        with self.lock:
            if self.serialPort.isOpen():
                self.serialPort.close()
                logging.info("Serial Port Closed")
            else:
                logging.warning("Serial Port Already Closed")

    async def device_write(self, write_data):
        r"""Write data to the device.

        Args:
            write_data (bytes): The data to write to the device.

        Returns:
            int: The number of bytes written if successful, 0 if there is an error, -1 if the serial port is not open.
        """
        with self.lock:
            if self.serialPort.isOpen():
                try:
                    write_length = self.serialPort.write(write_data)
                    if write_length == len(write_data):
                        return write_length
                    else:
                        logging.debug(f"Write error! send_buff: {write_data}")
                        return 0
                except serial.SerialTimeoutException as e:
                    logging.debug(f"Write Timeout Exception: {e}")
                    return 0
            else:
                logging.error("Serial Port Not Open")
                return -1

    async def device_read(self, wlen):
        r"""Read data from the device.

        Args:
            wlen (int): The number of bytes to read from the device.

        Returns:
            bytes: The data read from the device if successful, -1 if the serial port is not open.
        """
        with self.lock:
            if self.serialPort.isOpen():
                try:
                    responseData = self.serialPort.read(wlen)
                    return responseData
                except serial.SerialTimeoutException as e:
                    logging.error(f"Read Timeout Exception: {e}")
                    return -1
            else:
                logging.error("Serial Port Not Open")
                return -1
```

# 4. 实现夹具控制类
接下来，我们创建一个名为`dh_modbus_gripper`的类来实现对夹具的具体控制。以下是`PGC_gripper.py`的实现：
`PGC_gripper.py`
```python
import asyncio
import threading
import logging
from PGC_device import dh_device

logging.basicConfig(level=logging.INFO)


class dh_modbus_gripper(object):
    def __init__(self):
        r"""Initialize the gripper with default ID and create a lock for thread safety."""
        self.gripper_ID = 0x01
        self._transaction_lock = threading.Lock()
        self.device = dh_device()

    def CRC16(self, nData, wLength):
        r"""Calculate the CRC16 checksum for the given data.

        Args:
            nData (list): The data to calculate the checksum for.
            wLength (int): The length of the data.

        Returns:
            int: The calculated CRC16 checksum.
        """
        if nData == 0x00:
            return 0x0000
        wCRCWord = 0xFFFF
        poly = 0xA001
        for num in range(wLength):
            date = nData[num]
            wCRCWord = (date & 0xFF) ^ wCRCWord
            for bit in range(8):
                if (wCRCWord & 0x01) != 0:
                    wCRCWord >>= 1
                    wCRCWord ^= poly
                else:
                    wCRCWord >>= 1
        return wCRCWord

    def open(self, PortName, BaudRate):
        r"""Open the connection to the gripper device.

        Args:
            PortName (str): The name of the port to connect to.
            BaudRate (int): The baud rate for the connection.

        Returns:
            int: 0 if successful, negative value if failed.
        """
        with self._transaction_lock:
            ret = self.device.connect_device(PortName, BaudRate)
            if ret < 0:
                logging.error("Failed to open connection")
            else:
                logging.info("Connection opened successfully")
            return ret

    def close(self):
        r"""Close the connection to the gripper device."""
        with self._transaction_lock:
            self.device.disconnect_device()
            logging.info("Connection closed")

    async def _send_command(self, send_buf, expected_length):
        r"""Send a command to the device and read the response.

        Args:
            send_buf (list): The command to send.
            expected_length (int): The expected length of the response.

        Returns:
            list: The response from the device.
        """
        send_temp = send_buf
        retrycount = 3

        while retrycount > 0:
            wdlen = await self.device.device_write(send_temp)
            if len(send_temp) != wdlen:
                logging.debug(f"Write error! Sent: {send_temp}")
                retrycount -= 1
                continue

            rev_buf = await self.device.device_read(expected_length)
            if len(rev_buf) == expected_length:
                return rev_buf

            retrycount -= 1

        logging.debug("Failed to communicate with device after retries")
        return None

    def _prepare_command(self, function_code, index, value=None):
        r"""Prepare a command buffer with CRC16 checksum.

        Args:
            function_code (int): The function code for the command.
            index (int): The register index.
            value (int, optional): The value to write (for write commands).

        Returns:
            list: The prepared command buffer.
        """
        send_buf = [self.gripper_ID, function_code, (index >> 8) & 0xFF, index & 0xFF]
        if value is not None:
            send_buf.extend([(value >> 8) & 0xFF, value & 0xFF])
        else:
            send_buf.extend([0x00, 0x01])
        crc = self.CRC16(send_buf, len(send_buf))
        send_buf.extend([crc & 0xFF, (crc >> 8) & 0xFF])
        return send_buf

    async def WriteRegisterFunc(self, index, value):
        r"""Write a value to a specific register of the gripper.

        Args:
            index (int): The register index to write to.
            value (int): The value to write to the register.

        Returns:
            bool: True if the write operation was successful, False otherwise.
        """
        with self._transaction_lock:
            send_buf = self._prepare_command(0x06, index, value)
            response = await self._send_command(send_buf, 8)
            return response is not None

    async def ReadRegisterFunc(self, index):
        r"""Read a value from a specific register of the gripper.

        Args:
            index (int): The register index to read from.

        Returns:
            int: The value read from the register.
        """
        with self._transaction_lock:
            send_buf = self._prepare_command(0x03, index)
            response = await self._send_command(send_buf, 7)
            if response:
                return (response[3] << 8) | (response[4] & 0xFF)
            return None

    async def Initialization(self):
        """Initialize the gripper by writing to the initialization register."""
        await self.WriteRegisterFunc(0x0100, 0xA5)

    async def SetTargetPosition(self, refpos):
        r"""Set the target position of the gripper.

        Args:
            refpos (int): The target position to set.
        """
        await self.WriteRegisterFunc(0x0103, refpos)

    async def SetTargetForce(self, force):
        r"""Set the target force of the gripper.

        Args:
            force (int): The target force to set.
        """
        await self.WriteRegisterFunc(0x0101, force)

    async def SetTargetSpeed(self, speed):
        r"""Set the target speed of the gripper.

        Args:
            speed (int): The target speed to set.
        """
        await self.WriteRegisterFunc(0x0104, speed)

    async def GetCurrentPosition(self):
        r"""Get the current position of the gripper.

        Returns:
            int: The current position of the gripper.
        """
        return await self.ReadRegisterFunc(0x0202)

    async def GetCurrentTargetForce(self):
        r"""Get the current target force of the gripper.

        Returns:
            int: The current target force of the gripper.
        """
        return await self.ReadRegisterFunc(0x0101)

    async def GetCurrentTargetSpeed(self):
        r"""Get the current target speed of the gripper.

        Returns:
            int: The current target speed of the gripper.
        """
        return await self.ReadRegisterFunc(0x0104)

    async def GetInitState(self):
        r"""Get the initialization state of the gripper.

        Returns:
            int: The initialization state of the gripper.
        """
        return await self.ReadRegisterFunc(0x0200)

    async def GetGripState(self):
        r"""Get the grip state of the gripper.

        Returns:
            int: The grip state of the gripper.
        """
        return await self.ReadRegisterFunc(0x0201)
```
#  5. 用法
## 5.2 示例代码
以下是如何使用dh_modbus_gripper类来控制大寰PGC夹具的具体示例：
```python
import asyncio
from PGC_gripper import dh_modbus_gripper

# 定义串口和波特率
LEFT_GRIPPER_PORT = "/dev/ttyUSB0"
BAUDRATE = 115200

# 创建夹具控制对象
gripper = dh_modbus_gripper()

# 打开串口连接
gripper.open(LEFT_GRIPPER_PORT, BAUDRATE)

# 初始化夹具
asyncio.run(gripper.Initialization())

# 设置目标位置
angle = 500  # 角度范围为0-1000
asyncio.run(gripper.SetTargetPosition(int(angle)))

# 关闭串口连接
gripper.close()
```
## 5.2 代码说明

1. **定义异步函数**：
   异步函数使用`async def`定义，例如`device_write`和`device_read`。
   ```python
   async def device_write(self, write_data):
       # 异步写入数据逻辑
       pass

   async def device_read(self, wlen):
       # 异步读取数据逻辑
       pass
   ```

2. **创建事件循环**：
   使用`asyncio.run(main())`来创建和运行事件循环。
   ```python
   async def main():
       # 主函数逻辑
       pass

   if __name__ == "__main__":
       asyncio.run(main())
   ```

3. **执行异步任务**：
   在异步函数内部使用`await`关键字等待异步操作完成，例如：
   ```python
   async def main():
       device = dh_device()
       await device.device_write(b'example data')
       response = await device.device_read(10)
       print(response)
   ```

