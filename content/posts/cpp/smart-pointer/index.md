---
title: C++ 智能指针学习总结
date: 2025-01-27
lastmod: 2025-01-27
draft: false
tags: ["C++"]
categories: ["编程技术"]
authors: ["chase"]
summary: C++ 智能指针学习总结
showToc: true
TocOpen: true
hidemeta: false
comments: false
---

`智能指针`是C++11引入的一种用于`自动管理动态内存`的工具。它们通过`RAII（Resource Acquisition Is Initialization）`机制，确保在智能指针对象超出作用域时，自动释放所管理的资源，从而避免内存泄漏。C++标准库提供了几种常用的智能指针类型：`std::unique_ptr`、`std::shared_ptr`和`std::weak_ptr`。


`RAII（Resource Acquisition Is Initialization，资源获取即初始化）`是一种C++编程技术，用于管理资源的生命周期。该技术的核心思想是将资源的获取与对象的生命周期绑定在一起，通过对象的构造函数获取资源，并在对象的析构函数中释放资源。这样可以确保资源在对象的生命周期内始终有效，并在对象销毁时自动释放资源，从而避免资源泄漏和其他资源管理问题。

### RAII的工作原理
1. **资源获取**：在对象的构造函数中获取资源（如内存、文件句柄、网络连接等）。
2. **资源释放**：在对象的析构函数中释放资源。
3. **作用域管理**：对象在其作用域结束时自动调用析构函数，从而自动释放资源。

### 示例
以下是一个使用RAII管理动态内存的示例：

```cpp
#include <iostream>

class RAIIExample {
public:
    // 构造函数：获取资源
    RAIIExample(size_t size) {
        data = new int[size];
        std::cout << "Resource acquired\n";
    }

    // 析构函数：释放资源
    ~RAIIExample() {
        delete[] data;
        std::cout << "Resource released\n";
    }

private:
    int* data;
};

int main() {
    {
        RAIIExample example(10); // 在作用域内创建对象，获取资源
        // 使用资源
    } // 作用域结束，自动调用析构函数，释放资源

    return 0;
}
```

在这个示例中，`RAIIExample`类的构造函数获取动态内存资源，而析构函数释放资源。当`example`对象超出其作用域时，析构函数会自动调用，从而释放资源。

### RAII的优点
1. **自动资源管理**：通过对象的生命周期自动管理资源，减少手动管理资源的复杂性。
2. **异常安全**：在异常情况下，析构函数仍会被调用，从而确保资源被正确释放。
3. **代码简洁**：通过RAII技术，可以使代码更加简洁和易于维护。

### RAII在智能指针中的应用
C++11引入的智能指针（如`std::unique_ptr`和`std::shared_ptr`）就是RAII技术的典型应用。智能指针在构造函数中获取资源，并在析构函数中释放资源，从而自动管理动态内存的生命周期。

#### 示例
```cpp
#include <memory>
#include <iostream>

class MyClass {
public:
    MyClass() { std::cout << "MyClass Constructor\n"; }
    ~MyClass() { std::cout << "MyClass Destructor\n"; }
};

int main() {
    {
        std::unique_ptr<MyClass> ptr = std::make_unique<MyClass>();
        // 使用资源
    } // 作用域结束，自动调用析构函数，释放资源

    return 0;
}
```

在这个示例中，`std::unique_ptr`在其作用域结束时自动释放所管理的`MyClass`对象，从而避免内存泄漏。

RAII技术通过将资源管理与对象生命周期绑定在一起，提供了一种简洁、安全和高效的资源管理方式。

### `std::unique_ptr`
`std::unique_ptr`是独占所有权的智能指针，意味着同一时间只能有一个`std::unique_ptr`实例拥有某个对象的所有权。它不能被复制，但可以通过`std::move`转移所有权。

#### 示例
```cpp
#include <memory>
#include <iostream>

class MyClass {
public:
    MyClass() { std::cout << "MyClass Constructor\n"; }
    ~MyClass() { std::cout << "MyClass Destructor\n"; }
};

int main() {
    std::unique_ptr<MyClass> ptr1 = std::make_unique<MyClass>();
    // std::unique_ptr<MyClass> ptr2 = ptr1; // 错误：不能复制
    std::unique_ptr<MyClass> ptr2 = std::move(ptr1); // 转移所有权
    return 0;
}
```

### `std::shared_ptr`
`std::shared_ptr`是共享所有权的智能指针，多个`std::shared_ptr`实例可以共享同一个对象的所有权。对象会在最后一个`std::shared_ptr`销毁时被释放。

#### 示例
```cpp
#include <memory>
#include <iostream>

class MyClass {
public:
    MyClass() { std::cout << "MyClass Constructor\n"; }
    ~MyClass() { std::cout << "MyClass Destructor\n"; }
};

int main() {
    std::shared_ptr<MyClass> ptr1 = std::make_shared<MyClass>();
    std::shared_ptr<MyClass> ptr2 = ptr1; // 共享所有权
    return 0;
}
```

### `std::weak_ptr`
`std::weak_ptr`是一种不拥有对象所有权的智能指针，它必须从`std::shared_ptr`构造。`std::weak_ptr`不会影响对象的生命周期，主要用于打破循环引用。

#### 示例
```cpp
#include <memory>
#include <iostream>

class MyClass {
public:
    MyClass() { std::cout << "MyClass Constructor\n"; }
    ~MyClass() { std::cout << "MyClass Destructor\n"; }
};

int main() {
    std::shared_ptr<MyClass> sharedPtr = std::make_shared<MyClass>();
    std::weak_ptr<MyClass> weakPtr = sharedPtr; // 不影响对象生命周期

    if (auto ptr = weakPtr.lock()) { // 检查对象是否仍然存在
        std::cout << "Object is still alive\n";
    } else {
        std::cout << "Object has been destroyed\n";
    }

    return 0;
}
```

### 区别总结
- **`std::unique_ptr`**：独占所有权，不能复制，只能通过`std::move`转移所有权。
- **`std::shared_ptr`**：共享所有权，多个`std::shared_ptr`可以共享同一个对象，使用引用计数管理对象生命周期。
- **`std::weak_ptr`**：不拥有对象所有权，不影响对象生命周期，主要用于打破循环引用。

智能指针在C++中提供了更安全和方便的内存管理方式，避免了手动管理动态内存带来的复杂性和潜在的内存泄漏问题。


