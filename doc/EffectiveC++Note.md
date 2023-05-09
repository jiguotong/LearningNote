# 一、概述
# 二、正文
## 条款01：视C++为一个语言联邦
⭐四个次语言：
C
Object-Oriented c++
template C++
STL
## 条款02：尽量以const，enum，inline替换#define
⭐对于单纯常量，最好以const对象或enums替换#defines
⭐对于形似函数的宏，最好改用inline函数替换#defines
```c
#define CALL_WITH_MAX(a,b) f((a)>(b) ? (a) : (b))

template<typename T>
inline void callWithMax(const T&a, const T&b){
    f(a > b ? a : b);
}
```

## 条款03：尽可能使用const
⭐const出现在星号左边，表示被指物是常量；出现在星号右边，表示指针自身是常量；出现在星号两边，表示被指物和指针两者都是常量。

## 条款04：确定对象被使用前已先被初始化
⭐对于内置类型以外的数据类型，确保每一个构造函数都将对象的每一个成员初始化。
构造函数一：
```c
Base::Base(const string name,int age):m_name(name),m_age(age){

}
```
构造函数二：
```c
Base::Base(const string name,int age){
    m_name = name;
    m_age = age;
}
```
以上两种构造函数的方法，方法一比方法二要高效，因为方法一是初始化，而方法二是先默认初始化再赋值

## 条款05：了解C++默默编写并调用哪些函数
## 条款06：若不想使用编译器自动生成的函数，就该明确拒绝
⭐=delete可以阻止编译器自动生成的函数，例：
```c
Base &operator=(const Base&) = delete;
```

## 条款07：为多态基类声明virtual析构函数
⭐virtual ~Base()；替代~Base();
（1）析构函数定义为虚函数时：基类指针可以指向派生类的对象（多态性），如果删除该指针delete p；就会调用该指针指向的派生类析构函数，而派生类的析构函数又自动调用基类的析构函数，这样整个派生类的对象完全被释放。
（2）析构函数不定义为虚函数时：编译器实施静态绑定，在删除基类指针时，只会调用基类的析构函数而不调用派生类析构函数，这样就会造成派生类对象析构不完全。

## 条款08：不在析构函数中抛出异常

## 条款09：绝不在构造和析构过程中调用virtual函数

⭐
⭐
⭐
⭐
⭐

⭐