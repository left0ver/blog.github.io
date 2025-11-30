---
title: python的内存管理和GIL
mathjax: true
date: 2025-11-30 17:04:45
tags:
  - python
---

## python的内存管理
### 引用计数
所谓引用计数，就是该对象有多少个变量引用了它，如果其引用计数减为了0，即表示没有变量引用它了，则该对象可以被回收

```python
import sys
a =[1,2]

b = [3,4]

print(sys.getrefcount(a)) # 2
print(sys.getrefcount(b)) #2 
```

> 调用getrefcount函数的时候，会产生一个临时的引用，所以结果会+1
>

```python
del a
del b
```

执行了del a, del b之后，其引用计数为0了，这两个对象会被回收

<!-- more -->

### 标记-清除
python中的垃圾回收主要是靠引用计数来实现，引用计数的方法可以解决大部分的垃圾回收问题，但是有时候我们会遇到循环引用的情况，如下：

```python
import sys
a =[1,2]

b = [3,4]

print(sys.getrefcount(a)) # 2
print(sys.getrefcount(b)) #2 
a.append(b)
b.append(a)

print("*"*100)
print(sys.getrefcount(a)) # 3
print(sys.getrefcount(b)) # 3
```

```python
del a
del b
```

执行了del a ，b之后，引用计数为1，并不会被回收，如果单靠引用计数的机制的话，碰到循环引用的情况就会导致内存泄漏。

顾名思义，标准的标记-清除在进行垃圾回收时分成了两步：

1. 标记： 垃圾回收器（GC）从一组“根对象”开始遍历。根对象通常包括全局变量、当前调用栈中的变量、寄存器等（即程序肯定能访问到的对象）。从根对象出发，顺着引用链遍历所有能被访问到的对象，都被标记为可达的对象。
2. 检查所有对象，如果某个对象没有被标记**（即"不可达"），**则说明这个对象没有变量引用它，可以被回收

**python中的实现**

python中的标记回收算法主要关注容器对象（list，dict，tuple，class）等，因为只有容器对象才会产生循环引用，像字符串，整数是不会产生循环引用的

Python 的 GC 会将所有可能产生循环引用的对象（容器对象）放到一个特殊的双向链表中进行监控。

GC开始时

1. 先复制一份所有对象的当前的引用计数，我们记为`ref_count`
2. 遍历链表中的所有对象，将它引用的其他对象的引用计数-1，即`ref_count -1 `
3. 最后如果`ref_count ==0` , 说明该对象只被循环结构内部引用，它**可能是垃圾，**如果`ref_count > 0`，则说明该对象还被外部的遍历所引用，则它不是垃圾
4. 从那些`ref_count > 0`的对象出发，将他们引用的对象重新标记为"存活"状态，剩下的`ref_count == 0`且没有被重新标记的对象则是真正需要被回收的对象

#### 例子
1. 无外部变量引用的例子

![](https://img.leftover.cn/img-md/202511301703000.png)

2. 有外部变量引用了的例子

![](https://img.leftover.cn/img-md/202511301703347.png)

上面的图中，x还引用了a变量，因此一开始a的引用计数为3，b的引用计数为2，`del a,b`之后，引用计数-1，此时引用计数不为0，不会被回收；



![](https://img.leftover.cn/img-md/202511301704213.png)

之后gc开始时，执行标记-清除算法，按照上面标记-清除算法的逻辑，a和b的`ref_count`最后分别为1，0。因为有个外部的变量引用了a；

找到`ref_count > 0 `的对象，将其引用的对象重新标记为"活动"，因为某个对象还存活的话，其引用的对象肯定不能被回收。

所以最后a和b都不会被回收

![](https://img.leftover.cn/img-md/202511301704225.png)

如果我们之后还执行了`del x`，则之后执行标记-清除算法的时候，a和b的`ref_count==0`，他们两都会被回收

### 分代回收
<font style="color:rgb(83, 88, 97);">上面描述的垃圾回收的阶段，会暂停整个应用程序，等待标记清除结束后才会恢复应用程序的运行。</font>

上面**引用计数+标记-清除**算法以及可以解决垃圾回收的问题了，但是标记清除算法很费时间，因此通常会结合`分代回收`机制一起使用

> <font style="color:rgb(83, 88, 97);">分代回收是基于这样的一个统计事实，对于程序，存在一定比例的内存块的生存周期比较短；而剩下的内存块，生存周期会比较长，甚至会从程序开始一直持续到程序结束。生存期较短对象的比例通常在 80%～90%之间。 因此，简单地认为：对象存在时间越长，越可能不是垃圾，应该越少去收集。这样在执行标记-清除算法时可以有效减小遍历的对象数，从而提高垃圾回收的速度，</font>**<font style="color:rgb(83, 88, 97);">是一种以空间换时间的方法策略</font>**

**python中将对象分为三代：**

1. 第0代：新创建的对象
2. 在第0代的gc扫描中存活下来的对象
3. 在第1代的gc扫描中存活下来的对象

**分代回收的阈值设置：**

```python
import gc
print(gc.get_threshold()) # (700, 10, 10)
```

+ 700=新分配的对象数量-释放的对象数量，第0代gc扫描被触发
+ 第一个10：第0代gc扫描发生10次，则第1代的gc扫描被触发
+ 第二个10：第1代的gc扫描发生10次，则第2代的gc扫描被触发

> 当某一代被扫描时，比它年轻的所有代都会被扫描，因此当第2代被扫描时，则第0，1代中的对象都会触发gc扫描
>

## GIL
GIL（全称：golbal interpreter lock），GIL 并不是python独有的，在Jython中并没有GIL，而在Cpython中有GIL，并且Cpython是大部分环境下默认的执行环境。

上面有将python的内存管理，python通过引用计数的方式来管理内存，简单来说就是记录有多少个变量引用了我这个python对象，当引用计数为0时，则回收内存。这种方式再多线程的情况下如果没有加锁的话并不是线程安全的，因为`--`操作并不是一个原子操作，因此则会出现竞争冒险的问题，导致内存泄漏（计数没有减为0）或者程序崩溃（增加计数的时候少加了，之后在减少计数的时候减为0了然后释放了这个对象的内存，但其实还有变量引用它）

因此当时的python设计者则加了一个全局的锁**（任何python字节码的执行都需要先获取GIL，这样就可以防止死锁并解决了竞争冒险问题）**，这种方式相当于是加了一个粗粒度的锁，相比于给每个分别加锁来说。

> python起源于上世纪90年代初，当时的计算机并不像现在有多核，当时只有单核CPU，只能运行一个线程代码，并没有多核并行这个概念，因此当时使用GIL锁确实是一个不错的选择，但是随着技术的发展，现在多核已经成了标配，我们可以利用多核来加快运算速度.
>

**优点**

1. GIL的实现会简单很多
2. 因为只有一个锁，因此他避免了死锁的问题，当一个线程中存在两个以上的锁的时候可能会导致死锁
3. 在单线程的情况下和不并行的多线程的情况下，它速度更快，因为每个字节码最多只需要要一次锁，如果是给每个对象分别加锁的话，如果一个字节码中需要访问多个对象，那么它需要获取多个锁
4. python的很多扩展是c/c++写的，GIL锁让很多第三方开发者的开发工作容易了很多，他们不需要处理复杂的同步的逻辑

**缺点**

在多线程的时候，对于CPU密集型的任务来说，因为只有一把锁，因此其他所有的线程在执行代码的时候都需要等待获取这个锁，**即使不同的线程是访问的不同的对象，**

+ 因此对于CPU密集型的任务来说，多线程的的速度约等于单线程，并不能利用多核提升运算速度。因此对于CPU密集型的任务来说，我们通常会用多进程的方式，每个进程分别有其对应的GIL锁，并不互相影响

> pytorch中的分布式训练一开始的时候就是使用的多线程的方式，后面切换为了现在的多进程
>

+ 而对于IO密集型的任务来说，由于IO的时间比较久，在执行IO之前，线程会释放掉GIL锁，因此对于IO密集型的任务来说、GIL锁并无太大的影响

## 有无GIL锁的运行时间测试
代码reference from [Python常见面试题(ㄧ)：GIL](https://medium.com/@jesshsieh9/%E3%84%A7-python-%E5%B8%B8%E8%A6%8B%E9%9D%A2%E8%A9%A6%E9%A1%8C-gil-d0b0a63c3271)

### 有GIL
#### IO密集型任务
```python
import threading 
import multiprocessing 
import time 

def  io_task (): 
    time.sleep( 10 ) 
if __name__ == "__main__" : 
    # 单线程
    start_time = time.time() 
    io_task() 
    io_task() 
    print ( f"单线程总执行时间：{time.time() - start_time} " ) 
    
    # 多线程
    start_time = time.time() 
    t1 = threading.Thread(target=io_task) 
    t2 = threading.Thread(target=io_task) 
    t1.start() 
    t2.start() 
    t1.join() 
    t2.join() 
    print ( f"多线程总执行时间：{time.time() - start_time} " ) 
    
    # 多进程
    start_time = time.time() 
    p1 = multiprocessing.Process(target=io_task) 
    p2 = multiprocessing.Process(target=io_task) 
    p1.start() 
    p2.start() 
    p1.join() 
    p2.join() 
    print ( f"多进程总执行时间：{time.time() - start_time} " )
```

![](https://img.leftover.cn/img-md/202511301704210.png)

最终多线程和多进程的时间基本一样

#### CPU密集型任务
```python
import threading 
import multiprocessing 
import time 

def  cpu_task (): 
    return  sum (i** 2  for i in  range ( 100_000_000 )) 
if __name__ == "__main__" : 
    # 单线程
    start_time = time.time() 
    for i in  range ( 2 ): 
        cpu_task() 
    print ( f"单线程总执行时间：{time.time() - start_time} " ) 
    # 多线程
    start_time = time.time() 
    t1 = threading.Thread(target=cpu_task) 
    t2 = threading.Thread(target=cpu_task) 
    t1.start() 
    t2.start() 
    t1.join() 
    t2.join() 
    print ( f"多线程总执行时间：{time.time() - start_time} " ) 
    # 多进程
    start_time = time.time() 
    p1 = multiprocessing.Process(target=cpu_task) 
    p2 = multiprocessing.Process(target=cpu_task) 
    p1.start() 
    p2.start() 
    p1.join() 
    p2.join() 
    print ( f"多进程总执行时间：{time.time() - start_time} " )
```

![](https://img.leftover.cn/img-md/202511301705267.png)

可以看出，对于CPU密集型任务来说，多线程的执行时间甚至比单线程还慢，而使用两个多进程的执行时间约等于单线程的一半

> 在cpu密集型任务中，由于在执行期间GIL锁并不会被释放，因此多线程的效果和单线程一样，同一时间只有一个线程在执行任务
>

### 无GIL
在python3.14中，支持无GIL锁的构建了，我们可以

```powershell
conda create -n py314-no-gil -y
conda activate py314-no-gil
# 下载无GIL锁的python版本
conda install python-freethreading
```

#### IO密集型任务
```python
import threading 
import multiprocessing 
import time 
import sys
def  io_task (): 
    time.sleep( 10 ) 
if __name__ == "__main__" :
        
    # 检查 GIL 是否开启 (Python 3.13+ 新增的 API)
    print(f"GIL Enabled: {sys._is_gil_enabled()}") 
    
    # 单线程
    start_time = time.time() 
    io_task() 
    io_task() 
    print ( f"单线程总执行时间：{time.time() - start_time} " ) 
    
    # 多线程
    start_time = time.time() 
    t1 = threading.Thread(target=io_task) 
    t2 = threading.Thread(target=io_task) 
    t1.start() 
    t2.start() 
    t1.join() 
    t2.join() 
    print ( f"多线程总执行时间：{time.time() - start_time} " ) 
    
    # 多进程
    start_time = time.time() 
    p1 = multiprocessing.Process(target=io_task) 
    p2 = multiprocessing.Process(target=io_task) 
    p1.start() 
    p2.start() 
    p1.join() 
    p2.join() 
    print ( f"多进程总执行时间：{time.time() - start_time} " )
```

无GIL:

![](https://img.leftover.cn/img-md/202511301706127.png)

有GIL：

![](https://img.leftover.cn/img-md/202511301706116.png)

> 对于IO密集型任务来说，有无GIL锁对于程序的执行并无太大的影响
>

#### CPU密集型任务
```python
import threading 
import multiprocessing 
import time 
import sys

def  cpu_task (): 
    return  sum (i** 2  for i in  range ( 100_000_000 )) 
if __name__ == "__main__" :

    # 检查 GIL 是否开启 (Python 3.13+ 新增的 API)
    # 如果返回 False，说明 GIL 已被禁用（成功）
    print(f"GIL Enabled: {sys._is_gil_enabled()}") 

    # 单线程
    start_time = time.time() 
    for i in  range ( 2 ): 
        cpu_task() 
    print ( f"单线程总执行时间：{time.time() - start_time} " ) 
    # 多线程
    start_time = time.time() 
    t1 = threading.Thread(target=cpu_task) 
    t2 = threading.Thread(target=cpu_task) 
    t1.start() 
    t2.start() 
    t1.join() 
    t2.join() 
    print ( f"多线程总执行时间：{time.time() - start_time} " ) 
    # 多进程
    start_time = time.time() 
    p1 = multiprocessing.Process(target=cpu_task) 
    p2 = multiprocessing.Process(target=cpu_task) 
    p1.start() 
    p2.start() 
    p1.join() 
    p2.join() 
    print ( f"多进程总执行时间：{time.time() - start_time} " )
```

无GIL：

![](https://img.leftover.cn/img-md/202511301706299.png)

有GIL：

![](https://img.leftover.cn/img-md/202511301706042.png)

> 可以看出
>
> 1. 无GIL锁的情况下，单线程的运行速度比有GIL锁的时候慢，因为增加了一些开销
> 2. 多线程和多进程的速度差不多都约等于单线程的时间的一半
> 3. 由于没了GIL锁，因此在无GIL锁的情况下，多线程和多进程的效果差不多；而有GIL锁的时候，由于有GIL锁的存在，多线程并没有发挥效果，其执行时间约等于单线程的执行时间
>

## Reference
1. [没白熬夜，终于把Python的内存管理机制搞明白了](https://zhuanlan.zhihu.com/p/164627977)
2. [Python常见面试题(ㄧ)：GIL](https://medium.com/@jesshsieh9/%E3%84%A7-python-%E5%B8%B8%E8%A6%8B%E9%9D%A2%E8%A9%A6%E9%A1%8C-gil-d0b0a63c3271)
3. [【python】天使还是魔鬼？GIL的前世今生。一期视频全面了解GIL！](https://www.bilibili.com/video/BV1za411t7dR/?vd_source=3c93d521158d3aa4f74c71c5140ba8dc)
