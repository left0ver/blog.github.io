---
title: Listpack 的实现原理
date: 2024-08-29 15:44:41
tags:
 - Redis
---
# Listpack

## 为什么需要 listpack，listpack 解决了什么问题？

​    Listpack 可以说是用来替代 ziplist 的，zipList 是一种特殊的双向链表（不使用指针来找到它前一个或者后一个元素），它的 entry 是使用 `prebious_entry_length` 记录前一个节点的长度，从而实现从后往前遍历，使用 `encoding` 来记录当前节点的数据类型和长度，从而知道当前节点的长度，实现从前往后遍历。因为使用`prebious_entry_length` 记录前一个节点的长度，因此它有`连锁更新`的问题

​    在 Listpack 中，每一个 entry 只记录自己的长度，因此在新增或者修改元素时，只会涉及当前 entry 的更新，而不会影响到别的 
<!-- more -->

## Listpack 的 entry 的结构

<img src="https://img.leftover.cn/img-md/202408291421524.png" alt="image-20240829142153403" style="zoom:50%;" />

> 为了节省内存，有一些小的数字/字符串会直接存储在 encoding-type 中，不会使用 element-data 存储

- encoding-type：变长的字段，其主要的作用就是**记录数据的类型以及数据的长度**,具体可以看[官方文档](https://github.com/antirez/listpack/blob/master/listpack.md#elements-representation)

- element-data：存储数据

- element-tot-len：表示 entry 的大小(即 encoding-type 和 element-data 的长度)，用于从后往前遍历。

  - 该变量是可变长的，如果 entry 比较小，可能`element-tot-len` 只使用 1B，如果 entry 比较大，就会逐渐增加字节数

  - 该字段是从右往左解析的，例如 element-tot-len 为 0000011 1110100 ，则从最后面的 0 开始解析。
  - 每个字节第一个 bit 表示是否结束，0 表示结束，1 表示没结束，因此只有 7bit 真正有用

  > 例如需要存储长度为 500 的 entry，500 的二进制 111110100
  >
  >因此 element-tot-len 为： 0000011 1110100

## Listpack 如何进行遍历

### 从左往右遍历

根据`encoding-type` 可以知道 encoding-type 和 element-data 的总字节数，从而可以计算出 element-tot-len 的字节数，因此可以得到当前整个 entry 的字节数，从而找到下一个 entry

```c
static inline uint32_t lpCurrentEncodedSizeUnsafe(unsigned char *p) {
    if (LP_ENCODING_IS_7BIT_UINT(p[0])) return 1;
    if (LP_ENCODING_IS_6BIT_STR(p[0])) return 1+LP_ENCODING_6BIT_STR_LEN(p);
    if (LP_ENCODING_IS_13BIT_INT(p[0])) return 2;
    if (LP_ENCODING_IS_16BIT_INT(p[0])) return 3;
    if (LP_ENCODING_IS_24BIT_INT(p[0])) return 4;
    if (LP_ENCODING_IS_32BIT_INT(p[0])) return 5;
    if (LP_ENCODING_IS_64BIT_INT(p[0])) return 9;
    if (LP_ENCODING_IS_12BIT_STR(p[0])) return 2+LP_ENCODING_12BIT_STR_LEN(p);
    if (LP_ENCODING_IS_32BIT_STR(p[0])) return 5+LP_ENCODING_32BIT_STR_LEN(p);
    if (p[0] == LP_EOF) return 1;
    return 0;
}

unsigned char *lpSkip(unsigned char *p) {
  // 计算出当前元素的encoding-type 和 element-data 的总字节数
    unsigned long entrylen = lpCurrentEncodedSizeUnsafe(p);
  // lpEncodeBacklen 函数是计算当前元素的 element-tot-len 的字节大小，1-5B
  // 从而得到当前整个entry的总字节数
    entrylen += lpEncodeBacklen(NULL,entrylen);
    p += entrylen;
    return p;
}

/* If 'p' points to an element of the listpack, calling lpNext() will return
 * the pointer to the next element (the one on the right), or NULL if 'p'
 * already pointed to the last element of the listpack. */
unsigned char *lpNext(unsigned char *lp, unsigned char *p) {
    assert(p);
  // 找到下一个元素
    p = lpSkip(p);
  // 做一些edge case 的判断
    if (p[0] == LP_EOF) return NULL;
    lpAssertValidEntry(lp, lpBytes(lp), p);
    return p;
}
```



### 从右往左遍历

我们通过对上一个元素的 element-tot-len 的每个字节，从而可以得到 element-tot-len 占用的字节数 以及根据 element-tot-len 的内容可以得到 encoding-type 和 element-data 的字节数,从而完成从右往左遍历



```c
unsigned char *lpPrev(unsigned char *lp, unsigned char *p) {
    assert(p);
    if (p-lp == LP_HDR_SIZE) return NULL;
  // 指向前一个entry的最后一个字节
    p--; 
  
  // 从右往左遍历 element-tot-len ，计算出encoding-type 和 element-data 的总字节数
    uint64_t prevlen = lpDecodeBacklen(p);
  // lpEncodeBacklen 函数的作用是根据计算出的encoding-type 和 element-data 的总字节数得出 element-tot-len的字节数（1B-5B）
  // 从而得到了前一个entry的总字节数
    prevlen += lpEncodeBacklen(NULL,prevlen);
  // 将指针移动到上一个元素的开头（因为最初的时候-1，因此这里要少减一个字节）
    p -= prevlen-1;
    lpAssertValidEntry(lp, lpBytes(lp), p);
    return p;
}

static inline uint64_t lpDecodeBacklen(unsigned char *p) {
    uint64_t val = 0;
    uint64_t shift = 0;
    do {
        val |= (uint64_t)(p[0] & 127) << shift;
        if (!(p[0] & 128)) break;
        shift += 7;
        p--;
        if (shift > 28) return UINT64_MAX;
    } while(1);
    return val;
}


static inline unsigned long lpEncodeBacklen(unsigned char *buf, uint64_t l) {
    if (l <= 127) {
        if (buf) buf[0] = l;
        return 1;
    } else if (l < 16383) {
        if (buf) {
            buf[0] = l>>7;
            buf[1] = (l&127)|128;
        }
        return 2;
    } else if (l < 2097151) {
        if (buf) {
            buf[0] = l>>14;
            buf[1] = ((l>>7)&127)|128;
            buf[2] = (l&127)|128;
        }
        return 3;
    } else if (l < 268435455) {
        if (buf) {
            buf[0] = l>>21;
            buf[1] = ((l>>14)&127)|128;
            buf[2] = ((l>>7)&127)|128;
            buf[3] = (l&127)|128;
        }
        return 4;
    } else {
        if (buf) {
            buf[0] = l>>28;
            buf[1] = ((l>>21)&127)|128;
            buf[2] = ((l>>14)&127)|128;
            buf[3] = ((l>>7)&127)|128;
            buf[4] = (l&127)|128;
        }
        return 5;
    }
}

```

### 巨人的肩膀

[官方文档关于 listpack 的规范](https://github.com/antirez/listpack/blob/master/listpack.md#elements-representation)

[Redis 底层数据结构 listpack](https://juejin.cn/post/7240666488336007224?searchId=20240828192714B0DDE2F5B4398680DE98)

[深入分析 redis 之 listpack，取代 ziplist?](https://juejin.cn/post/7093530299866284045?searchId=20240828192714B0DDE2F5B4398680DE98#heading-17)
