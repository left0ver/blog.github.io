---
title: HTML5的Blob,FormData，URL, File接口介绍和使用
date: 2022-08-29 20:06:10
tags:
  - javaScript
  - html
---


# Blob

`Blob` 对象表示一个不可变、原始数据的类文件对象。它的数据可以按文本或二进制的格式进行读取，也可以转换成 [`ReadableStream`](https://developer.mozilla.org/zh-CN/docs/Web/API/ReadableStream) 来用于数据操作。

## Blob构造函数

```javascript
const  blob = new Blob( array, options );
```

*array* 是一个由[`ArrayBuffer`](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Global_Objects/ArrayBuffer), [`ArrayBufferView`](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Global_Objects/TypedArray), [`Blob`](https://developer.mozilla.org/zh-CN/docs/Web/API/Blob), [`DOMString`](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Global_Objects/String) 等对象构成的 `Array` ,或者其他类似对象的混合体，它将会被放进 `Blob`。DOMStrings 会被编码为 UTF-8

`options`是一个对象，`{type:xxx,endings:xxx}` ，

`type`是将会被放入到 blob 中的数组内容的 [MIME](https://developer.mozilla.org/zh-CN/docs/Web/HTTP/Basics_of_HTTP/MIME_types) 类型,

endings ：默认值为`"transparent"`，用于指定包含行结束符`\n`的字符串如何被写入。 它是以下两个值中的一个：`"native"`，代表行结束符会被更改为适合宿主操作系统文件系统的换行符，或者 `"transparent"`，代表会保持 blob 中保存的结束符不变 

<!-- more -->
## 方法

- `arrayBuffer()`,返回一个 promise 对象，在 resolved 状态中以二进制的形式包含 blob 中的数据，res是对应Blob的arrayBuffer的形式的数据
- `slice（）`切割blob对象返回一个新的Blob对象，类似数组的slice方法
- `stream()`方法返回一个[`ReadableStream`](https://developer.mozilla.org/zh-CN/docs/Web/API/ReadableStream)对象，读取它将返回包含在`Blob`中的数据
- **`text()`** 方法返回一个 `Promise` 对象，包含 blob 中的内容，使用 UTF-8 格式编码

# FormData

**`FormData`** 接口提供了一种表示表单数据的键值对 `key/value` 的构造方式，如果送出时的编码类型被设为 `"multipart/form-data"`，它会使用和表单一样的格式。这个接口的api都比较简单，一般`append`方法用的多，如果送出时的编码类型被设为 `"multipart/form-data"`，它会使用和表单一样的格式

# URL

## 构造函数

- 我们可以通过构造函数的方式创建一个URL对象,`new URL（url,baseurl）`，`baseurl`是可选的,填写了`baseurl`，则`url`则使用相对的`url`，如果 `url` 是绝对 URL，则无论参数`base`是否存在，都将被忽略

  ```javascript
  const urlObj= new URL("/blog","https://leftover.cn")
    console.log(urlObj)//打印结果如下：
  ```

  ![image-20220829113124465](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220829113124465.png)

## 方法

- 方法：`createObjectURL()`,`revokeObjectURL()`,`toJSON()`，`toString()`

  1. `createObjectURL(object)`（静态方法）

     object: 用于创建 URL 的 `File` 对象、`Blob`对象或者 [`MediaSource`](https://developer.mozilla.org/zh-CN/docs/Web/API/MediaSource) 对象。

     返回值：返回一个URL对象

  2. `URL.revokeObjectURL(objectURL)`（静态方法）

     - **`URL.revokeObjectURL()` **静态方法用来释放一个之前已经存在的、通过调用 [`URL.createObjectURL()`](https://developer.mozilla.org/zh-CN/docs/Web/API/URL/createObjectURL) 创建的 URL 对象。当你结束使用某个 URL 对象之后，应该通过调用这个方法来让浏览器知道不用在内存中继续保留对这个文件的引用了。防止内存泄漏

     - `objectURL`:之前通过`createObjectURL(object)`方法创建的URL对象

       返回值 undefined
  
  3. ``toJSON()`，`toString()`（原型上的方法）
  
     `toJSON()`方法将一个URL对象以字符串的形式返回，在实际使用中和`toString()`基本没区别
  
     ```javascript
     const url = new URL("https://leftover.cn");
     url.toJSON(); // 以字符串形式返回 URL
     ```

# File

## 构造函数

```javascript
var myFile = new File(data, name[, options]);
```

- data表示文件的内容，可以是`ArrayBuffer`、`ArrayBufferView`、`Blob`、字符串数组，或者这些对象的组合

- name：文件名称或者文件路径

- options（可选）:一个对象,有type和lastModified属性

  type:表示将要放到文件中的内容的 [MIME](https://developer.mozilla.org/zh-CN/docs/Web/HTTP/Basics_of_HTTP/MIME_types) 类型。默认值为 `""`

  lastModified：数字，表示文件最后修改时间的Unix时间戳（毫秒）。默认值为`Date.now()`

  ![image-20220829192445618](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/image-20220829192445618.png)

## 方法

- `File`接口继承自Blob接口，他没有自定义方法，支持Blob接口的所有方法
- 一个文件对象类似这样，有自己的一些专属的属性

