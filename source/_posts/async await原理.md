---
title: async await原理
date: 2022-08-20 11:14:25
tags:
  - javaScript
---

## 介绍

通过generator 函数，使用`yield`关键字,配合next方法来控制函数的进行，如果你对于generator函数不熟悉，建议先看[generator](https://zh.javascript.info/generators)和[异步迭代和 generator](https://zh.javascript.info/async-iterators-generators)

next函数调用之后返回一个对象:`{value:xxx,done:xxx}`

- value：是yield后面跟的值，如果后面是函数，则value则是函数的返回值
- done：当done为tue时，则generator函数已经走完了，反之则没走完，可以继续调用next方法

<!-- more -->

```javascript
function *gen() {
 const num1=	yield 1
  console.log(num1);
 const num2=	yield 2
 const num3=	yield 3
	return num3
}
```

我们可以通过next方法控制函数的进行,同时给next( )方法传参数


如下

```javascript
const g =gen()
// 第一个next传不传参数都没影响
g.next(); //返回一个对象 {value：1，done：false}

// 如果下面这个next传了参数，则上面num1为121 ，否则为undefined
g.next(121); // 返回一个对象 {value:2,done:false} 
```

因此我们可以yield后面跟一个返回Promise的函数,然后调用next的时候把上一次resolve的值传进来，最后generator函数就可以返回最终Promise的值,代码如下


```javascript
function fn (num) {
  return new Promise ((resolve, reject) => {
    setTimeout(() => {
      resolve(2*num);
    },1000)
  })
}

function *gen() {
   const num1= yield fn(1)
   const num2 = yield fn(num1)
   console.log(num2);  //4
   const num3 = yield fn(num2)
   console.log(num3);  //8
  return num3
}

const  g =gen()
const next1 =g.next()
next1.value.then((res1) =>{
  console.log(next1) 
  console.log(res1);
  const next2 = g.next(res1)
  next2.value.then((res2) =>{
    const next3 = g.next(res2)
    console.log(next3)
    console.log(next3.value)
  })
})

```

当然，上面的代码是写死的，实际中我们并不知道`generator`函数中有多少`yield`

## 实现一下async await

```javascript

// 将生成器函数转成async函数
function generatorToAsync (generatorFn) {
  return  function () {
		// gen有可能传参
    const gen =generatorFn.apply(this,arguments)
		// async函数返回一个Promise
    return new Promise ((resolve, reject) => {
      function go(key,arg) {
        let res
        try {
			// 这里可能调用了throw方法，则会reject，arg则是调用next或者throw方法时传递的参数
          res= gen[key](arg)
        } catch (error) {
        return reject(error)
        }
        const {value,done} = res
		// 最后一个，直接返回最后的值即可
        if (done) {
          return resolve(value)
        }
		// value可能是常量或者promise，不是最后一个，则继续调用next方法        
				return Promise.resolve(value).then((res)=>go('next',res),(error) =>go('throw',error))
      }
		// 第一次首先调用一次next
      go('next')
    })
  }
}
```

最后实现的函数就是[co](https://github.com/tj/co)函数库的核心原理（自动执行generator的库）

我们一般使用async await 来处理Promise, await 后面跟的函数也是会返回一个Promise，但如果我们使用js时，await后面跟一个原始值,例如`await 4 ` 也会有'排队'的效果，可以看作它使用了Promise.resolve(4)包装了一层

参考文章:[7张图，20分钟就能搞定的async/await原理!](https://juejin.cn/post/7007031572238958629)



