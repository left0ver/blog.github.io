---
title: 解决刷新页面Vuex数据丢失问题
date: 2022-07-01 14:04:28
tags:
 - vue
---
我们使用 Vue 和 Vuex 的时候，当我们刷新页面的时候， Vuex 里的数据就会恢复为初始状态，要想解决这个问题，实现 Vuex 数据的持久化

1. 自己实现
   
-  我们可以在刷新页面之前将数据存储到 `sessionStorage` 、 `localStorage`、`cookie` 里面，然后我们进入页面之前从 `sessionStorage` 、 `localStorage` 、 `cookie` 里面读取数据保存到 Vuex 里即可，具体的代码如下：
-  ,如果用户退出浏览器，则 `sessionStorage` 里面的数据就消失了，而 `localStorage` 里的数据除非你自己手动清除，否则一直存在，而 cookie 一般是有时效性的，而且    cookie 里面可以存储的数据大小有限，最多只能储存 `4KB` 的数据

- 个人建议储存在 `sessionStorage` 里面会更好
  
<!-- more -->
 ```javascript
//  APP.vue

export default {
  name: 'App',
  created() {
    const oldStore = sessionStorage.getItem('store')
    // 第一次进入页面时为null，后面如果刷新页面将会有值，则会替换Vuex里面的数据
    if (oldStore !== null) {
      this.$store.replaceState(
        Object.assign({}, this.$store.state, JSON.parse(oldStore))
      )
    }
    // 监听页面刷新，将store里面的数据保存到sessionStorage
    window.addEventListener('beforeunload', () => {
      sessionStorage.setItem('store', JSON.stringify(this.$store.state))
    })
  },
}
```
2. 使用插件
   
  你可以使用 [vuex-persistedstate](https://github.com/robinvdvleuten/vuex-persistedstate) 这个插件，本质的原理也是使用了本地的储存，也可以分别存储在 cookie , sessionStorage , localStorage 里面

