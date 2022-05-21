---
title: react-router v6详解
date: 2022-05-21 17:54:33
tags:
  - react
---


## router组件简单介绍

- 使用`Navigate`组件来进行重定向

- v6中的`Routes`相当于v5的`switch`

- `route`里面也可以嵌套子路由，可以使用`Outlet`组件来实现类似`router-view` 的效果

<!-- more -->

- `Route`组件里面的index属性,可以在没有匹配到任何子元素的时候显示某个组件,可以用来做默认显示的值，使用index属性时不可以有path属性,如下示例

  ```tsx
         <Route path='/profile' element={<Profile />}>
               <Route  index  element={<ProfileSetting/>}></Route>
               <Route  path="setting" element={<ProfileSetting/>}></Route>
               <Route  path="detail"  element={<ProfileDetail/>}></Route>
         </Route>
  ```

## react-router-hooks

1. `useHref`，给一个to,返回一个相对于当前路由的链接，例如

   ```tsx
    const href = useHref("setting")
     console.log(href); //当前路由是/profile/detail,返回/profile/detail/setting
   //如果给的是绝对链接，则返回这个绝对链接
   ```

2. `useInRouterContext`，返回当前组件是否在react-router之中的一个布尔值，对于一些想知道当前组件有没有使用react-router的第三方扩展非常有用

3. `useLocation `返回一个当前的路由对象，对象的格式如下：

   ```tsx
   {
       hash: ""
       key: "default"
       pathname: "/cart/1"
   	search: ""
   	state: null
   }
   ```

   

4. ` useNavigate` 返回一个`NavigateFunction`，可以利用返回的函数进行编程式导航，相当于`Navigate`组件

   ```tsx
    const navigate = useNavigate()
    navigate('/home',{replace:true,state:{name:'zwc666'}})
   //在home页面的组件中可以使用useLocation获取到传过去的state
   //也可以navigate(-1)
   ```

5. `useNavigationType` 返回NavigationType,可以用来判断是怎么来到该页面的，用的不多

   ```tsx
   type NavigationType = "POP" | "PUSH" | "REPLACE";
   ```

6. `useOutlet`返回一个`React.ReactElement`，作用和`Outlet`组件差不多

7. `useOutletContext` ，当父路由使用了`Outlet`组件时，可以使用context属性向子路由传递一些数据

   `useOutletContext`则可以接收父路由传递过来的数据

   ```tsx
   import {Outlet,,useOutlet,useOutletContext} from 'react-router-dom'
   //导出类型
   export type contextType=[number,React.Dispatch<React.SetStateAction<number>>]
   export default function Parent() {
       const [count,setCount] =useState<number>(55)
       const Element = useOutlet([count,setCount])
    	 return (
       	<div>
               Profile
               {/* <Outlet /> */}
               {Element}
           </div>
   }
   //在父路由中定义一个hooks,提前给useOutletContext设置好类型，这样子路由就可以不需要再设置类型了，直接调用即可
   export function useCount () {
     return useOutletContext<contextType>()
   }
   
   //1.在ts中，要先在父组件中定义好context的类型，然后子组件使用useOutletContext的时候设置类型
   import type {contextType} from '../Parent/index'
   //第二种,直接导入定义好类型的hooks,然后直接使用即可
   import { useCount } from '../Parent/index'
   export default function Son() {
       //设置类型
      const [count,setCount] = useOutletContext<contextType>()
      //第二种
      //const [count,setCount]= useCount()
     return (
       <>
       <div>{count}</div>
       <button onClick={()=>setCount(count+1)}>点我加1</button>
       </>
     )
   }
   
   ```

8. `useParams `用来获取当前路由的params

9. `useResolvedPath `用来解析给定的路径，可以传入相对url，和绝对的url，返回一个对象

   ```tsx
   {
       pathname: '/cart/12/hhhxixi',
       search: '', 
       hash: ''
   }
   ```

10. `useRoutes `，传入一个路由的数组，返回对应要展示的组件，有点类似于vue的路由的配置，可以在别的文件夹配置好路由，然后传入APP.tsx中

11. `useSearchParams`返回一个数据,第一个是searchParams，第二个是setSearchParams,用来更新searchParams，

    searchParams是一个map集合,如果要获取某个参数,可以使用map集合的get方法

    ```tsx
      let [searchParams, setSearchParams] = useSearchParams();
        console.log(searchParams.get('age'));
        for (const [key,value] of searchParams) {
            console.log(key,value);
        }
    ```

12. `useLinkClickHandler`这个hooks返回一个点击事件的处理函数，可以用来创建一个自定义跳转link,点击之后即可跳转到目标链接，可以传递一个to（目标链接），和一个options

    ```tsx
    //参数
    {
      to: To,
      options?: {
        target?: React.HTMLAttributeAnchorTarget;
        replace?: boolean;
        state?: any;
            } 
    }
    //返回一个函数，函数调用时需要传递对应的event，使用方法如下
    
    const clickHandler =useLinkClickHandler<HTMLButtonElement>("/profile/setting")
    
    return (
        //点击了按钮就会跳转到/profile/setting
    <button onClick={(event)=>clickHandler(event)}>点我跳转到/profile/setting"</button>
    )
    ```

    

13. 注意：**路由懒加载时要配合Suspense组件一起使用，如果你使用useRoutes这个hooks返回对应的要渲染的路由元素时，不能使用路由懒加载，使用了懒加载的页面会出现空白或者报错**，期待后面可以修复。

