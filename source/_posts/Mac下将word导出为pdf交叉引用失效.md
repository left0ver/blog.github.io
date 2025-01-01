---
title: Mac下将word导出为pdf交叉引用失效
date: 2025-01-01 16:26:25
tags:
  - other
---

# Mac下将word导出为pdf交叉引用失效

1. mac下的word没有没有导出为pdf的的选项，只能`command + p` 进入到打印的页面，将其导出为pdf，但是导出的pdf交叉引用的跳转会失效。

   解决方法：

   在word的加载项，找到`Soda PDF Converter` 插件，使用它将docx导出为pdf，这种方法导出的pdf中，会保留交叉引用的跳转功能。

2. 由于我使用的是endnote21来对参考文献进行管理，使用endnote21导入参考文献，但是导入的参考文献并没有交叉引用，导出为pdf之后也不能直接跳转。

   解决方法：

   <img src="https://img.leftover.cn/img-md/202501011619603.png" alt="image-20250101161946555" style="zoom: 25%;" />

   将`Link in text ...` 勾选，然后更新参考文献，在word中点击文献编号即可调整到对应的位置，最后使用`Soda PDF Converter`将docx导出为pdf，导出的pdf中参考文献和交叉引用的图表均能正确跳转。

   ![image-20250101162021711](https://img.leftover.cn/img-md/202501011620751.png)
