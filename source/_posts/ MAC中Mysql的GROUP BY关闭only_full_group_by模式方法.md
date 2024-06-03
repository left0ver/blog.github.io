---
title: MAC中Mysql的GROUP BY关闭only_full_group_by模式方法
date: 2024-6-3 10:32:09
tags:
  - mysql
---
## MAC中Mysql的GROUP BY关闭only_full_group_by模式方法

1. 找到mysql的配置文件，若没有则创建一个，

   ```shell
   sudo vim /etc/my.cnf
   ```

   添加下面的内容

   ```shell
   [mysqld]
   sql_mode='STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION'
   ```

   之后再配置一下配置文件的路径，再重启mysql即可

   <img src="https://img.leftover.cn/img-md/202406032222697.png" alt="image-20240603222253609" style="zoom:33%;" />

<!-- more -->
2. sql_mode常用的值

   - ONLY_FULL_GROUP_BY： 对于 GROUP BY 聚合操作，如果在 SELECT 中的列，没有在 GROUP BY 中出现，那么这个 SQL 是不合法的，因为列不在 GROUP BY 从句中

   - NO_AUTO_VALUE_ON_ZERO： 该值影响自增长列的插入。默认设置下，插入 0 或 NULL 代表生成下一个自增长值。如果用户 希望插入的值为 0，而该列又是自增长的，那么这个选项就有用了。

   - STRICT_TRANS_TABLES： 在该模式下，如果一个值不能插入到一个事务表中，则中断当前的操作，对非事务表不做限制 NO_ZERO_IN_DATE： 在严格模式下，不允许日期和月份为零

   - NO_ZERO_DATE： 设置该值，mysql 数据库不允许插入零日期，插入零日期会抛出错误而不是警告。

   - ERROR_FOR_DIVISION_BY_ZERO： 在 INSERT 或 UPDATE 过程中，如果数据被零除，则产生错误而非警告。如 果未给出该模式，那么数据被零除时 MySQL 返回 NULL

   - NO_AUTO_CREATE_USER： 禁止 GRANT 创建密码为空的用户

   - NO_ENGINE_SUBSTITUTION： 如果需要的存储引擎被禁用或未编译，那么抛出错误。不设置此值时，用默认的存储引擎替代，并抛出一个异常

   - PIPES_AS_CONCAT： 将"||"视为字符串的连接操作符而非或运算符，这和 Oracle 数据库是一样的，也和字符串的拼接函数 Concat 相类似

   - ANSI_QUOTES： 启用 ANSI_QUOTES 后，不能用双引号来引用字符串，因为它被解释为识别符

mysql8.0中，sql_mode的值默认为`ONLY_FULL_GROUP_BY, STRICT_TRANS_TABLES, NO_ZERO_IN_DATE, NO_ZERO_DATE, ERROR_FOR_DIVISION_BY_ZERO, and NO_ENGINE_SUBSTITUTION`

因此上述的修改是去除了`ONLY_FULL_GROUP_BY`这个值，保留了mysql默认的sql_mode的其余的值



## 参考文献

1. [MySQL解决macOS下mysql的sql_mode=only_full_group_by问题](https://cclc.github.io/mess-knowledge/mac-mysql-sql-mode-only-full-group-by.html)
2. [MAC中Mysql的GROUP BY关闭only_full_group_by模式方法 Variable ‘sql_mode‘ can‘t be set](https://blog.csdn.net/myhAini/article/details/110628323)
3. [MySQL8 sql_mode简单介绍](https://www.modb.pro/db/107706)

