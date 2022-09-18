---
title: Linux上安装pyspark
date: 2022-04-21 17:34:53
tags:
  - Linux
---

```java
//先将文件下载到Windows上，再使用ssh工具连接虚拟机，并将这三个文件上传到虚拟机，不要使用wmware-tool将文件直接拖进虚拟机，这样可能会有部分文件没有被下载，文件地址：
链接：https://pan.baidu.com/s/1-m4-SX0BuW20qmfusrfkHw 
提取码：left
```
<!-- more -->
![image-20220226101103434](https://img-blog.csdnimg.cn/img_convert/9ffb84007adf953ddfbc40b0da32ab86.png)

```java
mkdir /opt/jdk
mkdir /opt/apache-spark
tar xvzf jdk-8u251-linux-x64.tar.gz -C /opt/jdk
    
 vim ~/.bashrc 
 //添加下面两行
export JAVA_HOME=/opt/jdk/jdk1.8.0_251/jre
export PATH=$PATH:$JAVA_HOME/bin
    
source ~/.bashrc
java -version

```

```java
tar -zxvf spark-2.3.4-bin-hadoop2.7.tgz -C /opt/apache-spark/
 vim ~/.bashrc 
//修改成这三行
export JAVA_HOME=/opt/jdk/jdk1.8.0_251/jre
export SPARK_HOME=/opt/apache-spark/spark-2.3.4-bin-hadoop2.7
export PATH=$PATH:$JAVA_HOME/bin:$SPARK_HOME/bin
//使刚刚修改的配置文件生效
source ~/.bashrc
//测试有没有安装成功spark
spark-shell
```

![image-20220226091258772](https://img-blog.csdnimg.cn/img_convert/466d3e68afac300d55fd2954c2dabc4e.png)

![image-20220226091146303](https://img-blog.csdnimg.cn/img_convert/2bd7a433bafcd2298ab351fa09380aba.png)

```java
mkdir /opt/python3
tar -zxvf Python-3.6.5.tgz -C /opt/python3/
yum -y install zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel readline-devel tk-devel gcc
```

**安装完了会出现如图所示的complete**

![image-20220226092246667](https://img-blog.csdnimg.cn/img_convert/3222f60cc93461efac6569af515a8127.png)

```java
cd /opt/python3/Python-3.6.5
./configure --prefix=/usr/local/python3
//等待一小会
make && makeinstall
//等待几分钟，出现下图
```

![image-20220226092813454](https://img-blog.csdnimg.cn/img_convert/f109ce5e30b426b1fd5c831d0d070d51.png)

```java
//建立软链接
ln -s /usr/local/python3/bin/python3 /usr/bin/python3
ln -s /usr/local/python3/bin/pip3 /usr/bin/pip3
 //输入python3，出现下图所示
```

![](https://img-blog.csdnimg.cn/img_convert/3b4472b102fe5e60c789f3a78c8d7d34.png)

```java

vim ~/.bashrc  
//添加下面这一行，如下图所示
export PYSPARK_PYTHON=python3
//使配置生效
source ~/.bashrc
```

![image-20220226100003281](https://img-blog.csdnimg.cn/img_convert/4706d98ca46fcb30a6cfea9052764731.png)

```java
pyspark
//出现下图所示
```



![image-20220226095411859](https://img-blog.csdnimg.cn/img_convert/7977892c7f1608d2eca62002a56177e8.png)
**动动小手，点个赞！**



