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
![image-20220226101103434](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/202308181751333.png)

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

![image-20220226091258772](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/202308181751799.png)

![image-20220226091146303](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/202308181751321.png)

```java
mkdir /opt/python3
tar -zxvf Python-3.6.5.tgz -C /opt/python3/
yum -y install zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel readline-devel tk-devel gcc
```

**安装完了会出现如图所示的complete**

![image-20220226092246667](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/202308181751373.png)

```java
cd /opt/python3/Python-3.6.5
./configure --prefix=/usr/local/python3
//等待一小会
make && makeinstall
//等待几分钟，出现下图
```

![image-20220226092813454](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/202308181751428.png)

```java
//建立软链接
ln -s /usr/local/python3/bin/python3 /usr/bin/python3
ln -s /usr/local/python3/bin/pip3 /usr/bin/pip3
 //输入python3，出现下图所示
```

![](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/202308181751379.png)

```java

vim ~/.bashrc  
//添加下面这一行，如下图所示
export PYSPARK_PYTHON=python3
//使配置生效
source ~/.bashrc
```

![image-20220226100003281](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/202308181751127.png)

```java
pyspark
//出现下图所示
```



![image-20220226095411859](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/202308181751844.png)
**动动小手，点个赞！**



