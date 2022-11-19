---
title: 关于我是如何使用Jenkins进行CI CD
date: 2022-11-19 20:58:04
tags:
  - 部署
  - Linux
---

# overView

这篇文章旨在讲述我是如何使用jenkins做CI，CD的

# 使用docker启动jenkins容器

```shell
docker run --name jenkins-docker --rm --detach \
  --privileged --network jenkins --network-alias docker \
  --env DOCKER_TLS_CERTDIR=/certs \
  --volume jenkins-docker-certs:/certs/client \
  --volume jenkins-data:/var/jenkins_home \
  --publish 2376:2376 \
  docker:dind --storage-driver overlay2
```

```shell
docker run --name jenkins-blueocean --restart=on-failure --detach --privileged \
  --network jenkins --env DOCKER_HOST=tcp://docker:2376 \
  --env DOCKER_CERT_PATH=/certs/client --env DOCKER_TLS_VERIFY=1 \
  --publish 8080:8080 --publish 50000:50000 \
  --volume $(which docker):/usr/bin/docker \
  --volume jenkins-data:/var/jenkins_home \
  --volume jenkins-docker-certs:/certs/client:ro \
```

--privileged 和  --volume $(which docker):/usr/bin/docker是为了能够在jenkins中使用容器来进行构建 

<!-- more -->

# 插件

对于jenkins插件，我首先安装了推荐的插件，其次安装了[Blue Ocean](https://plugins.jenkins.io/blueocean)，[Email Extension Plugin](https://plugins.jenkins.io/email-ext)[Email Extension Template Plugin](https://plugins.jenkins.io/emailext-template)[Publish Over SSH](https://plugins.jenkins.io/publish-over-ssh)

1. Blue Ocean可以帮助你更好地配置流水线，几乎必装

2. Email Extension Plugin 和 Email Extension Template 是针对邮件发送的，Email Extension Template可以定义邮件的模版，Email Extension Plugin用来发送邮件（相比jenkins提供的原生的邮件发送，他具有更加强大的功能）

3. Publish Over SSH是针对文件传送的，利用ssh连接，可以将打包后的文件传送到自己的服务器上

# 配置邮箱

![image-20221113232614521](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/202211132326569.png)

这里需要用到邮箱的凭证，手动添加一个凭证，username就是你的邮箱地址，password是授权码

![image-20221113232740554](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/202211132327580.png)

Default Subject（ 默认的标题 ）：$PROJECT_NAME - Build # $BUILD_NUMBER - $BUILD_STATUS!

Maximum Attachment Size（设置允许的附件大小）：-1（不限制）

Default Content（默认的邮件内容）：

```html
$PROJECT_NAME - Build # $BUILD_NUMBER - $BUILD_STATUS:

Check console output at $BUILD_URL to view the results.


<hr/>(自动化构建邮件，无需回复！)<br/><hr/>
项目名称：$PROJECT_NAME<br/><br/>

项目描述：$JOB_DESCRIPTION<br/><br/>

运行编号：$BUILD_NUMBER<br/><br/>

运行结果：$BUILD_STATUS<br/><br/>

触发原因：${CAUSE}<br/><br/>

构建日志地址：<a href="${BUILD_URL}console">${BUILD_URL}console</a><br/><br/>

构建地址：<a href="$BUILD_URL">$BUILD_URL</a><br/><br/>

详情：${JELLY_SCRIPT,template="html"}<br/>
<hr/>
```

![image-20221113233035965](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/202211132330994.png)

Debug模式：可以在构建的日志中看到详细的邮件的发送情况以及报错信息

Enable watching for jobs: 这个一定得开，可以让你在job中使用这些变量属性，因为这个东西是不能在job中调用的，在freestyle中才可以用，我们在job中可以使用下面这些环境变量来设置邮件的标题，内容以及接收者

$DEFAULT_SUBJECT：默认标题

$DEFAULT_CONTENT：默认的邮件内容

$DEFAULT_RECIPIENTS：默认的接收者



# 配置Publish over SSH

Publish over SSH用来同步文件到自己的服务器上，当然也可以不用插件，使用rsync来进行同步，这里就是简单配置一下ip，username，以及自己的私钥的地址，或者可以直接粘贴私钥内容，也可以使用密码登陆，能用密钥登录尽量还是密钥登录

需要注意的是**Remote Directory**的配置，这个是设置你的上传的对应服务器的根目录，你以后写的路径都是基于这个路径来的，这里没什么特殊原因填写/即可

# 创建流水线

1. 一般来说可以使用freeStyle，或者流水线，也可以创建多分支流水线，或者使用blueOcean来创建，这里我一般选择第二个，freeStyle相对简单，但其局限性比较大，不能使用docker等，而pipeline相对灵活，blueOcean需要你将Jenkinsfile文件放在项目里面，这里我不太喜欢将Jenkinsfile放在项目中，因此使用单个流水线比较多

2. ![image-20221114000436410](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/202211140004439.png)

3. 这里我一般会勾选`github项目`，`参数化构建过程`（可以设置构建所需的参数，之后再下面的脚本中可以使用这些参数一般配合构建触发器中的触发远程构建来使用）,`丢弃旧的构建`（一般保留7天内的构建结果），构建触发器我一般选择`触发远程构建`,这个相对灵活，一般使用GitHub action 来运行curl命令触发构建

# 定义流水线脚本

这里agent我一般使用docker环境

```shell
pipeline {
    agent {
        docker {
        image 'node:16.18-alpine3.15'
  }
}
    parameters {
        string(name: 'PERSON', defaultValue: 'Mr Jenkins', description: 'Who should I say hello to?')
    }
    
    stages {
        stage('Hello') {
            steps {
              echo "${params.PERSON}"
              # 拉git代码
              git branch: 'master', credentialsId: 'leftover-github', poll: false, url: 'git@github.com:left0ver/building-a-multibranch-pipeline-project.git'
              # 下载依赖
              sh "node --version && yarn --version && npm install --registry https://registry.npmmirror.com/ "
              # 打包
              sh "npm run build"
              # 将产物上传到服务器
              sshPublisher(publishers: [sshPublisherDesc(configName: 'my-server', transfers: [sshTransfer(cleanRemote: false, excludes: '', execCommand: '', execTimeout: 120000, flatten: false, makeEmptyDirs: false, noDefaultExcludes: false, patternSeparator: '[, ]+', remoteDirectory: '/home/zwc', remoteDirectorySDF: false, removePrefix: '', sourceFiles: 'build/**/*')], usePromotionTimestamp: false, useWorkspaceInPromotion: false, verbose: false)])
              
            }
        }
    }
    post {
        always {
        			# 清除工作区
              cleanWs()
              # 邮件通知
              emailext body: '$DEFAULT_CONTENT', subject: '$DEFAULT_SUBJECT', to: '$DEFAULT_RECIPIENTS'
        }
    }
}

```


脚本的大概步骤就是先利用git插件拉取代码，下载依赖，打包，之后将打包之后的文件上传到服务器的指定文件内，清除工作区，最后发送邮件



# 参数化构建

- 我们可以使用手动触发配合参数化构建，手动触发的时候URL里面传递参数,勾选`参数化构建过程`来设置传递的参数以及默认值等等，然后在pipeline 里面使用${params.xxx}来取值

![image-20221118142105843](https://leftover-md.oss-cn-guangzhou.aliyuncs.com/img-md/202211181421962.png)

# 配合GitHub

我们可以使用参数化构建配合GitHub actions ,使用curl命令来触发构建，将curl命令的内容配置在Secrets里面，可以监听push或者pull request等事件触发actions，从而触发对应的构建
