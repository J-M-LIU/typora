### 1. 在Idea中创建maven项目

(1)、点击File，选择New，再点击Project

(2)、具体看图，别选错webapp了

<img src="https://img-blog.csdnimg.cn/20210316160813541.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3p6dmFy,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom: 50%;" />

(3)、接下来输入各类名称，填完后直接next下一步，名称具体含义如下图：

<img src="https://img-blog.csdnimg.cn/20210316163047352.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3p6dmFy,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom:50%;" />

(4)、配置maven信息，具体操作如下图

<img src="https://img-blog.csdnimg.cn/20210316165927988.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3p6dmFy,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom:50%;" />

(5)、添加配置 DarchetypeCatalog=internal，添加原因：每次创建maven项目时， IDEA 要使用插件进行创建，这些插件当你创建新的项目时，它每次都会去中央仓库下载，这样使得创建比较慢。所以在创建maven项目时，应该让它找本地仓库中的插件进行项目的创建。

**DarchetypeCatalog=internal**

<img src="https://img-blog.csdnimg.cn/20210316170658537.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3p6dmFy,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom:50%;" />

### 2. 完善maven-web项目模板

(1)、第一次加载maven项目比较慢，等待右下角的进度条加载结束，下面是刚建好的maven-web项目模板

<img src="https://img-blog.csdnimg.cn/20210316171955906.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3p6dmFy,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom: 67%;" />

(2)、上面的项目不完整，不能满足我们的开发需要，所以需要我们手动建设一些文件夹，需要新建的文件夹我用红色标出，以下是maven项目的标准目录结构：
src/main/**java**
src/main/**resources**
src/test/**java**
src/test/**resources**

(3)、接下来就是把新建立的文件夹进行关联了，看清楚红色的关联对象，关联错了就得重新关联，这一定不能出错，关联方法如下图。
src/main/java 关联为 Sources Root
src/main/resources 关联为Resources Root
src/test/java 关联为Test Sources Root
src/test/resources 关联为 Test Resources Root

(4)、也可以右键项目，然后选择Open Module Settings打开项目配置页面更改，关联方法如下图

<img src="https://img-blog.csdnimg.cn/2021031618354646.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3p6dmFy,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom: 67%;" />

### 3.Idea中部署Tomcat并进行

Idea中有自带的Tomcat，看个人需要，想配置Tomcat的就配置，不想配置Tomcat就使用开发工具自带的Tomcat

**首先是下载Tomcat 8/9/10 之前电脑上已经安装过，需要删除服务，管理员身份运行cmd 输入 sc delete tomcat9即可**

**键入localhost:8080(之前设置的端口)，如果正确显示安装的Tomcat的信息，说明安装成功**

(1)、直接进入Idea，点击Run——Edit Configurations…
<img src="https://img-blog.csdnimg.cn/20210316201513171.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3p6dmFy,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom:67%;" />

(2)、点击左侧“+”号，找到Tomcat Server——Local（若是没有找到Tomcat Server 可以点击最后一行 34 items more）
<img src="https://img-blog.csdnimg.cn/20210316202224408.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3p6dmFy,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom:67%;" />

(3)、点击点击configure… ,进入下面第二张图，按图二添加配置你的Tomcat路径，配置完点击apply
<img src="https://img-blog.csdnimg.cn/20210316213431184.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3p6dmFy,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom: 50%;" />

<img src="https://img-blog.csdnimg.cn/20210316213730530.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3p6dmFy,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom: 50%;" />

(4)、点击Deployment—> + —> Artifact… ,进入下面第二张图，按图二添加项目
<img src="https://img-blog.csdnimg.cn/20210316210246618.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3p6dmFy,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom:50%;" />



<img src="https://img-blog.csdnimg.cn/202103162104535.png" alt="在这里插入图片描述" style="zoom:50%;" />

(5) 运行