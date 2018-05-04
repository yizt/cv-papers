Fast R-CNN
=====



-   [依赖知识](#FastR-CNN-依赖知识)

-   [知识点](#FastR-CNN-知识点)

    -   [ ROI](#FastR-CNN-ROI)

    -   [end-to-end](#FastR-CNN-end-to-end)

    -   [Spatial Pyramid Pooling](#FastR-CNN-SpatialPyramidPooling)

-   [网络结构和训练](#FastR-CNN-网络结构和训练)

    -   [ 网络结构](#FastR-CNN-网络结构)

    -   [RoI池化层](#FastR-CNN-RoI池化层)

    -   [初始化预训练网络](#FastR-CNN-初始化预训练网络)

    -   [训练](#FastR-CNN-训练)

        -   [ 分级采样](#FastR-CNN-分级采样)

        -   [ 联合训练](#FastR-CNN-联合训练)

        -   [超参](#FastR-CNN-超参)

-   [关键点](#FastR-CNN-关键点)

-   [疑问点](#FastR-CNN-疑问点)

       Fast
R-CNN论文地址：<https://arxiv.org/pdf/1504.08083.pdf>。可以参考的文章：[Fast
R-CNN学习总结](https://zhuanlan.zhihu.com/p/30368989)。有了R-CNN的基础Faster
R-CNN相对容易理解。

**R-CNN有三个缺点**

-   训练分多个阶段

     首先精调CNN,然后训练SVM检测对象，最后训练边框回归

-   训练时空花费高


 空间上：SVM和边框回归使用的CNN特征保留在磁盘上，需要数百G空间。时间上：对于VGG16模型和VOC2007数据集(5千张图片),一块GPU需要2.5天才能完成训练。

-   检测耗时长

     预测时在GPU上检测1张图片需要47秒；无法实时预测。

**Faster R-CNN优势**

-   检测速度更快

-   单步训练


使用multi-task损失函数，一步训练SVM分类和边框回归，不再分步训练；同时训练可以更新整个网络，基本实现了end-to-end训练。

-   不再需要存储特征

-   训练更快

   不再是将region
proposals依次通过CNN，而是直接输入原图，来提取特征（这样一张图只会CNN一次）

 



 

依赖知识
--------

    a） 已经熟悉R-CNN

    b )  了解预训练、迁移学习

知识点
------

###  ROI

 ROI（region of
interest），感兴趣区域。机器视觉、图像处理中，从被处理的图像以方框、圆、椭圆、不规则多边形等方式勾勒出需要处理的区域，称为感兴趣区域，ROI。

### end-to-end

  端到端指的是输入是原始数据，输出是最后的结果。细分有端到端训练和端到端模型。

参考：<https://www.zhihu.com/question/51435499/answer/129379006>

### Spatial Pyramid Pooling

     Spatial Pyramid Pooling
空间金字塔池化。传统CNN中的全连接层需要输入是固定大小。在图像处理中，原始图片大小不一；经过卷积后大小还是不同，这样没有办法直接接入全连接层。

![特征金字塔池化](pic/FastR-CNN特征金字塔池化.jpg)




这是一个传统的网络架构模型，5层卷积层，这里的卷积层叫做convolution和pooling层的联合体，统一叫做卷积层，后面跟随全连接层。我们这里需要处理的就是在网络的全连接层前面加一层金字塔pooling层解决输入图片大小不一的情况。我们可以看到这里的spatital
pyramid pooling layer就是把前一卷积层的feature
maps的每一个图片上进行了3个卷积操作。最右边的就是原图像，中间的是把图像分成大小是4的特征图，最右边的就是把图像分成大小是16的特征图。那么每一个feature
map就会变成16+4+1=21个feature maps。这不就解决了特征图大小不一的状况了吗？

详见：[空间金字塔池化阅读笔记](http://blog.csdn.net/liyaohhh/article/details/50614380)

论文：[Spatial Pyramid Pooling in Deep Convolutional Networks.pdf](https://arxiv.org/pdf/1406.4729.pdf)

网络结构和训练
--------------

###  网络结构

![Fast R-CNN 网络结构](pic/Fast R-CNN 网络结构.jpg)

           Fast R-CNN网络的输入时一整张图片和Selective
Searchcong方法从图片中的获取的Proposals。首先通过若干卷积和池化层产生一个卷积特征图。然后对于每一个Proposal,RoI池化层从特征图中抽取定长的特征向量。每个特征向量送入一系列的全连接层；最后分支为两个兄弟输出层：一个使用softmax预测K+1个类别，一个对K类别做bounding-box
回归。



### RoI池化层

     RoI时一种特殊的Max Pooling; 每个RoI有四元组定义(r, c, h, w) ，(r,
c)代表左上角， (h,
w)代表高度和宽度。RoI最大池化把h\*w的RoI窗口分成固定的H\*W的网格，每个网格的大小约
h/H \*
w/W；然后在每个网格上去最大值。跟标准的最大池化意义，RoI池化在每个通道独立进行。

### 初始化预训练网络

     将一个CNN转为Fast R-CNN有三个转换步骤

a)
最后一个最大池化层使用RoI最大池化替换。固定H，W来适应后面的全连接层(如：VGG16,H
= W =7)。

b) 最后一个全连接层和softmax层，替换为一个fc层和两个兄弟层；softmax和
类别专用bounding-box回归。

c) 将网络的输入数据改为两部分：图片和图片的RoI。

### 训练

####  分级采样

     首先采样N张图片,然后从每张图片采样R/N个RoIs(N =
2,R=128效果较好)；25%的正样本，IoU大于0.5；75%的负样本IoU在[0.1,0.5)之间；受hard
example
mining启发，丢弃了IoU值小于0.1的样本。训练中图像水平。训练是图片以50%概率做水平翻转。

####  联合训练

     多任务联合训练，softmax分类和bounding
box回归在一个网络中同时训练，优化的损失函数时两部分损失函数线性和。

#### 超参

     a) softmax分类和bounding
box回归的权重都使用零均值高斯分布初始化，标准差0.01\~0.001；偏置都初始化为0;

     b) 每层的学习率权重为1，偏置为2；全局学习率设置为0.001.

关键点
------

     Proposal RoI投影计算方法 <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default">

$$x^‘ = \lfloor \frac x S\rfloor  $$  ; 其中$$x^‘$$</script>是ROI在特征图Feature Map中的横坐标，x时RoI在原图中的横坐标;S是所有卷积层和池化层步长strides的乘积，纵坐标也是同样的计算方法。ROI在Feature

Map中对应的区域后，就做RoI 最大池化转为固定长度的特征向量。

疑问点
------

1：在R-CNN训练SVM分类时使用的正样本仅仅时Ground-Truth，为什么这里softmax分类正样本使用IoU大于0.5？

 

 



## 关于我们

我司正招聘文本挖掘、计算机视觉等相关人员，欢迎加入我们；也欢迎与我们在线沟通任何关于数据挖掘理论和应用的问题；

在长沙的朋友也可以线下交流, 坐标: 长沙市高新区麓谷新长海中心 B1栋8A楼09室

公司网址：http://www.embracesource.com/

Email: mick.yi@embracesource.com 或 csuyzt@163.com

