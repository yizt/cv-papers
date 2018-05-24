R-CNN: Rich feature hierarchies for accurate object detection and semantic segmentation Tech report (v5)
=====

[TOC]

  

​        在标准的PASCAL VOC度量的**目标检测**性能在最近几年进入**停滞**状态。性能最好的方法是复杂的集成系统，通常组合**多个低级图像特征**和**高级环境信息**。本文提出一个简单的、可伸缩的检测算法，相对之前在VOC 2012数据集上最好的性能，mAP值**提升**了**超过30%**，达到**53.3%**的mAP值。我们的方法包含关键的两点洞悉: (1)可以将高容量的CNN应用到自底向上的region proposals中来定位和分隔对象；(2) 当标注训练数据缺乏时，可以**监督预训练**辅助任务，然后再做**domain-specific精调**，可以获得显著的性能提升。因为组合了region proposals和CNNs,我们的方法称作**R-CNN**：带CNN特性的Regions。我们同样将R-CNN与OverFeat比较，OverFeat是最近提出的基于类型CNN架构的滑动窗口检测器。在200类的ILSVRC2013检测数据集上，**R-CNN**性能**大幅度超过OverFeat**。整个系统的源码可在http://www.cs.berkeley.edu/~rbg/rcnn获得。

​       


        R-CNN论文地址：<https://arxiv.org/pdf/1311.2524.pdf> 

 

依赖知识
--------

    a） 熟悉CNN，以及常用的CNN分类网络AlexNet,VGG,GoogLeNet等

    b )  了解预训练、迁移学习

    c） 了解目标检测的基本概念，知道IoU、Ground Truth

知识点
------

### selective search

        一种从图片中提取bounding box(边界框)的方法. 主要思想是

a) 使用Efficient Graph-Based Image Segmentation获取原始图片分割区域

b) 根据各分割区域的颜色、小大、纹理、吻合等方面相似度合并分割区域。

详见：[selective
search算法](http://blog.csdn.net/lianhuijuan/article/details/64443008?locationNum=1&fps=1)，论文地址：<http://www.huppelen.nl/publications/selectiveSearchDraft.pdf>

### NMS


 non maximum suppression: 中文名非极大值抑制. 非极大值抑制算法（Non-maximum
suppression, NMS）的本质是搜索局部极大值，抑制非极大值元素。

详见：<http://blog.csdn.net/u014365862/article/details/52653834> 或<http://blog.csdn.net/shuzfan/article/details/52711706>

### hard negative mining method


首先是negative，即负样本，其次是hard，说明是困难样本，也就是说在对负样本分类时候，loss比较大（label与prediction相差较大）的那些样本，也可以说是容易将负样本看成正样本的那些样本。

        比如:
要在一张厨房的图像里判别一只蜘蛛。由于厨房的图像里objects（负样本）可能有许多，蜘蛛（正样本）样本却很少。这时候hard
mining的意义就是选出一些对训练网络有帮助的负样本部分，而不是所有负样本，来训练网络。比如说，training过程中，被当前模型误判成蜘蛛最厉害的那些负样本可以被挑出来，参与loss运算以及back
propagation。

详见：<https://www.zhihu.com/question/46292829> ；论文地址：<https://cs.brown.edu/~pff/papers/lsvm-pami.pdf>



## 介绍



​         特征非常重要。最近几十年在许多的视觉识别任务上基本上是基于**SIFT**[29]和**HOG**[7]。但是如果看看在标准视觉识别任务：PASCAL VOC目标检测[15]上的性能, 通常公认在2010-2012年间**进展很慢**；通过构建**集成系统**和采用**成功方法的变种**获得很小的提升。



​        SIFT和HOG是块方向直方统计，在V1中可以粗略的与复杂单元关联的代表; 是首要视觉路径的第一个表面区域。但同时我们也知道**识别**发生在**一系列下游阶段**，这暗示对于**视觉识别**肯能**层次的**、**多阶段****的**计算特征会更加有益。

​                

​        Fukushima’s的“新认知机”[19],一个受生物学启发的、层次的和平移不变的模式识别模型，是这样一个过程的早期尝试。但是新认知机缺乏监督预训练算法。构建在Rumelhart et al. [33]和LeCun et al. [26] 上，表明**反向传播**的**随机梯度下降**法对于训练卷积神经网络(CNNs)有效，CNNs是**新认知机**上**扩展的一类模型**。

​         CNNs在1990年代(e.g., [27])大量使用,但是随着**支持向量机**的崛起变得不流行了。2012年，Krizhevsky et al. [25]通过在ImageNet大规模视觉识别挑战赛(ILSVRC) [9, 10]上大幅提升**图像分类**精度，重燃了人们对CNNs的兴趣。

​        ImageNet结果的意义在ILSVRC 2012研讨会上引起激烈的讨论。争论的关键问题如下：在ImageNet 上CNN分类的结果**多大程度上**能够**泛化**到PASCAL VOC挑战赛的**目标检测**结果上。

​        我们通过桥接图像分类和目标检测的差异来回答此问题。本文是第一个表明，与基于简单的HOG-like特征的系统比较，CNN在PASCAL VOC上**目标检测**性能有**显著提升**。为达到此结果，我们关注两个问题：使用深度网络**定位对象** 和 使用**少量标注**的检测数据训练**大容量的模型**。

​        不同于图像分类，目标检测需要**定位**图像中的**目标**(通常多个). 一种是方法是将定位当做**回归问题**。但是Szegedy et al. [38]和我们做的研究表明这种策略实践中，**效果不好**(VOC 2007数据集上的mAP值为30.5%，与之相比较，我们的方法mAP到达58.5)。一种替代方案是构建滑动窗口检测器。过去20年,CNN经常这样使用，特别是在受限的目标类型上，如**人脸检测**[32,40]和**行人检测**[35]. 为了保持高的空间分辨率，典型的这些CNNs**仅仅**包含**两个卷积**和**池化层**。我们也考虑了滑动窗口的方法，但是我们的网络神经元数量很大，有5个卷积层，有很大的感受野(195 × 195像素)和步长 (32×32 像素) ，导致使用**滑动窗口**模式来**精确定位**成为一个开放的**技术难题**。

​        我们使用在**对象检测**[39]和**语义分割**[5]都取得成功的“recognition using regions”范式[21]来解决定位问题。预测时，我们的方法为输入图像生成约2000个**类别不相关**的region proposals,使用CNN从每个proposal提取固定长度的**特征向量**，然后使用**类别相关**的线性**SVMs分类**. 对于每个region proposal，不管形状如何，我们使用简单的**仿射变形**来计算出一个**固定尺寸**的CNN输入。图Figure 1 是我们方法的概览，并且高亮了我们的效果。我们的系统组合的region proposals和CNNs，我们称之为**R-CNN**: 带CNN特征的Regions.

​        在本文是论文的一个更新版本，我们正面比较了R-CNN和最近提出的OverFeat[34]检测系统在ILSVRC2013 200类检测数据集上的性能。OverFeat使用了一个滑动窗口CNN做检测，目前为止在ILSVRC2013上检测性能最好。结果显示，**R-CNN完胜OverFeat**，mAP达到31.4%，而OverFeat只有24.3%。

​        检测中面对的第二个挑战是标注数据太少，现在可获得的数据远远不够用来训练一个大型卷积神经网络。传统方法多是采用**无监督预训练**，再进行**有监督精调**（如[35]）。本文的第二个核心贡献是在**辅助数据集** (ILSVRC)上进行**有监督预训练**，再在小数据集(PASCAL)上针对特定问题进行**精调**，是在训练数据稀少的情况下一个非常有效的训练大型卷积神经网络的方法。我们的实验中，针对检测的**精调**将mAP**提高了8**个百分点。精调后，我们的系统在VOC 2010上mAP达到了54%的mAP，远远超过基于HOG的deformable part model(DPM)[17, 20]在**高度优化后**的33%。我们也向读者推荐Donahue et al. [12]同时期的工作，他表明Krizhevsky’s **CNN**可以当做(没有精调)一个黑盒的**特征提取器**，没有精调的情况下就可以在多个识别任务上包括**场景分类**、**细粒度的子分类**和**领域适应**方面都表现出色。

​        我们的系统也非常高效，仅有一个合理**小型矩阵向量乘积运算**和**贪婪非极大抑制**的计算是**类别相关**的。这个计算特性源自于**跨类别共享**的特征，维度比之前使用的区域特征(cf. [39])低了两个数量级。

​        理解我们方法的失败案例，对于进一步提高它很有帮助，所以我们借助Hoiem et al. [23]检测分析工具做实验结果的报告。作为本次分析的直接结果，我们发现一个简单的**边框回归**的方法会**明显地减少定位错误****，而**定位错误**也是**最主要**的**错误**情况。

​      介绍技术细节之前，我们注意，由于R-CNN是在**区域**上进行操作，因而可以很自然地扩展到**语义分割**任务上。经过很小的改动，我们就在PASCAL VOC语义分割任务上达到了很有竞争力的结果，在VOC2011测试集上平均语义分割精度达到了47.9%。



总体步骤
--------

 ![R-CNN流程图](pic/R-CNN流程图.png)



### Region proposals生成

       使用selective search为每个图片生成2k个Region proposals(建议区域/候选框)。

### 训练过程

#### a) 监督预训练

      已经在分类数据集上训练好的CNN。

#### b) 精调CNN

输入：

-        将Region
    proposals变形为CNN输入的大小（227\*227）；在变形前，将原始候选框周围增加p(p=16)个像素padding；如果padding超过了原始图片使用Region
    proposals的均值替换。这样生成了Warped region。

-        将warped region 减去自身均值后就是输入了（输入大小227\*227）。

模型说明：

-        cnn结构不变，只修改最后分类个数为N+1;N为对象类别数。

-        使用SGD训练，学习率为0.001（预训练模型学习率的1/10）

-        每个mini-batch 128个样本，其中正样本32个(涵盖所有类，N类)，负样本96个.

-        正样本为Region proposals与Ground Truth的IoU大于0.5的；其它都为负样本

#### c) 对象分类

      对每个类别训练一个SVM分类器。

输入：Region通过CNN提取的特征向量

输出：属于某个类别的评分

正样本：仅仅是Ground Truth代表的Region.

负样本：IoU值小于阈值0.3的Region. 其它Region的忽略

      由于训练样本太大使用hard negative mining method。

#### d) 边框回归

模型：

        算法生成的Region proposals
Box和实际的Ground-Truth肯定存在出入，我们希望这些box能尽可能接近Ground
Truth。对每个 假设Region proposals(P代表)经过如下线性变换到 Ground Truth( $\hat{G}​$代表)；其中x，y代表坐标，w,h代表宽度和高度。

​     $\hat G_x = P_wd_x(P) + P_x$    

​     $\hat G_y = P_hd_y(P) + P_y$  

​     $\hat G_w = P_wexp(d_w(P))$ 

​     $\hat G_h = P_hexp(d_h(P)))$

  

注：$d_*(P) = w^T_*\phi_5(P)$

     （**\* **是x,y,w,h任意一个，

是需要学习的模型参数；$\phi_5(P)$ 是Region proposals第5个池化层的特征）。

 $t_x = (G_x - P_x)/P_w$  (6)

 $t_y = (G_y - P_y)/P_h$  (7)

 $t_w = log(G_w/P_w) $    (8)

$t_h = log(G_h/P_h)$   (9)

 



对于训练的样本对(P, G)；优化的目标就是让$w^T_*\phi_5(P)$ 去拟合$t_*$；使用岭回归模型，优化目标如下：

$w_* = \underset{\hat w_*}{argmin} \sum_i^N(t_*^i - \hat w^T_*\phi_5(P^i))^2 + \lambda||\hat w_*||^2 $

按类别做Bounding-box 回归，所有一共有N\*4个回归函数。

 

训练样本：


 样本是成对出现的（P,G），对每个G，找IoU最大的那个P；并且IoU值大于阈值0.6；则(P,G)构成样本，其它不满足的丢弃。

 

 

### 测试过程

a) 为每个图片生成2k个Region proposals

b) 通过padding后变形为固定大小(227\*227)的Warped region

c) 每个Warped region的**mean-subtracted**值作为CNN的输入，获取CNN的特征值

d) 对CNN的特征值使用SVM预测分类评分。

e) 在每个类别上对2k个Region proposals的评分做non-maximum
suppression-非极大值抑制；排除超过IoU阈值的Region proposals

f) 对评分后保留的Region proposals做边框回归预测最终的边框值。

注意：训练的时候分类预测和边框回归是并行的；测试阶段是串行的，先做分类预测，然后使用对应类别的回归函数做边框回归。

关键点
------

1：精调CNN网络和SVM预测分类模型训练时使用的正负样本为什么不一致？

CNN精调：正样本为Region proposals与Ground Truth的IoU大于0.5的；其它都为负样本。

SVM分类：正样本仅仅是Ground Truth代表的Region；负样本为IoU值小于阈值0.3的Region.
其它Region的忽略。


 作者测试了精调CNN也使用SVM分类时的正负样本，发现结果比使用现在的样本差很多。据此推测正负样本怎么定义不是关键的地方，主要是精调的数据时受限的。当前方式将正样本扩招了30倍，在IoU
值0.5\~1之间有很多抖动的例子。据此推断在精调阶段需要大样本集来避免过拟合。但是，这些抖动的样例不是最优选择，因为对于精确定位没有精调。

     这也是为什么，在CNN精调后，为什么还有训练SVM?
为什么不直接使用CNN最后的N+1分类做目标检测。作者测试了发现在VOC
2007数据集上mAP从54.2%下降到50.9%；这是多个因素组合引起的，包括精调阶段没有突出精确定位，以及训练softmax分类器使用的使用随机负样本，而不是像SVM那样使用困难负样本。

     作者猜测，不一定非要在精调
后使用SVM分类；使用一些其它的tweaks也能有同样的效果

疑问点
------

1：边框回归中的优化目标定义问题


一般回归函数的优化目标(即损失函数)都是预测的Y和实际的Y'的差距最小；这里的优化目标在回归函数的参数项这里。

 

 

 



## 关于我们

我司正招聘文本挖掘、计算机视觉等相关人员，欢迎加入我们；也欢迎与我们在线沟通任何关于数据挖掘理论和应用的问题；

在长沙的朋友也可以线下交流, 坐标: 长沙市高新区麓谷新长海中心 B1栋8A楼09室

公司网址：http://www.embracesource.com/

Email: mick.yi@embracesource.com 或 csuyzt@163.com

