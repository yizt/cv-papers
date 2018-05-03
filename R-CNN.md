# cv-papers
R-CNN
=====

 

-   [依赖知识](#R-CNN-依赖知识)

-   [知识点](#R-CNN-知识点)

    -   [selective search](#R-CNN-selectivesearch)

    -   [NMS](#R-CNN-NMS)

    -   [hard negative mining method](#R-CNN-hardnegativeminingmethod)

-   [总体步骤](#R-CNN-总体步骤)

    -   [Region proposals生成](#R-CNN-Regionproposals生成)

    -   [训练过程](#R-CNN-训练过程)

        -   [a) 监督预训练](#R-CNN-a)监督预训练)

        -   [b) 精调CNN](#R-CNN-b)精调CNN)

        -   [c) 对象分类](#R-CNN-c)对象分类)

        -   [d) 边框回归](#R-CNN-d)边框回归)

    -   [测试过程](#R-CNN-测试过程)

-   [关键点](#R-CNN-关键点)

-   [疑问点](#R-CNN-疑问点)



        R-CNN
论文地址：<https://arxiv.org/pdf/1311.2524.pdf> 。目前网上讲解的比较好的 [R-CNN学习总结](https://zhuanlan.zhihu.com/p/30316608)
。R-CNN
论文里面涉及的知识点还是挺多的，网上资料基本上没有特别清晰的。建议看看网上资料对于有疑问的地方再看论文。本文以 [R-CNN学习总结](https://zhuanlan.zhihu.com/p/30316608)为基础，在 [R-CNN学习总结](https://zhuanlan.zhihu.com/p/30316608)中有说明的不再列出来。

 

依赖知识
--------

    a） 熟悉CNN，以及常用的CNN分类网络AlexNet,VGG,GoogLeNet等

    b )  了解预训练、迁移学习

    c） 了解目标检测的基本概念，指导IoU、Ground Truth

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

总体步骤
--------

 

![C:\\9e1bb637e7aa4a712fac928e9e068587](media/9eee282c61c93c8cdc441fc2dfedd396.tmp)

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
Truth。对每个 假设Region proposals(P代表)经过如下线性变换到 Ground Truth( 

![C:\\2a32294f61074c2af22a5f349c080ab3](media/d9a149ccf75079ac13dd606487a229a6.tmp)

代表)；其中x，y代表坐标，w,h代表宽度和高度。

![C:\\7942d4e2187774bda1e82e67fc4f6602](media/2f7e0de51beb59ecb03e42d10a87d0ed.tmp)

注：

![C:\\b9f1e21c4e8ca2cf3eade5bc620a9b37](media/be08cfc5867835d4f38a08b22045ad88.tmp)

     （**\* **是x,y,w,h任意一个，

![C:\\0711687b161c50cd2f1dba8afd4bce31](media/cf1f1b97a15d02dd6a741c1234f546cc.tmp)

是需要学习的模型参数；

![C:\\09352378e53006f2a44291213bc72288](media/0d014a27c415e466c4db56301ba843f6.tmp)

 是Region proposals第5个池化层的特征）。

 

 

![C:\\5be4ac1dcb37c933b15ddcb03a6b33aa](media/53880b17f18bffca23165b94d5bdfdfc.tmp)

对于训练的样本对(P, G)；优化的目标就是让

![C:\\deb97ceab91d80e9f1d20652fca7fe21](media/c30cec92109a348d5312542d27f0394d.tmp)

 去拟合

![C:\\dd33a066c8433a5cf5a03728ad23b857](media/036c7abbe349e1683314371d2e557482.tmp)

；使用岭回归模型，优化目标如下：

![C:\\17f9e5add7b8387e7d61e85abeb08ca4](media/b9c0f1d681ee6bd09c43d2994d1ab781.tmp)

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

