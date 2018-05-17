# G-CNN: an Iterative Grid Based Object Detector

[TOC]





概述
====

本文介绍G-CNN原理及其重要细节，论文地址<https://arxiv.org/abs/1512.07729>,其他参考网址[论文笔记G-CNN](http://blog.csdn.net/u012905422/article/details/52664918)。

G-CNN简介
=========

G-CNN是基于CNN但不需要proposal的目标检测技术。G-CNN从一个固定bounding
boxes的多尺度网格开始，训练一个迭代回归器使得bounding
boxes不断的地朝着目标移动或缩放。G-CNN效果与Fast R-CNN性能相当，速度快5倍多。

在训练阶段，G-CNN首先在图像中获取叠加的多尺度的规则网格（实际网格相互重叠），然后通过ground
truth与每一个网格的IoU进行每一个网格ground
truth的分配，并完成训练过程，使得网格在回归过程中渐渐接近ground truth，见下图。

![C:\\1cc7b447eb582763ef64b8d249cffe5a](media/f002425a0e0af5e657602b9286fd6f35.tmp)

G-CNN训练
=========

单次回归网络无法处理由原始bounding box到groud
truth的非线性变换，因此G-CNN网络通过迭代回归网络的思想一步步完成由原始bounding
box到groud truth的非线性变换。G-CNN结构见下图。

![C:\\f319c31e80cc8de7e17139fbe679ecb0](media/f374ac0e1f400d6699e4d5a4e4c69a59.tmp)

注释：

![C:\\9635a8756930977adf61cf4f124277e5](media/8fda1a31d29b131955f7dbda511d4390.tmp)

![C:\\87c5829b7c39b99a65a46339c5a6692f](media/9e1e79c0e5c068c266aee13a6cd84087.tmp)

    ：第s次迭代的第i个bounding box

![C:\\13f2e2613372f9da7c650fc45400f6b8](media/7b9dede7aafa5d0a844e431adf789131.tmp)

    : 总迭代次数

![C:\\c3dc56ecdd511373c255f2c44d24b697](media/d285715ee9c615a701dfaaebfa12d881.tmp)

    ：与bounding box重叠度最高的groundtruth

![C:\\89f88c5fe17a1ac5c23055924e0cb6e1](media/78561efedd6f5cbd3a27834d8303d71d.tmp)

    =

    ![C:\\c902222ddf8f9f31bf44475a8021982e](media/a0c3c5c04fe61a3490d6cea408c8f3d0.tmp)

    ：每一步迭代的target bounding box

-   红色框：bounding Box；蓝色框：target bounding box

**图中训练步骤**

获取适量spatial pyramid grid of boxes，在每次迭代中进行如下操作：

（1） 对每个bounding box

![C:\\313348e62f8e4e43efce2c8a7ed54e57](media/8da3ca7e479cf9fcf1c0f23826d713ee.tmp)

，通过公式计算target bounding box

![C:\\c902222ddf8f9f31bf44475a8021982e](media/a0c3c5c04fe61a3490d6cea408c8f3d0.tmp)

；

（2）通过CNN网络生成feature map（该CNN网络可以是任意图片分类网络），通过RIO
pooling将

![C:\\313348e62f8e4e43efce2c8a7ed54e57](media/8da3ca7e479cf9fcf1c0f23826d713ee.tmp)

映射到feature map上，并固定大小输送给Fully connected layers；

（3）fast
R-CNN分类器对物体进行分类，然后对每类物体进行回归，回归网络输出4个参数（由

![C:\\313348e62f8e4e43efce2c8a7ed54e57](media/8da3ca7e479cf9fcf1c0f23826d713ee.tmp)

到

![C:\\c902222ddf8f9f31bf44475a8021982e](media/a0c3c5c04fe61a3490d6cea408c8f3d0.tmp)

的变换参数），使

![C:\\313348e62f8e4e43efce2c8a7ed54e57](media/8da3ca7e479cf9fcf1c0f23826d713ee.tmp)

接近

![C:\\c902222ddf8f9f31bf44475a8021982e](media/a0c3c5c04fe61a3490d6cea408c8f3d0.tmp)

，也可以说，

![C:\\313348e62f8e4e43efce2c8a7ed54e57](media/8da3ca7e479cf9fcf1c0f23826d713ee.tmp)

向Ground
Truth方向移动一步，该回归步骤原理详见[边框回归](/confluence/pages/viewpage.action?pageId=10846253)；

（4）更新

![C:\\313348e62f8e4e43efce2c8a7ed54e57](media/8da3ca7e479cf9fcf1c0f23826d713ee.tmp)

，

![C:\\c902222ddf8f9f31bf44475a8021982e](media/a0c3c5c04fe61a3490d6cea408c8f3d0.tmp)

，进行下一次迭代。

**目标函数**

G-CNN目标函数包括两个部分（1）每个训练样本的loss和，（2）每个迭代步骤的loss和，见如下公式。

![C:\\3f4f5b5f6d2f43faed6addfd56507ed1](media/fad870fa9f4546e08b1412e7558cbb2f.tmp)

注释：

![C:\\e19e2ec17fd254bbaa5d2acfa33e6275](media/174c63bec04b0f23aed888328d593820.tmp)

    ：与Ground Truth IOU 小于阈值的bounding boxes

![C:\\e646421448949e9d7d5264502fd90187](media/8d15e4819e23eed8cf77575b0936e76d.tmp)

    ：指示函数，满足条件输出1，否则输出0

![C:\\682322cebd71dc44ebf65a9456431d5b](media/8139dfe5538fbaa77f3d2edb6cd19588.tmp)

    ：由

    ![C:\\313348e62f8e4e43efce2c8a7ed54e57](media/8da3ca7e479cf9fcf1c0f23826d713ee.tmp)

    到

    ![C:\\c902222ddf8f9f31bf44475a8021982e](media/a0c3c5c04fe61a3490d6cea408c8f3d0.tmp)

    的4个预测变换参数，

    ![C:\\d26c14c466f323bbc4717b3eaff30f10](media/388ba9c98a7ad04ece7162bc710d98c5.tmp)

    是bounding box 的类别标签

![C:\\d9333b511bea9e289b3c7191445830dd](media/8d75063b30f4b4394b9a478bc86f775c.tmp)

    ：由

    ![C:\\313348e62f8e4e43efce2c8a7ed54e57](media/8da3ca7e479cf9fcf1c0f23826d713ee.tmp)

    到

    ![C:\\c902222ddf8f9f31bf44475a8021982e](media/a0c3c5c04fe61a3490d6cea408c8f3d0.tmp)

    的4个真实变换参数

G-CNN测试
=========

在测试时将整个网络层分为两部分：global parts和regression parts。global
parts的输入是初始化的bounding box

![C:\\a96b9abe0ce3c176dba7ec0b57c0839d](media/473eda22c607f5c84bcae31869b7954a.tmp)

 ，对于每幅图像前向计算仅计算一次。regression parts以global
parts最后一个层的输出作为输入，迭代运行

![C:\\f10d7a8d2ccc0ac4be1e63aba0b8f5eb](media/eaecac03f2238250e66aad54ccd33787.tmp)

次以生成对bounding box的修改(bounding box
modification)。对于每一个box针对每一类获得置信分数，用最可能类别的回归器来更新box的位置。

![C:\\71187ebda7d755838525828f3f83121f](media/1e4de02ad53641afdbb98810e1c1c8a3.tmp)

G-CNN效果与检测速度
===================

测试表明，当

![C:\\4ac5f2ab4a3e3bb8602fed41778bff71](media/43d53964ec88ab922cdd3d878bd05c2a.tmp)

时，G-CNN的性能和\~2k个proposal的fast R-CNN不相上下，见下表。

![C:\\3316ec2bedbd7f6b2d52c789c0f1b0c3](media/4498a6350896df2bc598467b75460393.tmp)

![C:\\44712e22bdd60692f3f54c2c2bf2cd91](media/4c8f70d71499e4d81529b52d174923b0.tmp)

下图展示了部分G-CNN测试效果。

![C:\\b19f6c239eb658e9bccc2cd6ea4d8d36](media/8365c348599290fdf00d97875d872d52.tmp)

G-CNN图片检测速度为363ms/image，Fast
R-CNN图片检测速度为2050ms/image，是G-CNN图片检测速度的5倍多。

G-CNN论文结论总结
=================

-   训练回归网络迭代次数=3，初始bounding box=180个，G-CNN的性能与Fast
    R-CNN不相上下；

-   训练回归网络迭代次数在3以上，继续增大迭代次数MAP提高不显著；

-   根据实验，单步回归网络误差较大，MAP与3步迭代回归网络小3%；

-   G-CNN图片检测速度比Fast R-CNN快5倍多。

G-CNN局限性
===========

（1）当目标物体与周边物体类似时，目标物体选择错误；

![C:\\208e733c8964dfad279a5bd1d8d4bafa](media/67d11ad82b93200fcb0d8b51ed48367b.tmp)

（2）类似物体重叠时localization出错；

![C:\\ad970ec8c7fa65543e4d47a9a7feeff3](media/55d7b32d423e2a4a0fc1af61964be7e2.tmp)

![C:\\6d0601e72fd3c966e534669d46532662](media/5dd0ea2b5412dc755db1e4f379f1e563.tmp)

（3）小物体检测出错；

![C:\\f65911bbd41132008d6299cd07aedbcc](media/fcb6c75af0164a9f1dcc6da113b7d5ed.tmp)

（4）当目标物体的姿态复杂或初始bounding box太小时，物体定位错误。

![C:\\ed76b25363890f5a786a30446a51df98](media/86833941e359c612cafcc831cc0e04e0.tmp)

疑问点
======

1、多尺度的规则网格怎样初始化来的？

论文中提到In all the experiments, the G-CNN regression network is trained on an
initial overlapping spatial pyramid with [2,5,10]scales(i.e.the bounding boxes
in the coarsest level are (imwidth/2,imheight/2) pixels etc.). During training,
we used [0.9,0.8,0.7] overlap for each spatial scale respectively. By overlap of
α we mean that the horizontal and vertical strides are widthcell ∗ (1 − α) and
heightcell ∗ (1 − α) respectively。目前G-CNN代码未开源，相关说明资料不足。

经讨论，论文中表述的意思是：在原图（1000\*600）的基础上提取多尺度box，scale是[2,5,10]，box_size=
原图size/scale,由于scale有3个数，box_size就有三种，scale[2,5,10]的alpha**分别**是 [0.9,0.8,0.7]，具体获取初始化box的计算过程见下图。

![C:\\5f2f1dc1838a224d74773af43d6c427e](media/296daae00b51fd77211d6222d0882a3b.tmp)

 

 

 



## 关于我们

我司正招聘文本挖掘、计算机视觉等相关人员，欢迎加入我们；也欢迎与我们在线沟通任何关于数据挖掘理论和应用的问题；

在长沙的朋友也可以线下交流, 坐标: 长沙市高新区麓谷新长海中心 B1栋8A楼09室

公司网址：http://www.embracesource.com/

Email: mick.yi@embracesource.com 或 csuyzt@163.com

