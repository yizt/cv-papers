[TOC]

We present a simple, highly modularized network architecture for image classification. Our network is constructed by repeating a building block that aggregates a set of transformations with the same topology. Our simple design results in a homogeneous, multi-branch architecture that has only a few hyper-parameters to set. This strategy exposes a new dimension, which we call “cardinality” (the size of the set of transformations), as an essential factor in addition to the dimensions of depth and width. On the ImageNet-1K dataset, we empirically show that even under the restricted condition of maintaining complexity, increasing cardinality is able to improve classification accuracy. Moreover, increasing cardinality is more effective than going deeper or wider when we increase the capacity. Our models, named ResNeXt, are the foundations of our entry to the ILSVRC 2016 classification task in which we secured 2nd place. We further investigate ResNeXt on an ImageNet-5K set and the COCO detection set, also showing better results than its ResNet counterpart. The code and models are publicly available online https://github.com/facebookresearch/ResNeXt
.

​        我们为图像分类提出一个简单、高度模块化的网络结构。网络通过重复一个block来构建，这个block聚合了一组有相同拓扑的转换。我们的简单设计产生了一个同构的、多分支的体系结构，它只需要设置几个超参数。这个策略发布了一个新的维度，我们称之为"基数-cardinality"(这组转换的大小)，除了深度和宽度之外的一个关键因子。在ImageNet-1K数据集上，我们的经验表明，在控制复制度的受限情况下，增加基数可以提升分类精度。而且当提升模型容量时(即增加模型复杂度)，提升基数比更深或更宽更加有效。我们的模型叫ResNeXt，是参加ILSVRC 2016分类任务的基础，我们获得了第二名。我们进一步在ImageNet-5K 数据集和COCO检测数据集上研究ResNeXt,同样展示了比对应的ResNet更好的结果。代码和模型公布在https://github.com/facebookresearch/ResNeXt



## 1.引言

Research on visual recognition is undergoing a transition from “feature engineering” to “network engineering” [25, 24, 44, 34, 36, 38, 14]. In contrast to traditional handdesigned features (e.g., SIFT [29] and HOG [5]), feature learned by neural networks from large-scale data [33] require minimal human involvement during training, and can be transferred to a variety of recognition tasks [7, 10, 28]. Nevertheless, human effort has been shifted to designing better network architectures for learning representations.

​         视觉识别的研究正经历从"特征工程"到"网络工程的转换"[25, 24, 44, 34, 36, 38, 14]。与传统的手工设计特征相比(如：SIFT [29] 和 HOG [5]), 通过神经网络从大规模数据[33]中学习特征,在训练时需要最少人工参与，并可以迁移到各种各样的识别任务上[7,10,28]. 不过，人的工作已经转移到为学习表示设计更好的网络架构。

Designing architectures becomes increasingly difficult with the growing number of hyper-parameters (width2 , filter sizes, strides, etc.), especially when there are many layers.The VGG-nets [36] exhibit a simple yet effective strategy of constructing very deep networks: stacking build ing blocks of the same shape. This strategy is inherited by ResNets [14] which stack modules of the same topology. This simple rule reduces the free choices of hyperparameters, and depth is exposed as an essential dimension in neural networks. Moreover, we argue that the simplicity of this rule may reduce the risk of over-adapting the hyperparameters to a specific dataset. The robustness of VGGnets and ResNets has been proven by various visual recognition tasks [7, 10, 9, 28, 31, 14] and by non-visual tasks involving speech [42, 30] and language [4, 41, 20]

​         随着超参数的增加(通道数、滤波器尺寸、步长等)设计网络架构变得日渐困难，特别是网络的层数很多时。VGG-nets [36]展示了一个简单有效构建非常深的网络的策略：堆叠相同形状的blocks. ResNets[14]继承了这种策略，堆叠的相同拓扑的模块。而且，我们认为，该规则的简单性可能会降低过度调整超参数以适应特定数据集的风险。VGGnets 和ResNets 的鲁棒性已经被各种视觉识别任务[7, 10, 9, 28, 31, 14]和非视觉任务包括语音[42,30]和语音[4,41,20]

Unlike VGG-nets, the family of Inception models [38,17, 39, 37] have demonstrated that carefully designed topologies are able to achieve compelling accuracy with low theoretical complexity. The Inception models have evolved over time [38, 39], but an important common property is a split-transform-merge strategy. In an Inception module, the input is split into a few lower-dimensional embeddings (by 1×1 convolutions), transformed by a set of specialized filters (3×3, 5×5, etc.), and merged by concatenation. It can be shown that the solution space of this architecture is a strict subspace of the solution space of a single large layer
(e.g., 5×5) operating on a high-dimensional embedding. The split-transform-merge behavior of Inception modules is expected to approach the representational power of large and dense layers, but at a considerably lower computational complexity

​        不同于VGG-nets,Inception模型家族[38,17,39,37]证明了精心设计的拓扑能够以较低的**理论复杂度**达到令人信服的准确性。Inception模型随着时间改进[38,39], 但是一个共有的属性是分割-转换-合并策略。在一个Inception模块中，输入分割为几个低维嵌入(通过$1 \times 1 $卷积) ，使用特定的一组滤波转换($3 \times 3,5\times 5$等)，然后通过拼接合并。可以看出，该体系结构的解空间是单个大层(例如,5×5)操作在一个高维嵌入解空间的严格子空间。Inception 模块中的分割-转换-合并行为，被期望可以接近大的、密集层的表示能力，但又有一个相对低的计算复杂度。





Despite good accuracy, the realization of Inception models has been accompanied with a series of complicating factors — the filter numbers and sizes are tailored for each individual transformation, and the modules are customized stage-by-stage. Although careful combinations of these components yield excellent neural network recipes, it is in general unclear how to adapt the Inception architectures to new datasets/tasks, especially when there are many factors and hyper-parameters to be designed.

​         

In this paper, we present a simple architecture which
adopts VGG/ResNets’ strategy of repeating layers, while
exploiting the split-transform-merge strategy in an easy, extensible
way. A module in our network performs a set
of transformations, each on a low-dimensional embedding,
whose outputs are aggregated by summation. We pursuit a
simple realization of this idea — the transformations to be
aggregated are all of the same topology (e.g., Fig. 1 (right)).
This design allows us to extend to any large number of
transformations without specialized designs.



Interestingly, under this simplified situation we show that
our model has two other equivalent forms (Fig. 3). The reformulation
in Fig. 3(b) appears similar to the InceptionResNet
module [37] in that it concatenates multiple paths;
but our module differs from all existing Inception modules
in that all our paths share the same topology and thus the
number of paths can be easily isolated as a factor to be investigated.
In a more succinct reformulation, our module
can be reshaped by Krizhevsky et al.’s grouped convolutions
[24] (Fig. 3(c)), which, however, had been developed
as an engineering compromise.



We empirically demonstrate that our aggregated transformations
outperform the original ResNet module, even
under the restricted condition of maintaining computational
complexity and model size — e.g., Fig. 1(right) is designed
to keep the FLOPs complexity and number of parameters of
Fig. 1(left). We emphasize that while it is relatively easy to
increase accuracy by increasing capacity (going deeper or
wider), methods that increase accuracy while maintaining
(or reducing) complexity are rare in the literature



Our method indicates that cardinality (the size of the
set of transformations) is a concrete, measurable dimension
that is of central importance, in addition to the dimensions
of width and depth. Experiments demonstrate that increasing
cardinality is a more effective way of gaining accuracy
than going deeper or wider, especially when depth and
width starts to give diminishing returns for existing models.



Our neural networks, named ResNeXt (suggesting the
next dimension), outperform ResNet-101/152 [14], ResNet200
[15], Inception-v3 [39], and Inception-ResNet-v2 [37]
on the ImageNet classification dataset. In particular, a
101-layer ResNeXt is able to achieve better accuracy than
ResNet-200 [15] but has only 50% complexity. Moreover,
ResNeXt exhibits considerably simpler designs than all Inception
models. ResNeXt was the foundation of our submission
to the ILSVRC 2016 classification task, in whichwe secured second place. This paper further evaluates
ResNeXt on a larger ImageNet-5K set and the COCO object
detection dataset [27], showing consistently better accuracy
than its ResNet counterparts. We expect that ResNeXt will
also generalize well to other visual (and non-visual) recognition
tasks.



## 2. 相关工作

Multi-branch convolutional networks. The Inception
models [38, 17, 39, 37] are successful multi-branch architectures
where each branch is carefully customized.
ResNets [14] can be thought of as two-branch networks
where one branch is the identity mapping. Deep neural decision
forests [22] are tree-patterned multi-branch networks
with learned splitting functions.
Grouped convolutions. The use of grouped convolutions
dates back to the AlexNet paper [24], if not earlier. The
motivation given by Krizhevsky et al. [24] is for distributing
the model over two GPUs. Grouped convolutions are supported
by Caffe [19], Torch [3], and other libraries, mainly
for compatibility of AlexNet. To the best of our knowledge,
there has been little evidence on exploiting grouped convolutions
to improve accuracy. A special case of grouped convolutions
is channel-wise convolutions in which the number
of groups is equal to the number of channels. Channel-wise
convolutions are part of the separable convolutions in [35].
Compressing convolutional networks. Decomposition (at
spatial [6, 18] and/or channel [6, 21, 16] level) is a widely
adopted technique to reduce redundancy of deep convolutional
networks and accelerate/compress them. Ioannou
et al. [16] present a “root”-patterned network for reducing
computation, and branches in the root are realized
by grouped convolutions. These methods [6, 18, 21, 16]
have shown elegant compromise of accuracy with lower
complexity and smaller model sizes. Instead of compression,
our method is an architecture that empirically shows
stronger representational power.
Ensembling. Averaging a set of independently trained networks
is an effective solution to improving accuracy [24],
widely adopted in recognition competitions [33]. Veit et al.
[40] interpret a single ResNet as an ensemble of shallower
networks, which results from ResNet’s additive behaviors
[15]. Our method harnesses additions to aggregate a set of
transformations. But we argue that it is imprecise to view
our method as ensembling, because the members to be aggregated
are trained jointly, not independently