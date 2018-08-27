# Aggregated Residual Transformations for Deep Neural Networks

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

​         尽管精度很好，Inception模块的实现伴随着一系列复杂的因素-滤波器的数量和尺寸对每个独立的转换都是定做的，模块是分阶段定制的。尽管精心的组合这些组件产生了优秀的神经网络配方，如何将Inception架构适应到新的数据集/任务上通常不是明确的，特别是有这么因子和超参需要设计。

In this paper, we present a simple architecture which adopts VGG/ResNets’ strategy of repeating layers, while exploiting the split-transform-merge strategy in an easy, extensible way. A module in our network performs a set of transformations, each on a low-dimensional embedding, whose outputs are aggregated by summation. We pursuit a simple realization of this idea — the transformations to be aggregated are all of the same topology (e.g., Fig. 1 (right)). This design allows us to extend to any large number of transformations without specialized designs.

​        本文，我们提出一个简单的架构，它采用了VGG/ResNets的重复层(layers)的策略，同时使用一种简单、可扩展的方式来利用分割-转换-合并策略。我们的网络的一个模块执行一组转换，每个转换在一个低维嵌入上执行，输出通过求和合并。我们追求这种概念(分割-转换-合并)的简单实现-聚合在一起的转换有相同的拓扑(如图Figure 1右)。这种设计思路允许将转换数量扩展到任意大，而不需要特殊设计。

Interestingly, under this simplified situation we show that our model has two other equivalent forms (Fig. 3). The reformulation in Fig. 3(b) appears similar to the InceptionResNet module [37] in that it concatenates multiple paths; but our module differs from all existing Inception modules in that all our paths share the same topology and thus the number of paths can be easily isolated as a factor to be investigated. In a more succinct reformulation, our module can be reshaped by Krizhevsky et al.’s grouped convolutions [24] (Fig. 3(c)), which, however, had been developed as an engineering compromise.

​        有趣的是，在这种简单情况下，我们说明我们的模型有其它两种等价形式(图Figure 3)，图Figure 3 (b)的重构呈现了类似InceptionResNet 模块的合并多个路径；但是我们的模块与所有现存的Inception 模块不同，我们的路径共享相同的拓扑，路径的数量可以轻易的独立出来作为一个因子来研究。在一个更简洁的重构中，我们的模块可以被Krizhevsky et al.’s的分组卷积[24] (Fig. 3(c))改造, 而分组卷积是被当做一个工程折中方案设计出来的。

We empirically demonstrate that our aggregated transformations outperform the original ResNet module, even under the restricted condition of maintaining computational complexity and model size — e.g., Fig. 1(right) is designed to keep the FLOPs complexity and number of parameters of Fig. 1(left). We emphasize that while it is relatively easy to increase accuracy by increasing capacity (going deeper or wider), methods that increase accuracy while maintaining (or reducing) complexity are rare in the literature

​        我们实验显示，聚合转换优于原本的ResNet模块，即使在保持计算复杂度和模型大小的受限条件下-如：图Figure 1(右) 被设计为保持与图Figure 1(左)的FLOPs复杂度和参数量大小。我们强调一下，通过增加容量(变得更深或更宽)来提升模型精度相对简单，提升精度同时保持(或减少)复杂度在文献中很少有。

Our method indicates that cardinality (the size of the set of transformations) is a concrete, measurable dimension that is of central importance, in addition to the dimensions of width and depth. Experiments demonstrate that increasing cardinality is a more effective way of gaining accuracy than going deeper or wider, especially when depth and width starts to give diminishing returns for existing models.

​         我们的方法表明，基数(一组转换的数量)是除了宽度和高度之外具体的、可度量的维度，并且是核心重要的。实验证明增加基数是比变得更深或更宽更加有效的提升精度的方法，特别是当深度和宽度开始使现有模型的收益递减时。

Our neural networks, named ResNeXt (suggesting the next dimension), outperform ResNet-101/152 [14], ResNet200 [15], Inception-v3 [39], and Inception-ResNet-v2 [37] on the ImageNet classification dataset. In particular, a 101-layer ResNeXt is able to achieve better accuracy than ResNet-200 [15] but has only 50% complexity. Moreover, ResNeXt exhibits considerably simpler designs than all Inception models. ResNeXt was the foundation of our submission to the ILSVRC 2016 classification task, in whichwe secured second place. This paper further evaluates ResNeXt on a larger ImageNet-5K set and the COCO object detection dataset [27], showing consistently better accuracy than its ResNet counterparts. We expect that ResNeXt will also generalize well to other visual (and non-visual) recognition tasks.

​          我们的神经网络，称作ResNeXt(提出了下一个维度)，在ImageNet分类数据集上由于ResNet-101/152 [14], ResNet200 [15], Inception-v3 [39], and Inception-ResNet-v2 [37]。特别是一个101-层的ResNeXt能够达到比ResNet-200[15]更好的精度，但只有其50%的复杂度。而且，ResNeXt展示了相对所有Inception模型来说更简单的设计。ResNeXt 是我们在ILSVRC 2016分类任务中提交物的基础，获得了第二名。本文进一步在更大的ImageNet-5K数据集和COCO对象检测数据集[27]上评估ResNeXt，显示了比它对应的ResNet一致的、更好的精度。我们认为ResNeXt同样能够很好的泛化的其它视觉(和非视觉的)识别任务上。



## 2. 相关工作

### Multi-branch convolutional networks

The Inception models [38, 17, 39, 37] are successful multi-branch architectures where each branch is carefully customized. ResNets [14] can be thought of as two-branch networks where one branch is the identity mapping. Deep neural decision forests [22] are tree-patterned multi-branch networks with learned splitting functions.。

​         Inception模块[38,17,39,37]是成功的多分支架构，它的每个分支是精心定制的。ResNets[14]可以看做是一个二分支网络，其中一个分支是恒等映射。深度神经决策森林[22]是具有学习分裂功能的树状多分支网络。

### Grouped convolutions

The use of grouped convolutions dates back to the AlexNet paper [24], if not earlier. The motivation given by Krizhevsky et al. [24] is for distributing the model over two GPUs. Grouped convolutions are supported
by Caffe [19], Torch [3], and other libraries, mainly for compatibility of AlexNet. To the best of our knowledge, there has been little evidence on exploiting grouped convolutions to improve accuracy. A special case of grouped convolutions is channel-wise convolutions in which the number of groups is equal to the number of channels. Channel-wise convolutions are part of the separable convolutions in [35].

​        分组卷积的使用可以追溯到AlexNet论文[24], 如果没有更早。Krizhevsky et al. [24]给出的动机是将模型分布到两个GPU上。Caffe [19], Torch [3], 和其它库都支持分组卷积，主要为了兼容AlexNet。据我们所知，很少有证件表明利用分组卷积来提高精度。分组卷积的一个特例是分通道卷积，这种情况下分组数就是通道数。分通道卷积是可分卷积[35]的一部分。

### Compressing convolutional networks

Decomposition (at spatial [6, 18] and/or channel [6, 21, 16] level) is a widely adopted technique to reduce redundancy of deep convolutional networks and accelerate/compress them. Ioannou et al. [16] present a “root”-patterned network for reducing computation, and branches in the root are realized by grouped convolutions. These methods [6, 18, 21, 16] have shown elegant compromise of accuracy with lower complexity and smaller model sizes. Instead of compression, our method is an architecture that empirically shows stronger representational power.

​        分解(在空间[6,18]或通道[6,21,16]级别)是广泛采用的用于检测深度卷积网络冗余和加速/压缩它的技术。 Ioannou et al. [16] 提出"根"-模式网络来减少计算，根中的分支通过分组卷积实现。[6, 18, 21, 16]中的方法展示了精度和较低复杂度和较小模型大小的优雅折中。不同于压缩，我们的方法是在实验中显示更强的表达能力的一种架构。

### Ensembling

 Averaging a set of independently trained networks is an effective solution to improving accuracy [24], widely adopted in recognition competitions [33]. Veit et al. [40] interpret a single ResNet as an ensemble of shallower networks, which results from ResNet’s additive behaviors [15]. Our method harnesses additions to aggregate a set of transformations. But we argue that it is imprecise to view our method as ensembling, because the members to be aggregated are trained jointly, not independently。

​        取一组独立训练网络的均值是一种提升精度[24]的有效解决方案, 在识别竞赛中被广泛采用。Veit et al. [40]将单个ResNets解释为浅层网络的一个集成，ResNet的结果来自与加法行为[15]。我们的方法使用附加的聚合一组转换。但是我们认为将我们的当做集成是不确切的，因为聚合在一起的成员是联合训练的，不是独立训练的。



## 3. 方法

### 3.1 模板

We adopt a highly modularized design following VGG/ResNets. Our network consists of a stack of resid-ual blocks. These blocks have the same topology, and are subject to two simple rules inspired by VGG/ResNets: (i) if producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes), and (ii) each time when the spatial map is downsampled by a factor of 2, the width of the blocks is multiplied by a factor of 2. The second rule ensures that the computational complexity, in terms of FLOPs (floating-point operations, in #of multiply-adds), is roughly the same for all blocks.

​         我们遵循VGG/ResNets，采用高度模块化设计。我们的网络包含一堆残差block，这些block有相同的拓扑，受VGG/ResNets启发，这些block满足两个简单规则：(i) 如果产生空间大小相同的特征图，这些block共享相同的超参(宽度和滤波器尺寸)，(ii) 当空间特征图以因子2下采样时，block的宽度乘上因子2。第二条规则确保了计算复杂度，对于所有block而言FLOPs(在multiply-add中是浮点运算)大致相同。

With these two rules, we only need to design a template module, and all modules in a network can be determined accordingly. So these two rules greatly narrow down the design space and allow us to focus on a few key factors. The networks constructed by these rules are in Table 1.

​        有了这两个规则，我们只需要设计一个模板模块，网络中的所有模块都可以被相应地确定。因此，这两条规则大大缩小了设计空间，使得我们关注几个关键因素。根据这两条规则构建的网络见表Table(1)

### 3.2. Revisiting Simple Neurons

The simplest neurons in artificial neural networks perform inner product (weighted sum), which is the elementary transformation done by fully-connected and convolutional layers. Inner product can be thought of as a form of aggregating transformation:

​          人工神经网络最简单的神经元执行內积(加权求和),  是全连接层和卷积层执行基本转换。內积可以看做聚合转换的一种形式：
$$
\sum_{i=1}^D w_ix_i \tag 1
$$


​       其中 $x = [x_1, x_2, ..., x_D]$ is a D-channel input vector to the neuron and wi is a filter’s weight for the i-th chan-nel. This operation (usually including some output nonlinearity) is referred to as a “neuron”. See Fig. 2.

​       其中 $x = [x_1, x_2, ..., x_D]$ 是神经元的D通道输入向量，$w_i$ 是滤波器第i个通道的权重。这个操作(通常包括一些非线性输出)被称为“神经元”。

The above operation can be recast as a combination of splitting, transforming, and aggregating. (i) Splitting: the vector x is sliced as a low-dimensional embedding, and in the above, it is a single-dimension subspace xi. (ii)Transforming: the low-dimensional representation is transformed,and in the above, it is simply scaled: wixi. (iii)Aggregating: the transformations in all embeddings are aggregated by PD 	i=1.

​      上面的操作可以作为分割、转换和聚合的组合重新构建。(1)分割：向量$x$ 切片为一个低维嵌入，上面这个就是一个单维子空间$x_i$ ; (ii)转换：低维表示被转换，上面这个，它被简单缩放:$w_ix_i$ ; (iii)聚合：所有嵌入的转换通过$\sum_{i=1}^D$ 聚合。



### 3.3. Aggregated Transformations

Given the above analysis of a simple neuron, we consider replacing the elementary transformation (wixi) with a more generic function, which in itself can also be a network. In contrast to “Network-in-Network” [26] that turns out to increase the dimension of depth, we show that our “Network-in-Neuron” expands along a new dimension

​         分析了简单的神经元，我们考虑将基础的转换($w_ix_i$)替换为更通用的函数,函数本身可以是一个网络。与“Network-in-Network” [26] 相比，它增加深度，我们展示了我们的“Network-in-Neuron”沿着一个新的维度扩展。

Formally, we present aggregated transformations as:

形式上，我们提出的聚合转换为：

$\cal F(x)= \sum_{i=0}^C T_i(x), \tag 2$
where Ti(x) can be an arbitrary function. Analogous to a simple neuron, Ti should project x into an (optionally lowdimensional) embedding and then transform it

​        $\cal T_i(x)$ 可以是任意函数，类似一个简单神经元，$\cal T_i$ 投影x到一个嵌套(通常是低维的)，然后转换它。

In Eqn.(2), C is the size of the set of transformations to be aggregated. We refer to C as cardinality [2]. In
Eqn.(2) C is in a position similar to D in Eqn.(1), but C need not equal D and can be an arbitrary number. While the dimension of width is related to the number of simple transformations (inner product), we argue that the dimension of cardinality controls the number of more complex transformations. We show by experiments that cardinality is an essential dimension and can be more effective than the dimensions of width and depth.

​         在等式(2)中C是需要聚合的一组转换大小，我们称C为基数[2]。C在等式(2)中的位置与D在等式(1)中类似，但是C需要等于D，可以是任意数值。维度宽度与简单转换(内积)的数量有关，我们认为维度基数控制更复杂转换的数量。我们通过实验证明，基数是一个基本的维度，比宽度和深度(这两个维度)更有效。

In this paper, we consider a simple way of designing the transformation functions: all Ti’s have the same topology. This extends the VGG-style strategy of repeating layers of the same shape, which is helpful for isolating a few factors and extending to any large number of transformations. We set the individual transformation Ti to be the bottleneck shaped architecture [14], as illustrated in Fig. 1 (right). In this case, the first 1×1 layer in each Ti produces the lowdimensional embedding

​          本文中，我们考虑一个简单的方式设计转换函数：所有的$\cal T_i$ 有相同的拓扑，这扩展了重复相同形状的层的VGG-style策略, 这有助于隔离一些因素并扩展转换数量到任意大。如图Figure 1(右)描绘的，我们将单个转换$\cal T_i$设置为bottleneck形状的架构[14]。这种情况下，$\cal T_i$ 中第一个$1 \times 1$ 的层产生低维嵌入。

The aggregated transformation in Eqn.(2) serves as the residual function [14] (Fig. 1 right):

​        等式(2)中的聚合转换充当残差函数[14] (图Figure 1右)：
$$
\cal y=x+\sum_{i=1}^C T(x_i) \tag3
$$
where y is the output.



**Relation to Inception-ResNet.** 

Some tensor manipulations show that the module in Fig. 1(right) (also shown in Fig. 3(a)) is equivalent to Fig. 3(b).3 Fig. 3(b) appears similar to the Inception-ResNet [37] block in that it involves branching and concatenating in the residual function. But unlike all Inception or Inception-ResNet modules, we share the same topology among the multiple paths. Our module requires minimal extra effort designing each path.

​        一些张量操作表明，图Figure 1(右)中的模块(如图Figure 3(a)所示)与图Figure 3(b)等价。图Figure 3(b)与incep - resnet[37] block类似，因为它涉及到残差函数的分支和连接。但是不同于所有Inception或Inception-ResNet模块，我们在多个路径之间共享相同的拓扑。我们的模块需要最少的额外工作来设计每个路径。

**Relation to Grouped Convolutions.** 

The above module becomes more succinct using the notation of grouped convolutions [24]. This reformulation is illustrated in Fig. 3(c). All the low-dimensional embeddings (the first 1×1 layers) can be replaced by a single, wider layer (e.g., 1×1, 128-d in Fig 3(c)). Splitting is essentially done by the grouped convolutional layer when it divides its input channels into groups. The grouped convolutional layer in Fig. 3(c) performs 32 groups of convolutions whose input and output channels are 4-dimensional. The grouped convolutional layer concatenates them as the outputs of the layer. The block in Fig. 3(c) looks like the original bottleneck residual block in Fig. 1(left), except that Fig. 3(c) is a wider but sparsely connected module.

​          使用分组卷积[24]的符号，上面的模块变得更加简洁；图Figure 3(c)显示了这种重构。所有的低维嵌入(第一个1×1层)可以用一个更宽的层(例如,在图Figure 3 (c)中1×1,128维)替换。分割基本上是由分组卷积层在将输入通道划分为组时完成的。图Figure 3(c)中分组的卷积层执行32组输入和输出通道为4维的卷积。分组的卷积层将它们连接到一起作为层的输出。图Figure 3(c)中的block与图Figure 1(左)中的bottlenetck块相似，除了图3(c)是一个更宽的、稀疏的连接模块。

We note that the reformulations produce nontrivial topologies only when the block has depth ≥3. If the block has depth = 2 (e.g., the basic block in [14]), the reformulations lead to trivially a wide, dense module. See the illustration in Fig. 4.

​         我们注意到重构产生非凡的拓扑只有block的深度≥3；如果块的深度为2(如[14]中基本block), 重构就变成一个普通的宽的、密集模块了；见图Figure 4中的描绘。

**Discussion.** 

We note that although we present reformulations that exhibit concatenation (Fig. 3(b)) or grouped convolutions (Fig. 3(c)), such reformulations are not always applicable for the general form of Eqn.(3), e.g., if the transformation Ti takes arbitrary forms and are heterogenous. We choose to use homogenous forms in this paper because they are simpler and extensible. Under this simplified case, grouped convolutions in the form of Fig. 3(c) are helpful for easing implementation.

​       我们注意到，尽管我们提出了显示连接(图Figure 3(b))或分组卷积(图Figrue 3(c))的重构方式，但这种重构对公式(3)一般形式不总是合适的，如果变换$\cal T_i$是任意形式的并且是异构的。在本文中，我们选择使用同构形式，因为它们更简单，并且可扩展。在这种简化的情况下，以图Figure 3(c)中分组卷积的形式有助于简化实现。



### 3.4. Model Capacity

Our experiments in the next section will show that our models improve accuracy when maintaining the model complexity and number of parameters. This is not only interesting in practice, but more importantly, the complexity and number of parameters represent inherent capacity of models and thus are often investigated as fundamental properties of deep networks [8].

​        下一节的实验会展示，我们的模型在保持模型复杂度和参数量时提升了精度。这不仅在实践中很有趣，而且更重要的是，复杂度和参数量代表了模型的固有容量，因此常常被作为深网络的基本特性来研究[8].

When we evaluate different cardinalities C while preserving complexity, we want to minimize the modification of other hyper-parameters. We choose to adjust the width of the bottleneck (e.g., 4-d in Fig 1(right)), because it can be isolated from the input and output of the block. This strategy introduces no change to other hyper-parameters (depth or input/output width of blocks), so is helpful for us to focus
on the impact of cardinality

​         当我们在保持复杂度的同时评估不同的基数C时，我们希望最小化对其它超参数的修改。我们选择调整bottleneck的宽度(例如图Figure 1(右)中的4维)，因为它可以从块的输入和输出隔离出来。这种策略不会改变其他超参数(block的深度或输入/输出宽度)，因此有助于我们聚焦基数的影响。

In Fig. 1(left), the original ResNet bottleneck block [14] has 256 · 64 + 3 · 3 · 64 · 64 + 64 · 256 ≈ 70k parameters and proportional FLOPs (on the same feature map size). With bottleneck width d, our template in Fig. 1(right) has: C · (256 · d + 3 · 3 · d · d + d · 256) (4) parameters and proportional FLOPs. When C = 32 and d = 4, Eqn.(4) ≈ 70k. Table 2 shows the relationship between cardinality C and bottleneck width d.

​        在图1(左),原ResNet bottleneck block[14] 有256*64 + 3 * 3 * 64  + 64 * 256≈70 k参数和成比例FLOPs(在同一特征图大小)。在bottleneck宽度为d的情况下，我们图Figure 1(右)的模板有:
$$
C*(256*d + 3*3*d*d + d*256) \tag 4
$$
参数和成比例FLOPs。当C = 32和d = 4 ,等式(4)约有70 k。表Table 2显示了基数C和bottleneck宽度d之间的关系。

Because we adopt the two rules in Sec. 3.1, the above approximate equality is valid between a ResNet bottleneck block and our ResNeXt on all stages (except for the subsampling layers where the feature maps size changes). Table 1 compares the original ResNet-50 and our ResNeXt-50 that is of similar capacity.5 We note that the complexity can only be preserved approximately, but the difference of the complexity is minor and does not bias our results.

​        因为我们采用了3.1小节中的两条规则，所以上面的近似等式在ResNet bottle block和我们的ResNeXt的所有阶段(除了下采样层,特征图的尺寸有变化)是有效的。



## 4. 实现细节

Our implementation follows [14] and the publicly available code of fb.resnet.torch [11]. On the ImageNet
dataset, the input image is 224×224 randomly cropped from a resized image using the scale and aspect ratio augmentation of [38] implemented by [11]. The shortcuts are identity connections except for those increasing dimensions which are projections (type B in [14]). Downsampling of conv3, 4, and 5 is done by stride-2 convolutions in the 3×3 layer of the first block in each stage, as suggested in [11]. We use SGD with a mini-batch size of 256 on 8 GPUs (32 per GPU). The weight decay is 0.0001 and the momentum is 0.9. We start from a learning rate of 0.1, and divide it by 10 for three times using the schedule in [11]. We adopt the
weight initialization of [13]. In all ablation comparisons, we evaluate the error on the single 224×224 center crop from an image whose shorter side is 256.

​        我们的实现遵循[14]和开源的fb.resnet.torch [11]代码。在ImageNet数据集上，输入图像是224×224，随机从调整尺寸后的图像裁剪得到，这些调整大小的图片来自使用由[11]实现的[38]中的用缩放和长宽比做数据增广。shotcuts是恒等连接，除了那些增加的维度是投影([14]中的类型B)。如[11]中建议，conv3 4和5的下采样是通过每个stage中第一block的步长为2的$3 \times 3$ 卷积层完成的；使用SGD, 256的mini-batch在8个GPUs上(每个GPU32个样本)；权重衰减为0.0001，动量大小为0.9。初始学习率为0.1，按照[11]中的计划做3次除以10;采用[13]中的权重初始化策略。在所有的消融比较中，我们在单个$224 \times 224 $ 的裁剪图像上进行错误评估，这个图像从短边为256的图像中心裁剪而来。

Our models are realized by the form of Fig. 3(c). We perform batch normalization (BN) [17] right after the convolutions in Fig. 3(c).6 ReLU is performed right after each BN, expect for the output of the block where ReLU is performed after the adding to the shortcut, following [14].

​         我们的模型以图Figure 3(c)的形式实现, 在图Figure 3(c)中卷积之后，我们执行批标准化(BN)[17]，在每个BN后执行ReLU，除了block的输出部分，遵循[14]:ReLU在添加到shortcut之后才执行。

We note that the three forms in Fig. 3 are strictly equivalent, when BN and ReLU are appropriately addressed as mentioned above. We have trained all three forms and obtained the same results. We choose to implement by Fig. 3(c) because it is more succinct and faster than the other two forms.

​          我们注意到，当BN和ReLU合适的放置，图Figure 3中的三种形式是严格等价的。我们训练了所有的三种形式，获得了相同的结果。我们选择图Figure . 3(c)实现,因为它更加简洁，并且比其它两种形式更快。



## 5. 实验

### 5.1. Experiments on ImageNet-1K



We conduct ablation experiments on the 1000-class ImageNet classification task [33]. We follow [14] to construct 50-layer and 101-layer residual networks. We simply replace all blocks in ResNet-50/101 with our blocks.

​          我们在1000类ImageNet分类任务[33]上进行消融实验，遵循[14],构建了50层和101层的残差网络，简单的将ResNet-50/101中所有的block替换为我们的block。

**Notations.** 

Because we adopt the two rules in Sec. 3.1, it is sufficient for us to refer to an architecture by the template.
For example, Table 1 shows a ResNeXt-50 constructed by a template with cardinality = 32 and bottleneck width = 4d (Fig. 3). This network is denoted as ResNeXt-50 (32×4d) for simplicity. We note that the input/output width of the template is fixed as 256-d (Fig. 3), and all widths are doubled each time when the feature map is subsampled (see Table 1).



**Cardinality vs. Width.** 

We first evaluate the trade-off between cardinality C and bottleneck width, under preserved complexity as listed in Table 2. Table 3 shows the results and Fig. 5 shows the curves of error vs. epochs. Comparing with ResNet-50 (Table 3 top and Fig. 5 left), the 32×4d ResNeXt-50 has a validation error of 22.2%, which is 1.7%
lower than the ResNet baseline’s 23.9%. With cardinality C increasing from 1 to 32 while keeping complexity, the error rate keeps reducing. Furthermore, the 32×4d ResNeXt also has a much lower training error than the ResNet counterpart, suggesting that the gains are not from regularization but from stronger representations.



Similar trends are observed in the case of ResNet-101
(Fig. 5 right, Table 3 bottom), where the 32×4d ResNeXt101
outperforms the ResNet-101 counterpart by 0.8%. Although
this improvement of validation error is smaller than
that of the 50-layer case, the improvement of training error
is still big (20% for ResNet-101 and 16% for 32×4d
ResNeXt-101, Fig. 5 right). In fact, more training data
will enlarge the gap of validation error, as we show on an
ImageNet-5K set in the next subsection.



Table 3 also suggests that with complexity preserved, increasing
cardinality at the price of reducing width starts
to show saturating accuracy when the bottleneck width is small. We argue that it is not worthwhile to keep reducing
width in such a trade-off. So we adopt a bottleneck width
no smaller than 4d in the following.