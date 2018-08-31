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

$$
\cal F(x)= \sum_{i=0}^C T_i(x), \tag 2
$$
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

​          因为我们采用了3.1节中的两条规则，通过模板就足以说明一个架构了(因为所有的block都是一个逻辑)。例如，表Table 1显示了由基数= 32和bottleneck宽度= 4d的模板构建的ResNeXt-50(图Figure 3),这个网络简化的记做ResNeXt-50 (32×4d)。我们注意到模板的输入/输出宽度固定为256-d(图Figure 3), 且每次对feature map进行下采样时，所有宽度都翻倍(见表Table 1)。

**基数 vs. 宽度.** 

We first evaluate the trade-off between cardinality C and bottleneck width, under preserved complexity as listed in Table 2. Table 3 shows the results and Fig. 5 shows the curves of error vs. epochs. Comparing with ResNet-50 (Table 3 top and Fig. 5 left), the 32×4d ResNeXt-50 has a validation error of 22.2%, which is 1.7%
lower than the ResNet baseline’s 23.9%. With cardinality C increasing from 1 to 32 while keeping complexity, the error rate keeps reducing. Furthermore, the 32×4d ResNeXt also has a much lower training error than the ResNet counterpart, suggesting that the gains are not from regularization but from stronger representations.

​         如表Table 2所示，我们首先在的保持复杂度的情况下，评估基数C和bottleneck宽度之间的权衡，表Table 3 展示了结果，图Figure 5展示了error vs. epochs的曲线。与ResNet-50对比(表Table 3顶部和图Figure 5左侧),32×4d ResNeXt-50验证误差为22.2%,比ResNet基线的23.9%低1.7%。保持复杂度，随着基数C从1增加到32，错误率不断降低。此外,32×4 d ResNeXt也比ResNet同行低得多的训练误差,说明性能提升不是来自正则化,而是来源于更强的表达能力。



Similar trends are observed in the case of ResNet-101 (Fig. 5 right, Table 3 bottom), where the 32×4d ResNeXt101 outperforms the ResNet-101 counterpart by 0.8%. Although this improvement of validation error is smaller than that of the 50-layer case, the improvement of training error is still big (20% for ResNet-101 and 16% for 32×4d ResNeXt-101, Fig. 5 right). In fact, more training data will enlarge the gap of validation error, as we show on an ImageNet-5K set in the next subsection.

​          在resnet - 101(图Figure5 右边,表Table 3底部)下也观察到类似的趋势,其中32×4d ResNeXt101优于resnet- 101 0.8%。虽然这种验证误差的提升比50层的情况要小，但是训练误差的提升任然是很大的(resnet-101 20%,32×4d resnext-101 16%,图Figure 5右边)。事实上，更多的训练数据将扩大验证误差的差距，如我们在下一小节中的ImageNet-5K集中所展示的那样。



Table 3 also suggests that with complexity preserved, increasing cardinality at the price of reducing width starts to show saturating accuracy when the bottleneck width is small. We argue that it is not worthwhile to keep reducing width in such a trade-off. So we adopt a bottleneck width no smaller than 4d in the following.

​       表3还表明，保持复杂度，当bottleneck宽度很小时，以减少宽度为代价的增加基数开始显示精度饱和(不再提升)。我们认为，在这种权衡中继续减少宽度是不值得的，因此，我们在下面采用一个不小于4d的bottleneck宽度。



**增加基数 vs. 更深/更宽.** 

Next we investigate increasing complexity by increasing cardinality C or increasing depth or width. The following comparison can also be viewed as with reference to 2× FLOPs of the ResNet-101 baseline. We compare the following variants that have ∼15 billion FLOPs. (i) Going deeper to 200 layers. We adopt the ResNet-200 [15] implemented in [11]. (ii) Going wider by increasing the bottleneck width. (iii) Increasing cardinality by doubling C.

​        接下来，我们研究通过增加基数C或增加深度或宽度来增加复杂度。下面的比较也可以被视为关于2倍FLOPS resnet-101的基线; 我们比较有∼150亿FLOPs的如下几个变种。(i)深度加到200层,我们采用了在[11]中实现的ResNet-200[15]。(ii)通过增加bottleneck宽度来变宽。(iii)通过将C加倍来增加基数。



Table 4 shows that increasing complexity by 2× consistently reduces error vs. the ResNet-101 baseline (22.0%). But the improvement is small when going deeper (ResNet200, by 0.3%) or wider (wider ResNet-101, by 0.7%)

​          表Table 4表明相对ResNet-101基线误差(22.%)，复杂度加倍误差会一致减少;但是更深(ResNet200,提升0.3%)或更宽(更宽的ResNet-101,提升0.7%)时，提升很小。

On the contrary, increasing cardinality C shows much better results than going deeper or wider. The 2×64d
ResNeXt-101 (i.e., doubling C on 1×64d ResNet-101 baseline and keeping the width) reduces the top-1 error by 1.3% to 20.7%. The 64×4d ResNeXt-101 (i.e., doubling C on 32×4d ResNeXt-101 and keeping the width) reduces the top-1 error to 20.4%

​         相反，增加基数C比更深或更宽结果要好得多，2×64d ResNeXt-101将top-1误差减少1.3%到达20.7%; The 64×4d ResNeXt-101误差减小到20.4%。

We also note that 32×4d ResNet-101 (21.2%) performs better than the deeper ResNet-200 and the wider ResNet101, even though it has only ∼50% complexity. This again shows that cardinality is a more effective dimension than the dimensions of depth and width.

​         同样注意到32×4d ResNet-101 (21.2%)比更深的ResNet-200 和更宽的ResNet101表现更好，虽然只有它们50%的复杂度。这再次表明，基数比深度和宽度更有效的维度。

**残差连接**

The following table shows the effects of the residual (shortcut) connections:Removing shortcuts from the ResNeXt-50 increases the error by 3.9 points to 26.1%. Removing shortcuts from its ResNet-50 counterpart is much worse (31.2%). These comparisons suggest that the residual connections are helpful for optimization, whereas aggregated transformations are stronger representations, as shown by the fact that they perform consistently better than their counterparts with or without residual connections.

​         表显示了残差(shortcut)连接的影响:从ResNeXt-50删除shortcut将错误增加3.9个点，至26.1%。删除ResNet-50的shotcut要糟糕得多(31.2%)。这些比较表明，残差连接有助于优化，而聚合转换是更强的表示，因为它们的性能始终优于有或没有残差连接的对应的副本。

**性能**

For simplicity we use Torch’s built-in grouped convolution implementation, without special optimization. We note that this implementation was brute-force and not parallelization-friendly. On 8 GPUs of NVIDIA M40, training 32×4d ResNeXt-101 in Table 3 takes 0.95s per mini-batch, vs. 0.70s of ResNet-101 baseline that has
similar FLOPs. We argue that this is a reasonable overhead. We expect carefully engineered lower-level implementation (e.g., in CUDA) will reduce this overhead. We also expect that the inference time on CPUs will present less overhead. Training the 2×complexity model (64×4d ResNeXt-101) takes 1.7s per mini-batch and 10 days total on 8 GPUs.

​          为了简单起见，我们使用Torch的内置分组卷积实现，没有特别的优化。我们注意到这个实现是暴力的(brute-force)，不支持并行。在8块 NVIDIA M40 GPU上训练表Table 3所示的32×4d ResNeXt-101 每个mini-batch耗时0.95秒，对比ResNet-101 baseline耗时0.70s，FLOPs相当；我们认为这是合理的开销。我们认为经过精心设计的较低级别的实现(例如 CUDA)能够减少这种开销。我们还认为cpu上的推断时间会减少开销。训练2倍复杂度模型(64×4 d resnext - 101)需要1.7秒/ mini-batch和一共10天在8块GPU上。

**与state-of-the-art结果比较**

Table 5 shows more results of single-crop testing on the ImageNet validation set. In addition to testing a 224×224 crop, we also evaluate a 320×320 crop following [15]. Our results compare favorably with ResNet, Inception-v3/v4, and Inception-ResNet-v2, achieving a single-crop top-5 error rate of 4.4%. In addition, our architecture design is much simpler than all Inception models, and requires considerably fewer hyper-parameters to be set by hand.

​          表Table 5显示了在ImageNet验证集上进行单裁剪测试的更多结果。除了测试一个224×224裁剪,我们也遵循[15]评估320×320裁剪。我们的结果优于ResNet、Inception-v3/v4和Inception-ResNet-v2，到达单裁剪top-5个错误率4.4%。此外，我们的架构设计比所有的Inception模型都要简单得多，并且需要手工设置的超参数要少得多。

ResNeXt is the foundation of our entries to the ILSVRC 2016 classification task, in which we achieved 2nd place. We note that many models (including ours) start to get saturated on this dataset after using multi-scale and/or multicrop testing. We had a single-model top-1/top-5 error rates of 17.7%/3.7% using the multi-scale dense testing in [14], on par with Inception-ResNet-v2’s single-model results of 17.8%/3.7% that adopts multi-scale, multi-crop testing. We had an ensemble result of 3.03% top-5 error on the test set,on par with the winner’s 2.99% and Inception-v4/InceptionResNet-v2’s 3.08% [37].

​         ResNeXt是我们参加ILSVRC 2016分类任务的基础，我们获得了第二名。  我们注意到，许多模型(包括我们的)在使用多尺度和/或多裁剪测试后开始饱和。使用[14]中的多尺度密集测试，我们得到了一个单模型的top-1/top-5错误率为17.7%/3.7%，与采用多尺度、多裁剪测试的inception-resnet-v2的单模型结果为17.8%/3.7%。在测试集中，我们的集成模型top-5得错误率为3.03%，与获胜者的2.99%和Inception-v4/InceptionResNet-v2的3.08%[37]相当。

### 5.2. ImageNet-5K实验

The performance on ImageNet-1K appears to saturate.But we argue that this is not because of the capability of the models but because of the complexity of the dataset. Next we evaluate our models on a larger ImageNet subset that has 5000 categories.

​          ImageNet-1K上的性能似乎饱和了，但我们认为，这不是因为模型的能力，而是因为数据集的复杂度。接下来，我们在一个较大的ImageNet子集上评估模型，这个子集有5000个类别。

Our 5K dataset is a subset of the full ImageNet-22K set [33]. The 5000 categories consist of the original ImageNet1K categories and additional 4000 categories that have the largest number of images in the full ImageNet set. The 5K set has 6.8 million images, about 5× of the 1K set. There is no official train/val split available, so we opt to evaluate on the original ImageNet-1K validation set. On this 1K-class val set, the models can be evaluated as a 5K-way classification task (all labels predicted to be the other 4K classes are
automatically erroneous) or as a 1K-way classification task (softmax is applied only on the 1K classes) at test time

​        我们的5K数据集是完整的ImageNet-22K集[33]的一个子集，这5000个类别包括原始的ImageNet1K中的类别和在完整的ImageNet数据集中具有最多图像数量的额外4000个类别。5k数据集有680万图片,大约5倍于1k数据集。没有官方可用的train/val分割，所以我们选择在原始的ImageNet-1K验证集上评估。在这个1k类val集上，预测时，模型可以被当做一个5k类分类任务(所有预测为其他4K类的标签都自动认为错误);或一个1K类分类任务(softmax只应用于1K类)来评估



The implementation details are the same as in Sec. 4. The 5K-training models are all trained from scratch, and are trained for the same number of mini-batches as the 1Ktraining models (so 1/5× epochs). Table 6 and Fig. 6 show the comparisons under preserved complexity. ResNeXt-50 reduces the 5K-way top-1 error by 3.2% comparing with ResNet-50, and ResNetXt-101 reduces the 5K-way top-1 error by 2.3% comparing with ResNet-101. Similar gaps are observed on the 1K-way error. These demonstrate the stronger representational power of ResNeXt.

​          实现细节与节4一样，5K训练模型从头开始训练，跟1K 训练模型一样的mini-batch数(这样epochs就只有1/5)。表Table 6和图Figure 6展示了在保持复杂度情况下的比较。ResNeXt-50比ResNet-50减少5K类 top-1错误3.2%，ResNetXt-101比ResNet-101减少5K类 top-1误差2.3%，在1k类误差上也观察到类似的差距，这些表明ResNeXt具有更强的表示能力。

Moreover, we find that the models trained on the 5K set (with 1K-way error 22.2%/5.7% in Table 6) perform
competitively comparing with those trained on the 1K set (21.2%/5.6% in Table 3), evaluated on the same 1K-way classification task on the validation set. This result is achieved without increasing the training time (due to the same number of mini-batches) and without fine-tuning. We argue that this is a promising result, given that the training task of classifying 5K categories is a more challenging one.

​         此外，在验证集上评估相同的1K类分类任务，我们发现在5K集上训练的模型(在表Table 6中有1K类error 22.2%/5.7%)与在1K集(表Table 3中为21.2%/5.6%)上训练相比，表现出竞争力。此结果并没有增加训练时间(由于相同的mini-batch数量)，也没有微调。我们认为这是一个有希望的结果，因为5K类的分类训练任务是一个更具挑战性的任务。



### 5.3. CIFAR实验

datasets [23]. We use the architectures as in [14] and replace the basic residual block by the bottleneck template of $\left[1 \times1, 64 \\ 3 \times3,64 \\ 1 \times 1, 256  \right]$  Our networks start with a single 3×3 conv

layer, followed by 3 stages each having 3 residual blocks, and end with average pooling and a fully-connected classifier (total 29-layer deep), following [14]. We adopt the same translation and flipping data augmentation as [14]. Implementation details are in the appendix.



We compare two cases of increasing complexity based on the above baseline: (i) increase cardinality and fix all widths, or (ii) increase width of the bottleneck and fix cardinality = 1. We train and evaluate a series of networks under these changes. Fig. 7 shows the comparisons of test error rates vs. model sizes. We find that increasing cardinality is more effective than increasing width, consistent to what we have observed on ImageNet-1K. Table 7 shows the results and model sizes, comparing with the Wide ResNet [43] which is the best published record. Our model with a similar model size (34.4M) shows results better than Wide ResNet. Our larger method achieves 3.58% test error (average of 10 runs) on CIFAR-10 and 17.31% on CIFAR-100.
To the best of our knowledge, these are the state-of-the-art results (with similar data augmentation) in the literature including unpublished technical reports.



### 5.4. COCO目标检测实验

​         Next we evaluate the generalizability on the COCO object
detection set [27]. We train the models on the 80k training
set plus a 35k val subset and evaluate on a 5k val subset
(called minival), following [1]. We evaluate the COCOstyle
Average Precision (AP) as well as AP@IoU=0.5 [27].
We adopt the basic Faster R-CNN [32] and follow [14] to
plug ResNet/ResNeXt into it. The models are pre-trained
on ImageNet-1K and fine-tuned on the detection set. Implementation
details are in the appendix.



Table 8 shows the comparisons. On the 50-layer baseline,
ResNeXt improves AP@0.5 by 2.1% and AP by 1.0%,
without increasing complexity. ResNeXt shows smaller improvements
on the 101-layer baseline. We conjecture that
more training data will lead to a larger gap, as observed on
the ImageNet-5K set.
It is also worth noting that recently ResNeXt has been
adopted in Mask R-CNN [12] that achieves state-of-the-art
results on COCO instance segmentation and object detection
tasks.



## A. Implementation Details: CIFAR





## B. Implementation Details: Object Detection



