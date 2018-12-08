# Densely Connected Convolutional Networks

[TOC]



Recent work has shown that convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output. In this paper, we embrace this observation and introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections—one between each layer and its subsequent
layer—our network has L(L+1) 2 direct connections. For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters. We evaluate our proposed architecture on four highly competitive object recognition benchmark tasks (CIFAR-10, CIFAR-100, SVHN, and ImageNet). DenseNets obtain significant improvements over the state-of-the-art on most of them, whilst requiring less computation to achieve high performance. Code and pre-trained models are available at https://github.com/liuzhuang13/DenseNet

​           最近的研究表明，如果卷积网络在靠近输入层与输出层之间的地方有**更短的连接**（shorter connections），就可以训练更深、更准确、更有效的卷积网络。本文中，我们拥抱这个观点，介绍了稠密卷积网络（DenseNet），该网络在前馈时将每一层都与其他的任一层进行了连接。传统的$L$ 层卷积网络有$L$个连接——每一层与后一层有一个连接——我们的网络有$L(L+1)/2$个连接。每一层都将之前的所有层的特征图作为输入，而它自己的特征图是之后所有层的输入。DenseNets有一些显著优点：缓解梯度消失问题，加强特征传播，鼓励特征的重复利用，还大大减少参数量。我们在四个目标识别任务(CIFAR-10，CIFAR-100，SVHN和ImageNet）中评估了我们提出了结构。DenseNets在大部分数据集上相对state-of-the-art有明显提高，而且使用更少的计算量就可以获得高性能。代码和预训练模型可以在https://github.com/liuzhuang13/DenseNet上获得。

论文地址: https://arxiv.org/pdf/1608.06993.pdf



## 引言

Convolutional neural networks (CNNs) have become the dominant machine learning approach for visual object recognition. Although they were originally introduced over 20 years ago [18], improvements in computer hardware and network structure have enabled the training of truly deep CNNs only recently. The original LeNet5 [19] consisted of 5 layers, VGG featured 19 [29], and only last year Highway Networks [34] and Residual Networks (ResNets) [11] have surpassed the 100-layer barrier.

​        在视觉检测任务中，卷积神经网络（CNNs）已经成为占有绝对优势的机器学习方法。尽管它们在20年前就已经被提出来;直到最近，计算机硬件和网络结构的改善使的可以训练真正深的卷积网络。起初的LeNet5只有5层，VGG有19[29]层，只有去年的Highway网络[34]和ResNets网络[11]才克服了100层网络的障碍。

As CNNs become increasingly deep, a new research problem emerges: as information about the input or gradient passes through many layers, it can vanish and “wash out” by the time it reaches the end (or beginning) of the network. Many recent publications address this or related problems. ResNets [11] and Highway Networks [34] bypass signal from one layer to the next via identity connections. Stochastic depth [13] shortens ResNets by randomly dropping layers during training to allow better information and gradient flow. FractalNets [17] repeatedly combine several parallel layer sequences with different number of convolutional blocks to obtain a large nominal depth, while maintaining many short paths in the network. Although these different approaches vary in network topology and training procedure, they all share a key characteristic: they create short paths from early layers to later layers.

​        随着CNN网路的不断加深，新的研究问题出现了：输入信息或者梯度经过很多层，当它们传递到网络的尾端或开端时，这些信息可能已经消失或者被“冲掉”了。许多近期发布的成果都在解决这个问题或者相关问题。ResNets[11]和Highway Networks通[34]过恒等连接将信号从一个层绕道下一层。随机深度[13]方法通过在训练时随机丢弃层来缩短ResNet，使得信息和梯度能够在不同层之间更好的流动。FractalNets[17]用不同数量的卷积块重复地组合并行序列层，名义上能够获得更大的深度，同时在网络中维持许多短路径。尽管这些方法在网络拓扑和训练过程有所差异，但是它们都享有一个关键的特性：在前后层之间创建了短路径。

In this paper, we propose an architecture that distills this insight into a simple connectivity pattern: to ensure maximum information flow between layers in the network, we connect all layers (with matching feature-map sizes) directly with each other. To preserve the feed-forward nature, each layer obtains additional inputs from all preceding layers and passes on its own feature-maps to all subsequent layers. Figure 1 illustrates this layout schematically. Crucially, in contrast to ResNets, we never combine features
through summation before they are passed into a layer; instead, we combine features by concatenating them. Hence, the ` th layer has ` inputs, consisting of the feature-maps of all preceding convolutional blocks. Its own feature-maps are passed on to all L−` subsequent layers. This introduces L(L+1) 2 connections in an L-layer network, instead of just L, as in traditional architectures. Because of its dense connectivity pattern, we refer to our approach as Dense Convolutional Network (DenseNet).

在本文中，我们提出了一种将这种关键特性提炼为一个简单连接模式的架构：为了确保网络中各层之间的最大化信息流，我们将所有层（匹配的特征图大小）直接连接在一起。为了保持前馈性质，每个层从所有先前层中获得额外的输入，并将其自身的特征图传递给所有后续层。图1示意性地说明了这种布局。最重要的是，与ResNet相比，我们从不将特征通过求和合并后作为一层的输入，我们将特征串联成一个更长的特征。因此，第ℓ层有ℓ个输入，由所有先前卷积块的特征图组成。它自己的特征图传递给所有L−ℓ个后续层。这样，在L层网络中，有L(L+1)/2个连接，而不是传统架构中仅仅有L个连接。由于其密集的连接模式，我们将我们的方法称为密集连接卷积网络(DenseNet)。

A possibly counter-intuitive effect of this dense connectivity pattern is that it requires fewer parameters than traditional convolutional networks, as there is no need to relearn redundant feature-maps. Traditional feed-forward architectures can be viewed as algorithms with a state, which is passed on from layer to layer. Each layer reads the state from its preceding layer and writes to the subsequent layer. It changes the state but also passes on information that needs to be preserved. ResNets [11] make this information preservation explicit through additive identity transformations. Recent variations of ResNets [13] show that many layers contribute very little and can in fact be randomly dropped during training. This makes the state of ResNets similar to (unrolled) recurrent neural networks [21], but the number of parameters of ResNets is substantially larger because each layer has its own weights. Our proposed DenseNet architecture explicitly differentiates between information that is added to the network and information that is preserved. DenseNet layers are very narrow (e.g., 12 filters per layer), adding only a small set of feature-maps to the “collective knowledge” of the network and keep the remaining featuremaps unchanged—and the final classifier makes a decision based on all feature-maps in the network.

​        这种密集连接模式的可能的反直觉效应是，它比传统卷积网络需要的参数少，因为不需要重新学习冗余特征图。传统的前馈架构可以被看作是具有状态的算法，它是从一层传递到另一层的。每个层从上一层读取状态并写入后续层。它改变状态，但也传递需要保留的信息。 ResNet[11]通过附加的恒等转换使此信息保持明确。 ResNet[13]的最新变种表明，许多层次贡献很小，实际上可以在训练过程中随机丢弃。这使得ResNet的状态类似于（展开的）循环神经网络(recurrent neural network)，但是ResNet的参数数量太大，因为每个层都有自己的权重。我们提出的DenseNet架构明确区分添加到网络的信息和保留的信息。 DenseNet层非常窄（例如，每层12个卷积核），仅将一小组特征图添加到网络的“集体知识”，并保持剩余的特征图不变。最终分类器基于网络中的所有特征图。

Besides better parameter efficiency, one big advantage of DenseNets is their improved flow of information and gradients throughout the network, which makes them easy to train. Each layer has direct access to the gradients from the loss function and the original input signal, leading to an implicit deep supervision [20]. This helps training of deeper network architectures. Further, we also observe that dense connections have a regularizing effect, which reduces overfitting on tasks with smaller training set sizes.

​         除了有更好的，参数效率，DenseNet另一个大的优势在于能够提高网络中信息和梯度的传递和流动，这使得网络更易于训练。每一层都直接和原始输入信号以及来自损失函数的梯度连接，隐式的包含了一个深度监督[20]。这也有助于训练更深的网络结构。另外，我们还发现密集连接有一个正则化的效果，可以减轻小训练样本任务的过拟合现象。

We evaluate DenseNets on four highly competitive benchmark datasets (CIFAR-10, CIFAR-100, SVHN, and
ImageNet). Our models tend to require much fewer parameters than existing algorithms with comparable accuracy. Further, we significantly outperform the current state-ofthe-art results on most of the benchmark tasks.

​       我们在四个高竞争力的基准数据集(CIFAR-10，CIFAR-100，SVHN和ImageNet)上评估DenseNet。在精度相当的情况下，我们的模型往往需要比现有算法少得多的参数。此外，我们的模型在大多数基准任务中，精度明显优于其它最先进的方法。



## 相关工作

The exploration of network architectures has been a part of neural network research since their initial discovery. The recent resurgence in popularity of neural networks has also revived this research domain. The increasing number of layers in modern networks amplifies the differences between architectures and motivates the exploration of different connectivity patterns and the revisiting of old research ideas



A cascade structure similar to our proposed dense network layout has already been studied in the neural networks literature in the 1980s [3]. Their pioneering work focuses on fully connected multi-layer perceptrons trained in a layerby-layer fashion. More recently, fully connected cascade networks to be trained with batch gradient descent were proposed [40]. Although effective on small datasets, this approach only scales to networks with a few hundred parameters.

In [9, 23, 31, 41], utilizing multi-level features in CNNs through skip-connnections has been found to be
effective for various vision tasks. Parallel to our work, [1] derived a purely theoretical framework for networks with cross-layer connections similar to ours.



Highway Networks [34] were amongst the first architectures that provided a means to effectively train end-to-end networks with more than 100 layers. Using bypassing paths along with gating units, Highway Networks with hundreds of layers can be optimized without difficulty. The bypassing paths are presumed to be the key factor that eases the training of these very deep networks. This point is further supported by ResNets [11], in which pure identity mappings are used as bypassing paths. ResNets have achieved impressive, record-breaking performance on many challenging image recognition, localization, and detection tasks, such as ImageNet and COCO object detection [11]. Recently, stochastic depth was proposed as a way to successfully train a 1202-layer ResNet [13]. Stochastic depth improves the training of deep residual networks by dropping layers randomly during training. This shows that not all
layers may be needed and highlights that there is a great amount of redundancy in deep (residual) networks. Our paper was partly inspired by that observation. ResNets with pre-activation also facilitate the training of state-of-the-art networks with > 1000 layers [12].



An orthogonal approach to making networks deeper (e.g., with the help of skip connections) is to increase the network width. The GoogLeNet [36, 37] uses an “Inception module” which concatenates feature-maps produced by filters of different sizes. In [38], a variant of ResNets with wide generalized residual blocks was proposed. In fact, simply increasing the number of filters in each layer of ResNets can improve its performance provided the depth is sufficient [42]. FractalNets also achieve competitive results on several datasets using a wide network structure [17].



Instead of drawing representational power from extremely deep or wide architectures, DenseNets exploit the potential of the network through feature reuse, yielding condensed models that are easy to train and highly parameterefficient. Concatenating feature-maps learned by different layers increases variation in the input of subsequent layers and improves efficiency. This constitutes a major difference between DenseNets and ResNets. Compared to Inception networks [36, 37], which also concatenate features from different
layers, DenseNets are simpler and more efficient.



There are other notable network architecture innovations which have yielded competitive results. The Network in Network (NIN) [22] structure includes micro multi-layer perceptrons into the filters of convolutional layers to extract more complicated features. In Deeply Supervised Network (DSN) [20], internal layers are directly supervised by auxiliary classifiers, which can strengthen the gradients received by earlier layers. Ladder Networks [27, 25] introduce lateral connections into autoencoders, producing
impressive accuracies on semi-supervised learning tasks. In [39], Deeply-Fused Nets (DFNs) were proposed to improve information flow by combining intermediate layers of different base networks. The augmentation of networks with pathways that minimize reconstruction losses was also shown to improve image classification models [43].

