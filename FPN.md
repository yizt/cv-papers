# Feature Pyramid Networks for Object Detection

[TOC]

## 摘要

​        **特征金字塔**是识别系统中用于检测**不同尺寸**目标的基本组件。但最近的深度学习目标检测器已经避免了金字塔表示，部分因为它们是**计算和内存** **密集型**的。在本文中，我们利用深度卷积网络**内在的多尺寸**、**金字塔分层**来构造具有很少额外成本的特征金字塔。开发了一种具有**侧向连接**的**自顶向下**(top-down)架构，用于在**所有尺寸**上构建**高级语义特征图**。这种架构,称为特征金字塔网络(FPN); 在几个应用程序中作为**通用特征提取器**表现出了显著的提升。在一个基本的Faster R-CNN系统中使用FPN，没有花里胡哨，我们的方法可以在COCO检测基准数据集上取得state-of-the-art单模型结果，**超过了所有现有的单模型记录**，包括COCO 2016挑战赛的获奖者。此外，我们的方法可以在**单GPU**上以**6FPS**运行，因此是**多尺寸目标检测**的实用和准确的解决方案。代码将公开发布。

论文地址： <https://arxiv.org/pdf/1612.03144.pdf>



## 1. 引言

![](pic/FPN-fig1.png)

​       识别不同尺寸的目标是计算机视觉中的一个基本挑战。建立在**图像金字塔**之上的特征金字塔(我们简称为**特征化图像金字塔**-featurized image pyramids)构成了标准解决方案的基础[1] (图Figure 1(a))。这些金字塔是尺寸不变的，因而**目标的尺寸变化**是通过**在金字塔中移动它的层级**来弥补的。直观地说，该属性使模型能够通过在**位置**和**金字塔层级**上扫描,来检测**很大范围尺寸**的目标。


​        特征化图像金字塔在手工设计的时代被大量使用[5，25]。它们非常关键，以至于像DPM[7]这样的目标检测器需要**密集的尺寸采样**才能获得好的结果(例如每组10个尺寸)。对于识别任务，工程特征大部分已经被深度卷积网络(ConvNets)[19，20]计算的特征所取代。除了能够表示更高级别的语义，ConvNets对于尺寸变化也更加鲁棒，从而有助于从**单一输入尺寸上计算的特征**进行识别[15,11,29] (图Figure 1(b))。但即使有这种鲁棒性，**获取最准确的结果**，仍**需要金字塔**。在ImageNet[33]和COCO[21]检测挑战中，最近的所有排名靠前的记录，都使用了**特征化图像金字塔** 上的 **多尺寸测试**(例如[16,35])。对图像金字塔的每个层次进行特征化的主要优势在于它产生了**多尺寸的特征表示**，其中**所有层上的语义都很强**，包括高分辨率层。

​       尽管如此，**特征化**图像金字塔的**每个层**有**明显的局限性**。**预测时间显著增加**(例如，四倍[11])，使得这种方法在实际应用中不可行。此外，在图像金字塔上端到端地训练深度网络在**内存**而言是**不可行的**，所以如果采用该方法，图像金字塔**仅在测试时使用**[15，11，16，35]，这造成了**训练/测试**时预测的**不一致性**。出于这些原因，Fast和Faster R-CNN[11，29]选择**默认不使用特征化图像金字塔**。

​        但是，图像金字塔并不是计算多尺寸特征表示的唯一方法。深层ConvNet逐层计算特征层级，**通过下采样层**，特征层级具有**内在的多尺寸** **金字塔形状**。这种网内特征层级产生不同空间分辨率的特征图，但引入了由不同深度引起的较大的**语义差异**。高分辨率特征图有**低级特征**，**影响了其目标识别表示能力**。

​       Single Shot Detector(SSD)[22]是首先尝试使用ConvNet的金字塔特征层级中的一个，就像它是一个特征化的图像金字塔(图Figure 1(c))。理想情况下，SSD类的金字塔将**重用**前向传播中计算的**不同层中的多尺寸特征图**，因此是**零成本**的。但为了避免使用低级特征，SSD放弃重用已经计算好的特征图，而从网络的**高层开始构建金字塔**(例如，VGG网络的conv4_3[36])，然后**添加几个新层**。因此它**错过了重用** 特征层级中**高分辨率特征图**的机会。我们证明这对于**检测小目标很重要**。

​        本文的目标是**自然地利用**ConvNet特征层级的金字塔形状，同时创建一个在**所有尺寸上**都具有**强大语义信息**的特征金字塔。为了实现这个目标，我们依赖于一种结构，它将**低分辨率、强语义**的特征 和 **高分辨率、弱语义**的特征 通过**自顶向下**的路径和**侧向连接** 组合在一起(图Figure 1(d))。其结果是一个特征金字塔，在**所有层级**都具**有丰富的语义信息**，并且可以从**单尺寸的输入图像**上进行快速构建。换句话说，我们展示了如何创建**网络内**(in-network)的**特征金字塔**，可以用来代替 **特征化图像金字塔**，而**不牺牲表示能力，速度或内存**。

![fig2](pic/FPN-fig2.png)

​        最近的研究[28，17，8，26]中流行采用**自顶向下**和跳跃连接(skip connetion)的类似架构。他们的目标是生成**单个**有**高级特征**的**高分辨率特征图**，并在其上进行预测(图Figure 2顶部)。相反，我们的方法利用这个架构作为特征金字塔，其中预测(例如目标检测)在每个级别上独立进行(图2底部)。我们的模型反映了一个特征化的图像金字塔，这在这些研究中还没有探索过。

​        我们评估了我们的方法:称为特征金字塔网络(FPN)，其在各种用于检测和分割[11，29，27]系统中。没有任何花里胡哨，我们在具有挑战性的COCO检测基准数据集上报告了最新的单模型结果，仅仅基于FPN和基本的Faster R-CNN检测器[29]，就超过了所有现存的严重工程化的单模型的**竞赛获奖者的记录**。在消融实验中，我们发现对于**提议边框**，FPN将**平均召回率**(AR)显著的提升了**8个百分点**；对于目标检测，它将COCO类的**精度**(AP)提高了2.3个百分点，PASCAL类AP提高了3.8个百分点，超过了较强的**单尺寸模型**基线: 基于ResNet[16]的Faster R-CNN。我们的方法也很容易扩展到掩码提议框(mask proposals)，改进实例分隔AR, 并且速度超过 严重依赖图像金字塔的state-of-the-art方法。

另外，我们的金字塔结构可以通过所有尺寸进行端到端培训，并且在训练/测试时一致地使用，这在使用图像金字塔时是内存不可行的。因此，FPN能够比所有现有的state-of-the-art方法获得更高的准确度。此外，这种改进是在不增加单尺寸基准测试时间的情况下实现的。我们相信这些进展将有助于未来的研究和应用。我们的代码将公开发布。



## 2. 相关工作

### 手工设计特征和早期神经网络

​        SIFT特征[25]最初是从尺寸空间极值中提取的，用于特征点匹配。HOG特征[5]，以及后来的SIFT特征，都是在整个图像金字塔上密集计算的。这些HOG和SIFT金字塔已在许多工作中得到了应用，用于图像分类，目标检测，人体姿势估计等。这对快速计算特征化图像金字塔也很有意义。Dollar等人[6]通过先计算**一个稀疏采样(尺寸)金字塔**，然后插入缺失的层级，演示了快速金字塔计算。在HOG和SIFT之前，使用ConvNet[38，32]的早期人脸检测工作计算了**图像金字塔**上的**浅层网络**，以检测跨尺寸的人脸。

### Deep ConvNet目标检测器

​         随着现代深度卷积网络[19]的发展，像OverFeat[34]和R-CNN[12]这样的目标检测器在精度上显示出了显著的提高。OverFeat采用了一种类似于早期神经网络人脸检测器的策略，通过在图像金字塔上应用ConvNet作为滑动窗口检测器。R-CNN采用了基于区域提议的策略[37]，其中每个提议在用ConvNet进行分类之前都进行了尺寸归一化。SPPnet[15]表明，这种基于区域的检测器可以更有效地应用于在单个图像尺寸上提取的特征图。最近更准确的检测方法，如Fast R-CNN[11]和Faster R-CNN[29]提倡使用从单一尺寸计算出的特征，因为它提供了精确度和速度之间的良好折衷。然而，多尺寸检测性能仍然更好，特别是对于小型目标。

### 使用多层特征的方法

​        一些最近的方法通过使用ConvNet中的不同层来改进检测和分割。FCN[24]将多个尺寸上的每个类别的部分分数相加以计算语义分割。Hypercolumns[13]使用类似的方法进行目标实例分割。在计算预测之前，其他几种方法(HyperNet[18]，ParseNet[23]和ION[2])将多个层的特征连接起来，这相当于累加转换后的特征。SSD[22]和MS-CNN[3]可预测特征层级中多个层的目标，而不需要组合特征或评分。

​       最近有一些方法利用**横向/跳跃连接**将跨**分辨率**和**语义层次**的低级特征图关联起来，包括用于分割的U-Net[31]和SharpMask[28]，Recombinator网络[17]用于人脸检测以及Stacked Hourglass网络[26]用于关键点估计。Ghiasi等人[8]为FCN提出拉普拉斯金字塔表示，以逐步细化分割。尽管这些方法采用的是**金字塔形状**的架构，但它们**不同于** **特征化图像金字塔**[5，7，34]，其中所有层次上的预测都是独立进行的，参见图Figure 2。事实上，对于图Figure 2(顶部)中的金字塔结构，要识别**多个尺寸的目标**[28], **任需要图像金字塔**。

## 3. 特征金字塔网络

​       我们的目标是利用ConvNet的金字塔特征层级，该层次结构具有**从低到高**的**语义信息**，并在整个过程中构建**含有高级语义**的**特征金字塔**。由此产生的**特征金字塔网络是通用**的，在本文中，我们聚焦**滑动窗口提议**(Region Proposal Network，简称RPN)[29]和**基于区域的检测器**(Fast R-CNN)[11]。在第6节中我们还将FPN泛化到实例分割提议。

​        我们的方法以任意大小的**单尺寸图像**作为输入，并以全卷积的方式输出**多层比例大小**的**特征图**。这个过程独立于主干卷积结构(例如[19，36，16])，在本文中，我们呈现了使用ResNets[16]的结果。如下所述，我们的金字塔结构包括自底向上的路径，自上而下的路径和侧向连接。

### 自底向上的路径

​        自底向上的路径是主ConvNet的前馈计算，其计算**由缩放步长为2**的**多尺寸特征图**组成的**特征层级**。通常有很多层 产生**相同大小**的 **输出特征图**，并且我们认为这些层位于相同的网络**阶段**(stage)。对于我们的特征金字塔，我们为**每个阶段定义一个金字塔层**。我们选择每个阶段的最后一层的输出作为我们的特征图参考集合，我们将丰富它来创建我们的金字塔。这种选择是自然的，因为每个阶段的**最深层应具有最强的特征**。

​       具体而言，对于ResNets[16]，我们使用每个阶段的最后一个残差块激活输出的特征。对于conv2，conv3，conv4和conv5输出，我们将这些最后残差块的输出表示为{C2,C3,C4,C5}，注意它们相对于输入图像的步长为{4，8，16，32}个像素。由于其庞大的内存占用，我们**没有将conv1纳入金字塔。**

![](pic/FPN-fig3.png)

### 自顶向下的路径和侧向连接

​        自顶向下的路径通过**上采样** **空间上更粗糙**但在**语义上更强**的来自**较高金字塔层级的特征图**来幻化**更高分辨率的特征**。这些特征随后通过**侧向连接** **自底向上路径上的特征** 进行**增强**。每个侧向连接合并来自**自底向上**路径和**自顶向下**路径的具有**相同空间大小的特征图**。自底向上的特征图具有**较低级别的语义信息**，但其的激活可以**更精确地定位**，因为它被下**采样的次数更少**。

​        图Figure 3显示了建造我们的自顶向下特征图的构建块。对于较粗糙分辨率的特征图，我们**以因子2上采样**空间分辨率(为了简单起见，使用最近邻上采样)。然后通过逐个元素相加，将**上采样特征**与相应的**自底向上特征**(其经过1×1卷积层来减少通道维度)合并。迭代这个过程，直到生成**最佳分辨率特征图**。开始迭代时，我们只需在C5上添加一个**1×1卷积层**来生成最粗糙分辨率特征图。最后，我们在**每个合并后的特征**上添加一个3×3卷积来生成最终的特征图，这是为了**减少上采样的混叠效应**。这个最终的特征图集合称为{P2,P3,P4,P5}，对应于{C2,C3,C4,C5}，依次具有**相同的空间大小**。

​        由于**金字塔的所有层**都像传统的特征图像金字塔一样使用**共享分类器/回归器**，因此我们在所有特征图中**固定特征维度**(通道数记为dd)。我们在本文中设置d=256，因此所有额外的卷积层都有256个通道的输出。在这些额外的层中**没有非线性变化**，我们在实验中发现这影响很小。

​        简洁性是我们设计的核心，我们发现我们的模型对许多设计选择都**很鲁棒**。我们已经尝试了更复杂的块(例如，使用多层残差块[16]作为连接)并观察到稍好的结果。**设计更好的连接模块并不是本文的重点**，所以我们选择上述的简单设计。



## 4. 应用

​        我们的方法是在**深度ConvNets内部** **构建特征金字塔**的通用解决方案。接下来，我们采用我们的方法在**RPN**[29]中进行**提议边框生成**，并在**Fast R-CNN**[11]中进行**目标检测**。为了证明我们方法的**简洁性**和**有效性**，我们对[29,11]的原始系统进行最小修改，使其适应我们的特征金字塔。

### 4.1. RPN的特征金字塔网络

​        RPN[29]是一个**类别无关**的**滑动窗口目标检测器**。在原始的RPN设计中，在**单尺寸卷积特征图**上，使用一个小型子网络在密集的3×3滑动窗口上进行评估，执行**目标/非目标的二分类**和**边框回归**。这是通过**一个3×3的卷积层**实现的，后面跟着**两个**用于分类和回归的**1×1兄弟卷积**，我们称之为**网络头部**(network-head)。**目标/非目标标准**和**边框回归目标**的根据一组称为**锚点**(anchors)的参考框的[29]定义的。这些锚点具有**多个**预定义的**尺寸和长宽比**，以覆盖不同形状的目标。

​        我们通过用我们的FPN替换单尺寸特征图来适应RPN。我们在我们的特征金字塔的**每个层级上附加一个相同设计的** **头部**(3x3 conv和两个1x1兄弟convs)。由于头部在金字塔**所有层级**的**所有位置** 上密集滑动，所以**不需要**在特定层级上具有**多尺寸锚点**。相反，我们为**每个层级分配单尺寸的锚点**。在形式上，我们定义锚点{$P_2,P_3,P_4,P_5,P_6$}分别具有$\{32^2,64^2,128^2,256^2,512^2\}$像素的面积。正如在[29]中，我们在每个层级上也使用了**多个长宽比**{1:2,1:1,2:1}的锚点。所以在金字塔上总共有**15个锚点**。

​       如[29]，我们根据锚点和ground-truth边框的交并比(IoU)比例将训练标签分配给锚点。形式上，如果一个锚点对于一个给定的ground-truth边框具有**最高的IoU**或者与任何ground-truth边框的**IoU超过0.7**，则给其分配一个**正标签**，如果其**与所有ground-truth边框的IoU都低于0.3**，则为其分配一个**负标签**。请注意，**ground-truth边框的尺寸**并未**显式的用于分配**它们到金字塔的**哪个层级**；相反，**ground-truth边框**与已经分配给金字塔层级的**锚点相关联**。因此，除了[29]中的内容外，我们不引入额外的规则。

​        我们注意到**头部的参数**在所有特征金字塔层级上**共享**；我们也评估了替代方案，没有共享参数并且观察到类似的准确性。**共享参数的良好性能**表明我们的金字塔的所有层级**共享相似的语义级别**。这个优点类似于使用特征图像金字塔的优点，可以将**通用的头部分类器**应用于**任意图像尺寸计算的特征**。

​        通过上述调整，RPN可以自然地通过我们的FPN进行训练和测试，与[29]中的方式相同。我们在实验中详细说明实施细节。

### 4.2. Fast R-CNN的特征金字塔网络

​        Fast R-CNN[11]是一个基于区域的目标检测器，利用感兴趣区域(RoI)池化来提取特征。Fast R-CNN通常在单尺寸特征图上执行。要将其与我们的FPN一起使用，我们需要为金字塔层级**分配不同尺寸的RoI**。

​        我们将我们的特征金字塔看作是从图像金字塔生成的。因此，当它们在图像金字塔上运行时，我们可以适应基于区域的检测器的分配策略[15，11]。在形式上，我们通过以下公式将宽度为w和高度为h(在网络上的输入图像上)的RoI分配到特征金字塔的级别$P_k$上：
$$
k=⌊k0+log_2( \sqrt{wh}/224)⌋. \tag 1
$$
​        这里224是规范的ImageNet预训练大小，而$k_0$是大小为$w×h=224^2$的RoI应该映射到的目标级别。类似于基于ResNet的Faster R-CNN系统[16]使用$C_4$作为单尺寸特征图，我们将$k_0$ 设置为4。直觉上，公式(1)意味着如果RoI的尺寸变小了(比如224的1/2)，它应该被映射到一个更精细的分辨率级别(比如k=3)。

​        我们在所有级别的所有RoI中附加**预测器头部**(在Fast R-CNN中，预测器头部是**类别相关的分类器和边框回归器**)。再次，**预测器头部都共享参数**，不管他们在什么层级。在[16]中，ResNet的**conv5层**(9层深的子网络)被**用作conv4特征之上的头部**，但我们的方法已经利用了**conv5来构建特征金字塔**。因此，与[16]不同，我们只是采用RoI池化提取7×7特征，并在最终的分类层和边框回归层之前附加**两个隐藏单元为1024维的全连接(fc)层**(每层后都接ReLU层)。这些层是随机初始化的，因为ResNets中没有预先训练好的fc层。请注意，**与标准的conv5头部相比，我们的2-fc MLP头部更轻更快**。

​        基于这些改编，我们可以在特征金字塔之上训练和测试Fast R-CNN。实现细节在实验部分给出。



## 5. 目标检测实验

​        我们在80类的COCO检测数据集[21]上进行实验。我们**训练**使用80k张训练图像和35k大小的验证图像的子集(`trainval35k`[2])的联合，并报告了在5k大小的验证图像子集(`minival`)上的消融实验。我们还报告了在没有公开标签的标准测试集(`test-std`)[21]上的最终结果。

​       正如通常的做法[12]，所有的网络骨干都是在ImageNet1k分类集[33]上预先训练好的，然后在检测数据集上进行微调。我们使用公开可用的预训练的ResNet-50和ResNet-101模型。我们的代码是使用Caffe2重新实现`py-faster-rcnn`。

### 5.1. RPN和Region Proposals

​       根据[21]中的定义，我们评估了COCO类型的平均召回率(AR)和在小型，中型和大型目标($AR_s, AR_m,  AR_l$)上的AR。我们报告了每张图像使用100个提议和1000个提议的结果($AR^{100} , AR^{1k}$)。

**实施细节**

​       表Table 1中的所有架构都是端到端训练。输入图像的大小调整为其**较短边800像素**。我们采用8个GPU进行同步SGD训练。一个mini-batch包括**每个GPU**上**2张图像**和**每张图像256个锚点**。我们使用0.0001的权重衰减和0.9的动量。前30k次mini-batch数据的学习率为0.02，而下一个10k次的学习率为0.002。对于所有的RPN实验(包括基准数据集)，我们都包含了图像外部的锚点(**anchor边框超出图像边框的**)来进行训练；不同于[29]，忽略这些锚点。其它实现细节如[29]中所述。使用**带FPN的RPN**在8个GPU上训练COCO数据集需要约8小时。

![](pic/FPN-tab1.png)



#### 5.1.1 消融实验

**与基线进行比较**

​        为了与原始RPNs[29]进行公平比较，我们使用$C_4$(与[16]相同)或$C_5$的**单尺寸特征图** 运行了两个基线(表1(a，b))，都使用与我们相同的超参数，包括使用5种尺寸锚点$\{32^2, 64^2, 128^2, 256^2, 512^2 \}$。表Table 1(b)显示没有优于(a)，这表明**单个更高级别的特征图是不够的**，因为存在在**较粗分辨率**和**较强语义**之间的**权衡**。

​        将FPN放在RPN中可将$AR^{1k}$提高到56.3(表Table 1(c))，这比单尺寸RPN基线(表1(a))增加了**8.0**个点。此外，在**小型目标**($AR^{1k}_s$)上的性能也**大幅上涨了12.9个点**。我们的**金字塔表示** 极大提升了RPN对**目标尺寸变化的鲁棒性**。



**自顶向下改进的重要性如何？**

​        表Table 1(d)显示了**没有自顶向下路径**的特征金字塔的结果。这种修改中，将1×1侧向连接和后面的3×3卷积添加到自底向上的金字塔中。该架构**模拟了重用金字塔特征层次结构**的效果(图Figure 1(b))。

​         Table 1(d)的结果仅仅与RPN基线相当，远远落后我们的。我们推测是由于**自底向上的金字塔**(图Figure. 1(b))的**不同层级**之间有**很大的语义差距** ，特别是对于非常深的ResNets网络。我们同样评估了表Table 1(d)**一个不共享头部参数的变种** ，但观测到同样差的性能。这个问题不会简单地被**层级特定头部修复**。



**侧向连接有多重要？**

​        表Taible 1(e)显示了**没有1×1侧向连接**的自顶向下特征金字塔的消融结果。这个自顶向下的金字塔具有**强大的语义特征**和**良好的分辨率**。但是我们认为这些特征的**位置并不精确**，因为这些特征图已经进行了**多次下采样和上采样**。更精确的特征位置可以**通过侧向连接** 直接将**更精细的自底向上特征图** **传递到** **自顶向下的特征图中**。因此，FPN的$AR^{1k}$的得分比表Tabel 1(e)高10个点。



**金字塔表示有多重要？**

​        可以将头部附加到最高分辨率、强语义特征图$P_2$上(即我们金字塔中的最好的层级)，而**不采用金字塔表示**。与单尺寸基线类似，我们将所有锚点分配给$P_2$特征图。这个变种(表Table 1(f))**比基线要好**，但**不如我们的方法**。RPN是一个具有**固定窗口大小** (即$3*3$ 的卷积核)的滑动窗口检测器，因此在**金字塔层级上扫描**可以**增加其对尺寸变化的鲁棒性**。

​        另外，我们注意到由于$P_2$较大的空间分辨率，**单独使用$P_2$会导致更多的锚点**(750k，表Table 1(f))。这个结果表明，**大量的锚点本身并不足以提高准确率**。

### 5.2. 用Fast/Faster R-CNN做目标检测

​        接下来我们在基于区域(非滑动窗口)的检测器上研究FPN。我们通过COCO类型的平均精度(AP)和PASCAL类型的AP(单个IoU阈值为0.5)来评估目标检测。我们还按照[21]中的定义报告了在小尺寸，中尺寸和大尺寸(即$AP_s, AP_m,  AP_l$)目标上的COCO AP。



**实现细节**

​       调整输入图像大小，使其较短边为800像素。同步SGD用于在8个GPU上训练模型。每个mini-batch数据包括**每个GPU2张图像**和**每张图像上512个RoI**。我们使用0.0001的权重衰减和0.9的动量。前60k次mini-batch数据的学习率为0.02，而接下来的20k次迭代学习率为0.002。我们<u>每张图像使用2000个RoIs进行训练</u>，1000个RoI进行测试。使用FPN在COCO数据集上训练Fast R-CNN需要约**10小时**。

#### 5.2.1 Fast R-CNN(固定提议框)

![](pic/FPN-tab2.png)

​        为了更好地调查FPN对仅基于区域的检测器的影响，我们在***一组固定的提议框***上进行Fast R-CNN的消融。我们选择冻结RPN在FPN上计算的提议(表Table 1(c))，因为它在能被检测器识别的小目标上具有良好的性能。为了简单起见，我们**没有在Fast R-CNN和RPN之间共享特征**，除非特别指定。

​        作为基于ResNet的Fast R-CNN基线，遵循[16]，我们采用输出尺寸为14×14的RoI池化，	并将所有conv5层作为头部的隐藏层。**这得到了31.9的AP**，如表Table 2(a)。**表Table 2(b)是一个基线**,采用了**MLP头部**，其具有2个隐藏的fc层，类似于我们的架构中的头部。它得到了28.8的AP，**表明2-fc头部** **没有**给我们带来任何超过表Table 2(a)中基线的**正交优势**。

​        表Table 2(c)显示了Fast R-CNN中我们的FPN结果。与表Table 2(a)中的基线相比，我们的方法将AP**提高了2.0个点**，小型目标AP提高了2.1个点。**与也采用2-fc头部的基线**相比(表Table 2(b))，我们的方法将AP**提高了5.1个点**。这些比较表明，对于基于区域的目标检测器，我们的**特征金字塔优于单尺寸特征**。

​        表Table 2 (d)和(e)表明，去除自顶向下的连接或去除侧向连接会导致较差的结果，类似于我们在上面的RPN小节中观察到的结果。值得注意的是，去除**自顶向下的连接**(表Table 2(d))显著降低了准确性，表明Fast R-CNN在**高分辨率特征图**中**受困于低级特征**。

​        在表Table 2(f)中，我们在**$P_2$的单个最好的尺寸特征图**上采用了Fast R-CNN。其结果(33.4 AP)**略低于使用所有金字塔层级**(33.9 AP，表Table 2(c))的结果。我们认为这是因为RoI池化是一种弯曲状(warping-like)操作，对**区域尺寸较不敏感**。尽管这个变体具有很好的准确性，但它是**基于$\{P_k\}$的RPN提议框**的，因此**已经从金字塔表示中受益**。

#### 5.2.2 Faster R-CNN(一致提议框)

![](pic/FPN-tab3.png)

​       在上面我们使用了一组**固定的提议框**来研究检测器。但是在Faster R-CNN系统中[29]，RPN和Fast R-CNN必须使用*相同的骨干网络*来实现**特征共享**。表Table 3显示了我们的方法和两个基线之间的比较，所有这些RPN和Fast R-CNN都使用一致的骨干架构。表Table 3(a)显示了我们再现[16]中描述的**Faster R-CNN系统的基线**。在受控的环境下，我们的FPN(表Table 3(c))比这个强劲的基线要好2.3个点的AP和3.8个点的AP@0.5。

​      注意到，**表Table 3(a)和(b)的基线**比He et al.[16]在中提供的基线(**表Table 3(\*)**) **强得多**。我们发现以下实现有助于缩小差距：(i)我们使用**800像素**的图像尺寸，而不是[11，16]中的**600像素**；(ii)与[11，16]中的**64个ROI**相比，我们训练时**每张图像有512个ROIs**，可以**加速收敛**；(iii)我们**使用5种尺寸的锚点**，而不是[16]中的4个(添加$32^2$)；(iv)在**测试时**，我们每张图像**使用1000个提议框**，而**不是[16]中的300个**。因此，与表3(*)中的He et al. 的ResNet-50 Faster R-CNN基线相比，我们的方法将**AP提高了7.6点**个并且将**AP@0.5提高了9.6个点**。



**共享特征**

![](pic/FPN-tab5.png)

​        在上面，为了简单起见，我们不共享RPN和Fast R-CNN之间的特征。在表Table 5中，我们按照[29]中描述的4步训练评估了共享特征。与[29]类似，我们发现**共享特征提高了一点精度**。**特征共享也缩短了测试时间**。



**运行时间**

​       通过特征共享，我们的基于FPN的Faster R-CNN系统使用**ResNet-50**在单个NVIDIA M40 GPU上每张图像的预测时间为**0.148秒**，使用**ResNet-101**的时间为**0.172秒**。作为比较，表Table 3(a)中的**单尺寸ResNet-50基线**运行时间为**0.32秒**。我们的方法通过FPN中的额外层引入了较小的额外成本，但具有**更轻的头部**。总体而言，我们的系统**比对应的基于ResNet的Faster R-CNN更快**。我们相信我们方法的高效性和简洁性将有利于未来的研究和应用。

#### 5.2.3 与COCO竞赛获胜者的比较

![](pic/FPN-tab4.png)

​        我们发现表Table 5中我们的ResNet-101模型在**默认学习速率**的情况下**没有进行足够的训练**。因此，在训练Fast R-CNN步骤时，我们将每个学习速率的mini-batch数据的数量增加了2倍。这将`minival`上的AP增加到了35.6，没有共享特征。该模型是我们提交给COCO检测排行榜的模型，如表4所示。由于时间有限，我们尚未评估其特征共享版本，这应该稍微好一些，如表Table 5所示。

​       表Table 4将我们方法的单模型结果与COCO竞赛获胜者的结果进行了比较，其中包括2016年冠军G-RMI和2015年冠军Faster R-CNN+++。没有添加额外的东西，我们的单模型记录就已经**超越**了这些强大的，经过严格设计的竞争对手。在`test-dev`数据集中，我们的方法在现有最佳结果上增加了**0.5个点的AP(36.2 vs.35.7)**和**3.4个点的AP@0.5(59.1 vs. 55.7)**。值得注意的是，我们的方法**不依赖图像金字塔**，只使用**单个尺寸输入图像**，但**在小型目标上仍然具有出色的AP**。在以前的方法中，只能通过的**高分辨率输入图像**来实现。

​        此外，我们的方法没有利用许多流行的改进，如**迭代回归**[9]，**困难负样本挖掘**[35]，**上下文建模**[16]，**更强大的数据增强**[22]等。**这些改进与FPN互补**，应该会进一步提高准确度。

​         最近，FPN在COCO竞赛的所有方面都取得了新的最佳结果，包括**检测，实例分割和关键点估计**。详情请参阅[14]。

## 6. 扩展：分割提议

 ![](pic/FPN-fig4.png)

​       我们的方法是一种通用金字塔表示，可用于除目标检测之外的其他应用。在本节中，我们使用**FPN生成分割提议框**，遵循DeepMask/SharpMask框架[27，28]。

​        DeepMask/SharpMask在裁剪图像上进行训练，可以预测**实例分割**和**目标/非目标评分**。在推断时，这些模型是卷积运行的，以在图像中**生成密集的提议框**。为了在多个尺寸上生成分割块，图像金字塔是必要的[27，28]。

​        改变FPN来**生成掩码提议框**很容易。我们对训练和推断都使用**全卷积设置**。我们在5.1小节中构造我们的特征金字塔并设置d=128。在特征金字塔的每个层级上，我们应用一个小的**5×5 MLP**以全卷积方式预测**14×14掩码**和，参见图Figure 4。此外，由于在[27,28]的图像金字塔中每组使用2个尺寸，我们使用输入大小为**7×7的第二个MLP**来处理半个组。这**两个MLP**扮演着类似于**RPN中锚点的角色**。该架构是端到端训练的，完整的实现细节在附录中给出。



### 6.1. 分割提议结果

![](pic/FPN-tab6.png)

​        结果如表Table 6所示。我们报告了**分割AR**和在**小型，中型和大型**目标上的**分割AR**，都是对于1000个提议框而言的。我们的**基线FPN模型**使用**单个5×5 MLP**的达到了**43.4**的AR。切换到稍大的7×7MLP，精度基本保持不变。**同时使用两个MLP**将精度提高到了**45.7的AR**。将掩码输出尺寸从14×14增加到28×28会增加AR另一个点(更大的尺寸开始降低准确度)。最后，加倍训练迭代将AR增加到**48.1**。

​        我们还报告了与DeepMask[27]，Sharp-Mask[28]和InstanceFCN[4]的比较，这是以前的掩模提议生成中的先进方法。我们的准确度超过这些方法8.3个点的AR。尤其是我们几乎将**小目标的精度提高了一倍**。

​        现有的掩码提议方法[27，28，4]是基于密集采样的图像金字塔的(例如，[27，28]中缩放$2^{\{−2:0.5:1\}}$)，使得它们是计算昂贵的。我们的方法基于FPN，速度明显加快(我们的模型运行速度为**6~7 FPS**)。这些结果表明，我们的模型是一个通用的特征提取器，可以替代图像金字塔以用于其他多尺寸检测问题。

## 7. 结论

​        我们提出了一个干净简单的框架，用于在**ConvNets内部构建特征金字塔**。我们的方法与几个强大的基线和竞赛获胜者相比, 显示了**显著的改进**。因此，它为特征金字塔的研究和应用提供了一个实用的解决方案，而不需要计算图像金字塔。最后，我们的研究表明，尽管深层ConvNets具有**强大的表示能力**以及它们对**尺寸变化的隐式鲁棒性**，使用**金字塔表示**对于明确地**解决多尺寸问**题仍然至关重要。

## 参考

[1] E. H. Adelson, C. H. Anderson, J. R. Bergen, P. J. Burt, and J. M. Ogden. Pyramid methods in image processing. RCA engineer, 1984.

[2] S. Bell, C. L. Zitnick, K. Bala, and R. Girshick. Inside-outside net: Detecting objects in context with skip pooling and recurrent neural networks. In CVPR, 2016.

[3] Z. Cai, Q. Fan, R. S. Feris, and N. Vasconcelos. A unified multi-scale deep convolutional neural network for fast object detection. In ECCV, 2016.

[4] J. Dai, K. He, Y. Li, S. Ren, and J. Sun. Instance-sensitive fully convolutional networks. In ECCV, 2016.

[5] N. Dalal and B. Triggs. Histograms of oriented gradients for human detection. In CVPR, 2005.

[6] P. Dollar, R. Appel, S. Belongie, and P. Perona. Fast feature pyramids for object detection. TPAMI, 2014.

[7] P.F.Felzenszwalb,R.B.Girshick,D.McAllester,andD.Ramanan. Object detection with discriminatively trained part-based models. TPAMI, 2010.

[8] G.GhiasiandC.C.Fowlkes.Laplacianpyramidreconstruction and refinement for semantic segmentation. In ECCV, 2016.

[9] S. Gidaris and N. Komodakis. Object detection via a multi-region & semantic segmentation-aware CNN model. In ICCV, 2015.

[10] S. Gidaris and N. Komodakis. Attend refine repeat: Active box proposal generation via in-out localization. In BMVC, 2016.

[11] R. Girshick. Fast R-CNN. In ICCV, 2015.

[12] R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In CVPR, 2014.

[13] B.Hariharan,P.Arbelaez,R.Girshick,andJ.Malik.Hypercolumns for object segmentation and fine-grained localization. In CVPR, 2015.

[14] K. He, G. Gkioxari, P. Dollar, and R. Girshick. Mask r-cnn. arXiv:1703.06870, 2017.

[15] K. He, X. Zhang, S. Ren, and J. Sun. Spatial pyramid pooling in deep convolutional networks for visual recognition. In ECCV. 2014.

[16] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.

[17] S. Honari, J. Yosinski, P. Vincent, and C. Pal. Recombinator networks: Learning coarse-to-fine feature aggregation. In CVPR, 2016.

[18] T. Kong, A. Yao, Y. Chen, and F. Sun. Hypernet: Towards accurate region proposal generation and joint object detection. In CVPR, 2016.

[19] A. Krizhevsky, I. Sutskever, and G. Hinton. ImageNet classification with deep convolutional neural networks. In NIPS, 2012.

[20] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition. Neural computation, 1989.

[21] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick. Microsoft COCO: Common objects in context. In ECCV, 2014.

[22] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, and S. Reed. SSD: Single shot multibox detector. In ECCV, 2016.

[23] W. Liu, A. Rabinovich, and A. C. Berg. ParseNet: Looking wider to see better. In ICLR workshop, 2016.

[24] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015.

[25] D. G. Lowe. Distinctive image features from scale-invariant keypoints. IJCV, 2004.

[26] A. Newell, K. Yang, and J. Deng. Stacked hourglass networks for human pose estimation. In ECCV, 2016.

[27] P. O. Pinheiro, R. Collobert, and P. Dollar. Learning to segment object candidates. In NIPS, 2015.

[28] P. O. Pinheiro, T.-Y. Lin, R. Collobert, and P. Dollár. Learning to refine object segments. In ECCV, 2016.

[29] S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In NIPS, 2015.

[30] S. Ren, K. He, R. Girshick, X. Zhang, and J. Sun. Object detection networks on convolutional feature maps. PAMI, 2016.

[31] O. Ronneberger, P. Fischer, and T. Brox. U-Net: Convolutional networks for biomedical image segmentation. In MIC- CAI, 2015.

[32] H. Rowley, S. Baluja, and T. Kanade. Human face detection in visual scenes. Technical Report CMU-CS-95-158R, Carnegie Mellon University, 1995.

[33] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. IJCV, 2015.

[34] P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun. Overfeat: Integrated recognition, localization and detection using convolutional networks. In ICLR, 2014.

[35] A. Shrivastava, A. Gupta, and R. Girshick. Training region-based object detectors with online hard example mining. In CVPR, 2016.

[36] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.

[37] J. R. Uijlings, K. E. van de Sande, T. Gevers, and A. W. Smeulders. Selective search for object recognition. IJCV, 2013.

[38] R. Vaillant, C. Monrocq, and Y. LeCun. Original approach for the localisation of objects in images. IEE Proc. on Vision, Image, and Signal Processing, 1994.

[39] S. Zagoruyko and N. Komodakis. Wide residual networks. In BMVC, 2016.

[40] S. Zagoruyko, A. Lerer, T.-Y. Lin, P. O. Pinheiro, S. Gross, S. Chintala, and P. Dollár. A multipath network for object detection. In BMVC, 2016. 10



## 疑问

1：5.2节"The input image is resized such that its shorter side has 800 pixels. Synchronized SGD is used to train the model on 8 GPUs. Each mini-batch involves 2 image per GPU and 512 RoIs per image. We use a weight decay of 0.0001 and a momentum of 0.9. The learning rate is 0.02 for the first 60k mini-batches and 0.002 for the next 20k. We use 2000 RoIs per image for training and 1000 for testing. Training Fast R-CNN with FPN takes about 10 hours on the COCO dataset" ;一会说每张图512个RoIs,一会又说2000个RoIs？





## 关于我们

我司正招聘文本挖掘、计算机视觉等相关人员，欢迎加入我们；也欢迎与我们在线沟通任何关于数据挖掘理论和应用的问题；

在长沙的朋友也可以线下交流, 坐标: 长沙市高新区麓谷新长海中心 B1栋8A楼09室

公司网址：http://www.embracesource.com/

Email: mick.yi@embracesource.com 或 csuyzt@163.com

