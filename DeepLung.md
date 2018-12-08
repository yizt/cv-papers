# DeepLung: Deep 3D Dual Path Nets for Automated Pulmonary Nodule Detection and Classification

[TOC]



​       In this work, we present a fully automated lung computed  tomography (CT) cancer diagnosis system, DeepLung.  DeepLung consists of two components, nodule detection  (identifying the locations of candidate nodules) and classification  (classifying candidate nodules into benign or malignant).  Considering the 3D nature of lung CT data and  the compactness of dual path networks (DPN), two deep  3D DPN are designed for nodule detection and classification  respectively. Specifically, a 3D Faster Regions with  Convolutional Neural Net (R-CNN) is designed for nodule  detection with 3D dual path blocks and a U-net-like  encoder-decoder structure to effectively learn nodule features.  For nodule classification, gradient boosting machine  (GBM) with 3D dual path network features is proposed.  The nodule classification subnetwork was validated on a  public dataset from LIDC-IDRI, on which it achieved better  performance than state-of-the-art approaches and surpassed  the performance of experienced doctors based on  image modality. Within the DeepLung system, candidate  nodules are detected first by the nodule detection subnetwork,  and nodule diagnosis is conducted by the classification  subnetwork. Extensive experimental results demonstrate  that DeepLung has performance comparable to experienced  doctors both for the nodule-level and patient-level  diagnosis on the LIDC-IDRI dataset.1  

​         本文提出一个全自动的肺部CT癌症诊断系统，DeepLung。DeepLung包括两部分，结节检测(识别候选结节位置)和分类（分类候选结节为良性或恶性）。考虑到肺部CT数据的3D特性和双路径网络（DPN)的压缩性，设计了两个深度3D DPN分别用于结节检测和回归。特别地，一个带3D双路径块和U-net型编码-解码结构的Faster RCNN来高效的学习结节特征。对于结节分类，提出一个带3D双路径的梯度提升机（GBM)，结节分类子网络验证了来自LIDC-IDRI的公开数据集，取得了比state-of-the-art更好的性能，并且在基于图像模式上超过了有经验医生。在DeepLung系统中，首先通过结节检测子网络检测出候选结节，使用分类子网络做结节诊断。广泛的实验结果表明，DeepLung在LIDC-IDRI数据集上的结节级别和患者级别诊断方面的性能均与经验丰富的医生相当。



## 1. 引言

Lung cancer is the most common cause of cancer-related
death in men. Low-dose lung CT screening provides an effective
way for early diagnosis, which can sharply reduce
the lung cancer mortality rate. Advanced computer-aided
diagnosis systems (CADs) are expected to have high sensitivities
while at the same time maintaining low false positive
rates. Recent advances in deep learning enable us to rethink
the ways of clinician lung cancer diagnosis.

​         肺癌是男性因癌致死最常见的情况，低剂量的肺部CT扫描提供一个有效的早起诊断，可以极大减少肺癌死亡率。先进的计算机辅助诊断系统（CADs)被期望有很高的灵敏度同时保持低的假阳性。深度学习的最新进展使我们能够重新思考临床医生肺癌诊断的方法。



Current lung CT analysis research mainly includes nodule
detection [6, 5], and nodule classification [26, 25, 14,
33]. There is few work on building a complete lung CT
cancer diagnosis system for fully automated lung CT cancer
diagnosis using deep learning, integrating both nodule
detection and nodule classification. It is worth exploring a
whole lung CT cancer diagnosis system and understanding
how far the performance of current deep learning technology
differs from that of experienced doctors. To our best
knowledge, this is the first work for a fully automated and
complete lung CT cancer diagnosis system using deep nets.

​        当前肺部CT分析研究主要包括结节检测[6,5],和结节分类[26,25,14,33]；很少使用深度学习构建一个集成结节检测和结节分类的完整肺部CT癌症诊断系统，来全自动化肺部CT癌症诊断的工作。值得探索整个肺部CT癌症诊断系统，并了解当前深度学习技术的表现与经验丰富的医生的差异程度。据我们所知，这是第一个时候用深度网络全自动化整个肺部CT癌症诊断系统。



The emergence of large-scale dataset, LUNA16 [24],  accelerated the nodule detection related research. Typically,  nodule detection consists of two stages, region proposal  generation and false positive reduction. Traditional  approaches generally require manually designed features  such as morphological features, voxel clustering and pixel  thresholding [20, 15]. Recently, deep ConvNets, such as  Faster R-CNN [21, 17] and fully ConvNets [18, 37, 31, 30,  29], are employed to generate candidate bounding boxes  [5, 6]. In the second stage, more advanced methods or complex  features, such as carefully designed texture features,  are used to remove false positive nodules. Because of the  3D nature of CT data and the effectiveness of Faster R-CNN  for object detection in 2D natural images [13], we design a  3D Faster R-CNN for nodule detection with 3D convolutional  kernels and a U-net-like encoder-decoder structure to  effectively learn latent features [22]. The U-Net structure is  basically a convolutional autoencoder, augmented with skip  connections between encoder and decoder layers [22]. Although  it has been widely used in the context of semantic  segmentation, being able to capture both contextual and local  information should be very helpful for nodule detections  as well. Because 3D ConvNet has too large a number of parameters  and is difficult to train on public lung CT datasets  of relatively small sizes, 3D dual path network is employed  as the building block since deep dual path network is more  compact and provides better performance than deep residual  network at the same time [3].

​        大规模数据集LUNA16 [24]的出现加速了结节检测的相关研究。通常，结节检测包括两个阶段，区域建议生成和假阳性消减。 传统方法通常需要手动设计的特征，如形态特征，三位像素聚类和像素阈值[20,15]。最近，深度卷积网络，如Faster R-CNN [21,17]和全卷积网络 [18,37,31,30,29]，被用来生成候选边界框[5,6]。在第二阶段，使用更先进的方法或复杂的特征，例如精心设计的纹理特征，来消除假阳性结节。由于CT数据的3D特性和Faster R-CNN在2D自然图像在目标检测的有效性[13]，我们设计了一个3D Faster R-CNN用于结节检测，具有3D卷积核和U-net类编码器  - 解码器结构，有效地学习潜在特征[22]。尽管它已被广泛用于语义分割，但能够捕获上下文和局部信息对于结节检测也应该非常有用。由于3D 卷积网络具有过多的参数并且难以在相对较小规模的公开肺部CT数据集上进行训练，因此采用3D双路径网络作为构建块，因为深度双路径网络更紧凑并且同时提供比深度残差网络更好的性能[3]。



Before the era of deep learning, manual feature engineering followed by classifiers was the general pipeline for  nodule classification [10]. After the large-scale LIDC-IDRI  [2] dataset became publicly available, deep learning-based  methods have become the dominant framework for nodule  classification research [25, 35]. Multi-scale deep ConvNet  with shared weights on different scales has been proposed  for the nodule classification [26]. The weight sharing  scheme reduces the number of parameters and forces the  multi-scale deep ConvNet to learn scale-invariant features.  Inspired by the recent success of dual path network (DPN)  on ImageNet [3, 4], we propose a novel framework for CT  nodule classification. First, we design a deep 3D dual path  network to extract features. As gradient boosting machines  (GBM) are known to have superb performance given effective  features, we use GBM with deep 3D dual path features,  nodule size, and cropped raw nodule CT pixels for the nodule  classification [8]. 

​       在深度学习时代之前，手工特征工程和分类器是结节分类的一般流程[10]，在大规模LIDC-IDRI [2]数据集公开后，基于深度学习的方法已成为结核分类研究的主要框架[25,35]。已经提出了在不同尺度共享权重的多尺度深度ConvNet [26]用于结节分类。权重共享方案减少了参数量，并迫使多尺度深度ConvNet学习尺寸不变的特征。受双路径网络（DPN）最近在ImageNet上成功的启发[3,4]，我们提出了一种新的CT结节分类框架。首先，我们设计一个深度3D 双路径网络来提取特征，由于已知梯度提升机（GBM）具有出色的性能，因此我们使用具有深度3D双路径特征的GBM，结节尺寸和裁剪的原始结节CT像素[8]用于结节分类。

Finally, we built a fully automated lung CT cancer diagnosis  system, henceforth called DeepLung, by combining  the nodule detection network and nodule classification  network together, as illustrated in Fig. 1. For a CT image,  we first use the detection subnetwork to detect candidate  nodules. Next, we employ the classification subnetwork  to classify the detected nodules into either malignant  or benign. Finally, the patient-level diagnosis result can be  achieved for the whole CT by fusing the diagnosis result of  each nodule. 

​         最后，我们通过将结节检测网络和结节分类网络结合在一起，构建了一个全自动肺癌CT诊断系统，因此称为DeepLung，如图Fig 1所示。对于CT图像，我们首先使用检测子网来检测候选结节。 接下来，我们使用分类子网将检测到的结节分类为恶性或良性。 最后，通过融合每个结节的诊断结果，可以实现整个CT的患者级别诊断结果。

​          



Our main contributions are as follows: 1) To fully exploit  the 3D CT images, two deep 3D ConvNets are designed  for nodule detection and classification respectively.  Because 3D ConvNet contains too many parameters and is  difficult to train on relatively small public lung CT datasets,  we employ 3D dual path networks as the neural network  architecture since DPN uses less parameters and obtains  better performance than residual network [3]. Specifically,  inspired by the effectiveness of Faster R-CNN for object  detection [13], we propose 3D Faster R-CNN for nodule  detection based on 3D dual path network and U-net-like encoder-decoder structure, and deep 3D dual path network  for nodule classification. 2) Our classification framework  achieves better performance compared with state-of-the-art  approaches, and surpasses the performance of experienced  doctors on the public dataset, LIDC-IDRI. 3) Our fully automated  DeepLung system, nodule classification based on  detection, is comparable to the performance of experienced  doctors both on nodule-level and patient-level diagnosis.  



​           我们的主要贡献如下：1）为了充分利用3D CT图像，分别设计了两个深度3D ConvNets用于结节检测和分类。由于3D ConvNet包含太多参数且难以在相对较小的公开肺部CT数据集上进行训练，因此我们采用3D双路径网络作为神经网络架构，因为DPN使用较少的参数并获得比残差网络更好的性能[3]。具体来说，受到Faster R-CNN对物体检测有效性的启发[13]，我们提出了基于3D双路径网络和U-net类编码器 - 解码器结构的3D Faster R-CNN用于结节检测，和深度3D双路径网络用于结节分类。 2）与最先进的方法相比，我们的分类框架实现了更好的性能，并且在公共数据集LIDC-IDRI上超过了有经验的医生的表现。 3）我们的全自动DeepLung系统，基于检测的结节分类，与经验丰富的医生在结节水平和患者水平诊断性能相当。



## 2. 相关工作

Traditional nodule detection involves hand-designed features  or descriptors [19] requiring domain expertise. Recently,  several works have been proposed to use deep ConvNets  for nodule detection to automatically learn features,  which is proven to be much more effective than handdesigned  features. Setio et al. proposes multi-view ConvNet  for false positive nodule reduction [23]. Due to the  3D nature of CT scans, some work proposed 3D ConvNets  to handle the challenge. The 3D fully ConvNet (FCN) is  proposed to generate region candidates, and deep ConvNet  with weighted sampling is used for false positive reduction  [6]. Ding et al. and Liao et al. use the Faster R-CNN to  generate candidate nodules followed by 3D ConvNets to remove  false positive nodules [5, 17]. Due to the effective  performance of Faster R-CNN [13, 21], we design a novel  network, 3D Faster R-CNN with 3D dual path blocks, for  the nodule detection. Further, a U-net-like encoder-decoder  scheme is employed for 3D Faster R-CNN to effectively  learn the features [22].

​        传统的结节检测涉及手工设计的特征或描述符[19]，需要领域专业知识。最近，已经提出了几项工作来使用深度ConvNets进行结节检测以自动学习特征，这被证明比手工设计的特征更有效。 Setio et al. 提出多视图ConvNet用于假阳性结节消减[23]。由于CT扫描的3D特性，一些工作提出3D ConvNets来应对挑战。提出使用3D全卷积网络（FCN）来生成候选区域，并且使用带加权采样的深度ConvNet进行假阳性消减[6]。Ding et al. 和Liao et al.使用Faster R-CNN生成候选结节，然后使用3D ConvNets去除假阳性结节[5,17]。由于Faster R-CNN [13,21]的高效性能，我们设计了一种新的网络，带3D双路径块的3D Faster R-CNN，用于结节检测。此外，U-net型的编码器 - 解码器方案被用于3D Faster R-CNN以有效地学习特征[22]。



Nodule classification has traditionally been based on  segmentation [7] and manual feature design [1]. Several  works designed 3D contour feature, shape feature and texture  feature for CT nodule diagnosis [32, 7, 10]. Recently,  deep networks have been shown to be effective for medical  images. Artificial neural network was implemented for CT  nodule diagnosis [28]. More computationally effective network,  multi-scale ConvNet with shared weights for different  scales to learn scale-invariant features, is proposed for nodule classification [26]. Deep transfer learning and multi instance  learning is used for patient-level lung CT diagnosis  [25, 36]. A comparative study on 2D and 3D ConvNets  is conducted and 3D ConvNet is shown to be better than  2D ConvNet for 3D CT data [33]. Furthermore, a multitask  learning and transfer learning framework is proposed  for nodule diagnosis [14]. Different from their approaches,  we propose a novel classification framework for CT nodule  diagnosis. Inspired by the recent success of deep dual  path network (DPN) on ImageNet [3], we design a novel  3D DPN to extract features from raw CT nodules. In part to  the superior performance of GBM with complete features,  we employ GBM with different levels of granularity ranging  from raw pixels, DPN features, to global features such  as nodule size for the nodule diagnosis. Patient-level diagnosis  can be achieved by fusing the nodule-level diagnosis.

​       结节分类传统上基于分割[7]和手动特征设计[1]。 几项工作为CT结节诊断设计了3D轮廓特征，形状特征和纹理特征[32,7,10]。最近，深度网络已被证明对医学图像有效。 人工神经网络用于CT结节诊断[28]。 为结节分类提出了更具计算效率的网络，具有不同尺寸间共享权重以学习尺寸不变特征[26]的多尺寸ConvNet。深度迁移学习和多实例学习用于患者级别肺部CT诊断[25,36]。对2D和3D ConvNets进行的比较研究，显示对于3D CT数据[33] 3D ConvNet优于2D ConvNet。此外，提出了一个多任务学习和转移学习框架用于结节诊断[14]。与他们的方法不同，我们提出了一种新颖的CT结节诊断框架。 受最近深度双路径网络（DPN）在ImageNet成功的的启发[3]，我们设计了一种新颖的3D DPN，用于从原始CT结节中提取特征。 部分由于具有完整特征的GBM的卓越性能，我们对结节诊断采用具有不同粒度级别的GBM，从原始像素，DPN特征到结节大小等全局特征。 通过融合结节级别诊断可以实现患者级别的诊断。



​         

### 3. DeepLung框架

Our fully automated lung CT cancer diagnosis system
consists of two parts: nodule detection and classification.
We design a 3D Faster R-CNN for nodule detection, and
propose GBM with deep 3D DPN features, raw nodule CT
pixels and nodule size for nodule classification.

​           我们的全自动肺部CT癌症诊断系统包括两部分：肺部检测和分类；我们为结节检测设计了3D Faster R-CNN ，并为结节分类提出了带深度3D DPN特征、原始结节CT像素和结节尺寸的GBM。

### 3.1. 用于结节检测的带深度3D双路径网络的3D Faster R-CNN 

Inspired by the success of dual path network on the ImageNet  [3, 4], we design a deep 3D DPN framework for lung  CT nodule detection and classification in Fig. 3 and Fig.  4. Dual path connection benefits both from the advantage  of residual learning and that of dense connection [11, 12].  The shortcut connection in residual learning is an effective  way to eliminate vanishing gradient phenomenon in very  deep networks. From a learned feature sharing perspective,  residual learning enables feature reuse, while dense connection  has an advantage of exploiting new features [3]. Additionally,  densely connected network has fewer parameters  than residual learning because there is no need to relearn  redundant feature maps. The assumption of dual path con nection is that there might exist some redundancy in the exploited
features. And dual path connection uses part of feature maps for dense connection and part of them for residual learning. In implementation, the dual path connection splits its feature maps into two parts. One part, F(x)[d :], is used for residual learning, the other part, F(x)[: d], is used for dense connection as shown in Fig. 2. Here d is a hyper-parameter for deciding how many new features to be exploited. The dual path connection can be formulated as

​        受双路径网络在ImageNet上成功的启发[3,4]，我们在图Fig 3和图Fig 4中设计了一个用于肺部CT结节检测和分类的深度3D DPN框架。双路连接受益于残差学习和密集连接的优势[11,12]。 残差学习中的快捷连接是消除深度网络中消失梯度现象的有效方法。 从学习的特征共享角度来看，残差学习可以实现特征重用，而密集连接则具有利用新功能的优势[3]。此外，密集连接的网络比残差学习具有更少的参数，因为不需要重新学习冗余特征图。 双路径连接假设在被利用的特征中可能存在一些冗余。并且双路径连接使用部分特征进行密集连接，部分用于残差学习。 在实现中，双路径连接将其特征分成两部分。 一部分F（x）[d：]用于残差学习，另一部分F（x）[：d]用于密集连接，如图Fig 2所示。这里d是超参数 用于决定要采用的新功能的数量。 双路径连接可以表示为


$$
y = \bf G(x[:d], \bf F(x) [:d], \bf F(x)[d:] + x[d:]),  \tag 1
$$
where y is the feature map for dual path connection, G is  used as ReLU activation function, F is convolutional layer  functions, and x is the input of dual path connection block.  Dual path connection integrates the advantages of the two  advanced frameworks, residual learning for feature reuse  and dense connection for the ability to exploit new features,  into a unified structure which obtained success on the ImageNet  dataset[4]. We design deep 3D neural nets based on  3D DPN because of its compactness and effectiveness 。

​        y是双路径连接的特征，G是ReLU激活函数，F是卷积层函数，x是双路径连接块的输入。双路径连接集成了两个高级框架的优势，将残差学习的特征重用和密集连接的新特征利用的集成到一个统一的结构中，该结构在ImageNet数据集上取得了成功[4]。 由于其紧凑性和有效性，我们设计了基于3D DPN的深度3D神经网络。

The 3D Faster R-CNN with a U-net-like encoderdecoder  structure and 3D dual path blocks is illustrated in  Fig. 3. Due to the GPU memory limitation, the input of 3D  Faster R-CNN is cropped from 3D reconstructed CT images  with pixel size 96 × 96 × 96. The encoder network  is derived from 2D DPN [3]. Before the first max-pooling,  two convolutional layers are used to generate features. After  that, eight dual path blocks are employed in the encoder  subnetwork. We integrate the U-net-like encoder-decoder  design concept in the detection to learn the deep nets efficiently  [22]. In fact, for the region proposal generation, the  3D Faster R-CNN conducts pixel-wise multi-scale learning  and the U-net is validated as an effective way for pixel-wise  labeling. This integration makes candidate nodule generation  more effective. In the decoder network, the feature  maps are processed by deconvolution layers and dual path  blocks, and are subsequently concatenated with the corresponding  layers in the encoder network [34]. Then a convolutional  layer with dropout (dropout probability 0.5) is used  in the second to the last layer. In the last layer, we design 3  anchors, 5, 10, 20, for scale references which are designed  based on the distribution of nodule sizes. For each anchor,  there are 5 parts in the loss function, classification loss Lcls  for whether the current box is a nodule or not, regression  loss Lreg for nodule coordinates x, y, z and nodule size d.

​        具有U-net型编码器解码器结构和3D双路径块的3D Faster R-CNN如图Fig 3所示。由于GPU内存限制，3D Faster R-CNN的输入是从3D重建CT图像中裁剪出来的，像素大小为96×96×96。编码器网络来自2D DPN [3]。 在第一个最大池化之前，使用两个卷积层来生成特征。之后，在编码器子网中使用8个双路径块。我们在检测中集成了U-net型的编码器 - 解码器设计概念，以便有效地学习深度网络[22]。实际上，对于区域提议生成，3D Faster R-CNN进行像素方式的多尺度学习，并且U-net被验证为用于像素标注的有效方式。这种集成使候选结节的产生更加有效。在解码器网络中，特征图由反卷积层和双路径块处理，并随后与编码器网络中的相应层拼接[34]。然后在第二层到最后一层使用带dropout（丢失概率0.5）的卷积层。在最后一层，我们基于结节尺寸分布设计了3个anchor，5,10,20，用做尺寸参考。对于每个anchor，损失函数中有5个部分，当前边框是否为结节的分类损失$L_{cls}$，结节坐标x，y，z和结节大小d的回归损失$L_{reg}$。



If an anchor overlaps a ground truth bounding box with  the intersection over union (IoU) higher than 0.5, we consider  it as a positive anchor (p  ? = 1). On the other hand,  if an anchor has IoU with all ground truth boxes less than  0.02, we consider it as a negative anchor (p  ? = 0). The  multi-task loss function for the anchor i is defined as

​        如果anchor和ground truth边框的交并比（IoU)大于0.5，则为正anchor($p^*=1$), 另一方面,如果一个anchor与所有ground truth边框的IoU都小于0.02，则作为负anchor($p^*=0$)。对于anchor $i$的多任务损失函数定义为：
$$
L(p_i, t_i) = λL_{cls}(p_i, p^*
_i) + p^*_i L_{reg}(t_i
, t_i^*),  \tag 2
$$
where pi is the predicted probability for current anchor i being a nodule, ti is the predicted relative coordinates for nodule position, which is defined as

​        $p_i$ 当前anchor $i$ 预测为结节的概率，$t_i$ 是相应的结节位置坐标预测，定义为：
$$
t_i = (\frac {x-x_a} {d_a}, \frac {y-y_b} {d_a}, \frac {z- z_a} {d_a}, log(\frac {d} {d_a})) \tag 3
$$
 where (x, y, z, d) are the predicted nodule coordinates and diameter in the original space, (xa, ya, za, da) are the coordinates and scale for the anchor i. For ground truth nodule position, it is defined as

​          $ (x, y, z, d) $ 是预测的原空间中结节坐标和直径， $(x_a, y_a, z_a, d_a) $ 是anchor $i$ 的坐标和尺寸，对于ground truth结节位置，定义如下：
$$
t_i^* = (\frac {x^*-x_a} {d_a}, \frac {y^*-y_b} {d_a}, \frac {z^*- z_a} {d_a}, log(\frac {d^*} {d_a})) \tag 3
$$
where (x?, y?, z?, d?) are nodule ground truth coordinatesand diameter. The λ is set as 0.5. For Lcls, we used binary cross entropy loss function. For Lreg, we used smooth l1 regression loss function [9].

​         $(x^*, y^*, z^*, d^*)$ 是结节的实际坐标和直径， λ 为0.5; 对于$L_{cls}$ 使用二分类交叉熵损失函数，对于$L_{reg}$ 使用平滑L1回归损失函数[9].



### 3.2 用于结节分类的带3D 双路径网络特征的GBM

For CT data, advanced method should be effective to extract  3D volume feature [33]. We design a 3D deep dual  path network for the 3D CT lung nodule classification in Fig. 4. The main reason we employ dual modules for detection  and classification is that classifying nodules into benign  and malignant requires the system to learn finer-level  features, which can be achieved by focusing only on nodules.  In addition, it allows to introduce extra features in the  final classification. We first crop CT data centered at predicted  nodule locations with size 32 × 32 × 32. After that,  a convolutional layer is used to extract features. Then 30  3D dual path blocks are employed to learn higher level features.  Lastly, the 3D average pooling and binary logistic  regression layer are used for benign or malignant diagnosis.

