# DeepLung: Deep 3D Dual Path Nets for Automated Pulmonary Nodule Detection and Classification

​       In this work, we present a fully automated lung computed  tomography (CT) cancer diagnosis system, DeepLung.  DeepLung consists of two components, nodule detection  (identifying the locations of candidate nodules) and classification  (classifying candidate nodules into benign or malignant).  Considering the 3D nature of lung CT data and  the compactness of dual path networks (DPN), two deep  3D DPN are designed for nodule detection and classification  respectively. Specifically, a 3D Faster Regions with  Convolutional Neural Net (R-CNN) is designed for nodule  detection with 3D dual path blocks and a U-net-like  encoder-decoder structure to effectively learn nodule features.  For nodule classification, gradient boosting machine  (GBM) with 3D dual path network features is proposed.  The nodule classification subnetwork was validated on a  public dataset from LIDC-IDRI, on which it achieved better  performance than state-of-the-art approaches and surpassed  the performance of experienced doctors based on  image modality. Within the DeepLung system, candidate  nodules are detected first by the nodule detection subnetwork,  and nodule diagnosis is conducted by the classification  subnetwork. Extensive experimental results demonstrate  that DeepLung has performance comparable to experienced  doctors both for the nodule-level and patient-level  diagnosis on the LIDC-IDRI dataset.1  

​         本文提出一个全自动的肺部CT癌症诊断系统，DeepLung。DeepLung包括两部分，结节检测(识别候选结节位置)和分类（分类候选结节为良性或恶性）。考虑到肺部CT数据的3D特性和双路径网络（DPN)的压缩性，设计了两个深度3D DPN分别用于结节检测和回归。特别地，一个带3D双路径块和U-net型编码-解码结构的Faster RCNN来高效的学习结节特征。对于结节分类，提出一个带3D双路径的梯度提升机（GBM)，结节分类子网络验证了来自LIDC-IDRI的公开数据集，取得了比state-of-the-art更好的性能，并且在基于图像模式上超过了有经验医生。在DeepLung系统中，首先通过结节检测子网络检测出候选结节，使用分类子网络做结节诊断。广泛的实验结果表明，DeepLung在LIDC-IDRI数据集上的结节级别和患者级别诊断方面的性能均与经验丰富的医生相当。



## 引言

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



​           我们的主要贡献如下：1）为了充分利用3D CT图像，分别设计了两个深度3D ConvNets用于结节检测和分类。由于3D ConvNet包含太多参数且难以在相对较小的公开肺部CT数据集上进行训练，因此我们采用3D双路径网络作为神经网络架构，因为DPN使用较少的参数并获得比残差网络更好的性能[3]。具体来说，受到Faster R-CNN对物体检测有效性的启发[13]，我们提出了基于3D双路径网络和U-net类编码器 - 解码器结构以及深3D双路径的3D Faster R-CNN用于结节检测结核分类网络。 2）与最先进的方法相比，我们的分类框架实现了更好的性能，并且超过了公共数据集LIDC-IDRI上有经验的医生的表现。 3）我们的全自动DeepLung系统，基于检测的结节分类，与经验丰富的医生在结节水平和患者水平诊断方面的表现相当。