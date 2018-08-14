[TOC]



原文地址: <a href=http://www.vis.uky.edu/~cheung/courses/ee639/Hammersley-Clifford_Theorem.pdf>Proof of Hammersley-Clifford Theorem</a>



### 定义1

​          一个无向图模型G称之为马尔科夫随机场(MRF),如果两个顶点被观察顶点分割情况下条件独立。也就是说对图中任意顶点$X_i$ ，以下条件属性成立
$$
P(X_i|X_{G\backslash i}) = P(X_i|X_{N_i})   \tag 1
$$
​           $X_{G \backslash i}$ 代表除了$X_i$ 之外的所有顶点，$X_{N_i}$ 代表i的所有邻居顶点-即所有与$X_i$ 相连的顶点。 



### 定义2

​        在无向图模型G上的一个概率分布$P(X)$ 称之为吉布斯分布，如果它能够因子分解为定义在团(clique)上的正函数的乘积，这些团覆盖了G的所有顶点和边。即
$$
P(X) = \frac 1 Z \prod_{c \in C_G} \phi_c(X_c)  \tag 2
$$
​        $C_G$ 是G上所有(最大)团的集合，$Z=\sum_x \prod_{c \in C_G} \phi_c(X_c)$ 是归一化常量。

​      

## 证明过程

​        Hammersley Clifford告诉我们这两个定义是等价的，下面将证明这个定理。

### 反向证明(吉布斯分布=>MRF)

​       设$D_i = N_i \bigcup \{X_i\}$ 是包含$X_i$ 邻居顶点和$X_i$ 本身的集合。从等式(1)的右边开始
$$
P(X_i|X_{N_i}) = \frac {P(X_i,X_{N_i})} {P(X_{N_i})}  \tag 3
$$

$$
\ \ \ \ \ \ \ =\frac {\sum_{G \backslash D_i} \prod_{c \in C_G} \phi_c(X_c)} {\sum_{x_i} \sum_{G \backslash D_i} \prod_{c \in C_G} \phi_c(X_c)}   \tag 4
$$

​         基于是否包含$X_i$ 将最大团$C_G$ 分为两组：

$C_i = {c \in C_G: X_i \in c}$  和 $R_i = {c \in C_G: X_i \notin c}$ ；现在可以将等式(4)分为$C_i$ 和 $R_i$ 上的乘积。
$$
P(X_i|X_{N_i}) =\frac {\sum_{G \backslash D_i} \prod_{c \in C_i} \phi_c(X_c) \prod_{c \in R_i} \phi_c(X_c)} {\sum_{x_i} \sum_{G \backslash D_i} \prod_{c \in C_i} \phi_c(X_c) \prod_{c \in R_i} \phi_c(X_c)}    \tag 5
$$

$$
= \frac {\prod_{c \in C_i} \phi_c(X_c) \sum_{G \backslash D_i} \prod_{c \in R_i} \phi_c(X_c)} {\sum_{x_i}  \prod_{c \in C_i} \phi_c(X_c) \sum_{G \backslash D_i} \prod_{c \in R_i} \phi_c(X_c)}    \tag 5
$$

​         在$G \backslash D_i$ 上的求和可以移到$C_i$ 乘积的后面，因为$C_i$ 团中所有的顶点一定都来自$D_i$; 因为$C_i$ 只包含$X_i$ 和与$X_i$ 相邻的顶点，由$D_i$ 的定义可知。

​         同样注意到因子$\sum_{G \backslash D_i} \prod_{c \in R_i} \phi_c(X_c)$ 没有包含$X_i$ ,并且可以从分母移除，因为分子也包含了它。因此有：
$$
P(X_i|X_{N_i}) = \frac {\prod_{c \in C_i} \phi_c(X_c)} {\sum_{x_i} \prod_{c \in C_i} \phi_c(X_c)}    \tag 7
$$

$$
\ \ \ \ \ \ \ \ \ \ =\frac {\prod_{c \in C_i} \phi_c(X_c)} {\sum_{x_i} \prod_{c \in C_i} \phi_c(X_c)} * \frac {\prod_{c \in R_i} \phi_c(X_c)} {\prod_{c \in R_i} \phi_c(X_c)}          \tag 8
$$

$$
\ \ \ \ \ \ \ \ \ \ =\frac {\prod_{c \in C_G} \phi_c(X_c)} {\sum_{x_i} \prod_{c \in C_G} \phi_c(X_c)}  \tag 9
$$

$$
= \frac {P(X)} {P(X_{G \backslash \{i\}})}    \tag {10}
$$

$$
= P(X_i|X_{G \backslash \{i\}})           \tag {11}
$$

​       消除了$G \backslash D_i$ 上的求和项后，在公式(8)的分子分母乘上一个相同的因子，再次引入势函数；最终公式(11)与公式(1)的左边相等，证明了反向等价。



### 正向证明(MRF=>吉布斯分布)


$$
f_s(X_s=x_s) = \prod_{z \subset s} P(X_z=x_z,X_{G \backslash z} =0)^{-1^{|s|-|z|}}   \tag {12}
$$


需要证明两点

1. $\prod_{s \subset G} f_s(X_s) = P(X)$
2. $f_s(X_s) =1$ 如果$s$ 不是一个团

$$
0 = (1 - 1)^K = C_0^K - C_1^K + C_2^K + ... ... + (-1)^KC_K^K       \tag {13}
$$



​                              $f_s(X_s=x_s) $
$$
= \prod_{z \subset s} P(X_z=x_z,X_{G \backslash z} =0)^{-1^{|s|-|z|}}   \tag {14}
$$

$$
= \prod_{w \subset s \backslash \{a,b\}} 
\left[ \frac 
{P(X_w,X_{G \backslash w}=0)  P(X_{w \bigcup \{a,b\}},X_{G \backslash w \bigcup \{a,b\}}=0)} 
{P(X_{w \bigcup \{a\}},X_{G \backslash w \bigcup \{a\}}=0) P(X_{w \bigcup \{b\}},X_{G \backslash w \bigcup \{b\}}=0) } 
\right]^{-1^*}          \tag {15}
$$

$$
\frac {P(X_w,X_{G \backslash w}=0)} {P(X_{w \bigcup \{a\}},X_{G \backslash w \bigcup \{a\}}=0)}
$$

$$
=\frac {P(X_a=0|X_b=0,X_w,X_{G \backslash w \bigcup \{a,b\}}=0) P(X_b=0,X_w,X_{G \backslash w \bigcup \{a,b\}}=0)} 
 {P(X_a|X_b=0,X_w,X_{G \backslash w \bigcup \{a,b\}}=0) P(X_b=0,X_w,X_{G \backslash w \bigcup \{a,b\}}=0)}    \tag {16}
$$

$$
=\frac {P(X_a=0|X_b,X_w,X_{G \backslash w \bigcup \{a,b\}}=0) P(X_b,X_w,X_{G \backslash w \bigcup \{a,b\}}=0)} 
 {P(X_a|X_b,X_w,X_{G \backslash w \bigcup \{a,b\}}=0) P(X_b,X_w,X_{G \backslash w \bigcup \{a,b\}}=0)}    \tag {17}
$$

$$
=\frac {P(X_{w \bigcup \{b\}},X_{G \backslash w \bigcup \{b\}}=0)}
{P(X_{w \bigcup \{a,b\}},X_{G \backslash w \bigcup \{a,b\}}=0)}    \tag {18}
$$

