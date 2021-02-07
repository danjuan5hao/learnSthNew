# Efficient Large Scale Neural Domain Classification With Personalized Attention

## 为什么会有这篇文章

对话系统需要支持扩展Domaim。方便第三方快速扩展技能。但是第三方设计的技能鱼龙混杂，容易产生一下问题
1. 一条指令往往能激活多个domain
2. domain的扩展会非常迅速
3. domain的数量会非常多。


论文提出的架构就能快速帮助解决这个问题。

## 模型如何针对这些问题进化的

### Domain 非常多

1. 速度的损失在softmax

   1. 修改数据集，一个utterance对应一个domain

   2. 修改损失函数，让她从多分类问题 变成一个二分类问题。

      具体做法就是：把非类的损失加起来做平均，然后让 正类和非类 距离尽量拉开

### 一条指令往往能激活多个domain

加入对用户使用domain的偏好,之前用过的domain容易激活

1. 加上1-bit flag
2. 对使用过的embedding做attention

### Domain的增长速度非常快

所以新添加的domain，要快速训练并且 影响面要小；

1. utte 和 domain的编码分开了
2. 新增domain 需要训练embedding和他自己的clf，训练方法通过。
   1. 初始化embedding的方法：

<img src="rsrc\新domain_emb初始化方式.png" style="zoom:100%;" />



## 模型整体架构

![](rsrc\模型结构.png)









