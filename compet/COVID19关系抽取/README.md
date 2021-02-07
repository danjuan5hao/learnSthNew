# COVID19 关系抽取
## 想做成联合抽取的形式
第一步抽取s：
可以用【1. seq2seq】的方式，还可以用看作是【2. seq tagging】的方式。
第二步同时抽取p和o

想要用到共享特征：todo: 为什么会效果好
多任务是不是效果更好一些：todo: 为什么会效果好

## 模型结构
模型大量参考（照抄）苏神19年的模型架构。
博客地址：[基于DGCNN和概率图的轻量级信息抽取模型](https://kexue.fm/archives/6671)
<img src="rsrc\苏剑林百度信息抽取比赛模型架构图.png" style="zoom:50%;" />

1. 对text进行bert编码得到embedding

2. 找到text中的所有Subject
  2.1 用ptrNet找到S-head, 找到S-end 假设多个S之间不互相重合。

  2.2 或者还是用token classification的方式

3. 针对每个Subject找到其对应的 P 和 O。 

## 数据准备
### 准备S的数据

### 准备PO的数据

## 调参日记

## 参考文献
1. [基于DGCNN和概率图的轻量级信息抽取模型](https://kexue.fm/archives/6671)
2. [【AI Drive】第30期：平安人寿谢舒翼 | SemEval-2020自由文本信息抽取冠军方案解读 ](https://www.bilibili.com/video/av969121739/)
