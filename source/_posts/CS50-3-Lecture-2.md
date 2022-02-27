---
title: CS50.3 Lecture 2
author: cadake
top: true
hide: false
cover: false
toc: true
mathjax: true
date: 2022-02-25 09:32:53
img:
password:
summary:
tags:
    - AI
    - algorithm
categories: 课程
---

# **Uncertainty**

## **引言**

一般而言，在现实中，AI只可能拥有解决问题的一部分信息，而其他的信息则是不确定的(uncertain)。因此，我们希望AI能够基于有限的的信息做出最合理，最可能(possible)的决策，即使并不完全准确。

<br>

## **Probability**

> Uncertainty can be represented as a number of events and the likehood, or probability, of each of them happening.

<br>

### **Unconditional Probability**

> Unconditional probability is the degree of belief in a proposition in the absence of any other evidence. 

<br>

### **Conditional Probability**

> Conditional probability is the degree of belief in a proposition given some evidence that has already been revealed.

![](https://s3.bmp.ovh/imgs/2022/02/f1d27cca1dc2ad2a.png)

![](https://s3.bmp.ovh/imgs/2022/02/8551605e95e2b91a.png)

<br>

### **Bayes' Rule**

![](https://s3.bmp.ovh/imgs/2022/02/588a41b170378675.png)

<br>

### **Joint Probability**

> Joint probability is the likelihood of multiple events all occurring.

<br>

## **Bayesian Networks**

> A Bayesian network is a data structure that represents the dependencies among random variables. Bayesian networks have the following properties:
> * They are directed graphs.
> * Each node on the graph represent a random variable.
> * An arrow from X to Y represents that X is a parent of Y. That is, the probability distribution of Y depends on the value of X.
> * Each node X has probability distribution P(X | Parents(X)).

<br>

**eg:**

前往约会的准时与否取决于很多因素，如图以贝叶斯网络的形式表达出这些因素（随机变量）间的依赖(dependency)关系。

![](https://s3.bmp.ovh/imgs/2022/02/4f3b478cefb4bd4f.png)

下雨(rain)是一个根节点，这意味着它不受网络中其他节点的影响。
rain  | probability
------|------------
none  | 0.7
light | 0.2
heavy | 0.1

火车轨道维修(maintenance)有一个父节点rain，这说明maintenance的概率分布受rain的影响。

R \ M     | yes | no
----------|-----|----
**none**  | 0.4 | 0.6
**light** | 0.2 | 0.8
**heavy** | 0.1 | 0.9

火车(train)准时出发与否的概率分布由是否下雨(rain)和是否有轨道维修(maintenance)共同决定。

R&M \ T  | on time | delayed
---------|---------|--------
none&yes | 0.8     | 0.2
none&no  | 0.9     | 0.1
light&yes| 0.6     | 0.4
light&no | 0.7     | 0.3
heavy&yes| 0.4     | 0.6
heavy&no | 0.5     | 0.5

约会(appointment)有很多祖先节点，这说明appointment的概率分布受很多因素的影响，但appointment的父节点只有一个，在贝叶斯网络中，我们只关心那些具有直接影响作用的依赖关系。

T \ A   | attend | miss
--------|--------|-----
on time | 0.9    | 0.1
delayed | 0.6    | 0.4

现在，我想知道在一个小雨(light rain)的天气下，没有轨道维修(no maintenance)，火车(train delayed)延迟了，而我错过约会(miss appointment)的概率。

$P(light, no, delayed, miss) = ?$

$P(light)$

$P(light)P(no|light)$

$P(light)P(no|light)P(delayed|light,no)$

$P(light)P(no|light)P(delayed|light,no)P(miss|delayed)$

$P(light,no,delayed,miss)=0.2\*0.8\*0.3\*0.4$

<br>

## **Inference**

基于概率，我们并不能推论出百分百正确的知识，但我们可以找出它们的概率分布。

> Inference has multiple properties.
> * Query X: the variable for which we want to compute the probability distribution.
> * Evidence variables E: one or more variables that have been observed for event e.
> * Hidden variables Y: variables that aren’t the query and also haven’t been observed. 
> * The goal: calculate P(X | e). 

<br>

### **Inference by Enumeration**

> Inference by enumeration is a process of finding the probability distribution of variable X given observed evidence e and some hidden variables Y.

$P(X|e)=P(X,e)/P(e)=\alpha * P(X,e) = \alpha \sum_{y}P(X,e,y)$

> In this equation, X stand for the query variable, e for the observed evidence, y for all the values of the hidden variables.

**eg:**

我想知道在小雨(light rain)天气和没有轨道维修(no maintenance)的情况，下我准时前往约会的概率分布。

$P(appointment|light, no) = \alpha P(appointment, light, no)$

$P(appointment, light, no) = P(appointment, light, no, delayed) + P(appointment, light, no, on time)$

然而，当网络中的变量节点非常多时，枚举推理(enumeration inference)并不是一种高效的方法。另一种想法是，在牺牲一定准确度的情况下，我们是否可以更加快速地得到一个相对正确的结果。

<br>

### **Sampling**

> Sampling is one technique of approximate inference. In sampling, each variable is sampled for a value according to its probability distribution. 

**eg:**
在上述appointment的例子中，我们使用抽样来获得大量样本。

从根节点rain开始。根据rain的概率分布，通过使用随机数，我们获得rain的一个值，比如none。

rain  | probability
------|------------
none  | 0.7
light | 0.2
heavy | 0.1

接下来根据maintenance在rain为none的取值下的条件概率分布，通过相同的方式，获得maintenance的一个样本值，比如yes。

R \ M     | yes | no
----------|-----|----
**none**  | 0.4 | 0.6

然后根据train在rain为none和maintenance为yes的取值下的条件概率分布，通过相同的方式，获得train的一个样本值，比如on time。

R&M \ T  | on time | delayed
---------|---------|--------
none&yes | 0.8     | 0.2

最后根据appointment在train为on time的取值下的条件概率，通过相同的方式，获得appointment的一个样本值，比如attend。

T \ A   | attend | miss
--------|--------|-----
on time | 0.9    | 0.1

最后，我们得到了一个样本(none, yes, on time, attend)。

通过计算机随机抽样，我们可以快速获得大量样本。大数定律告诉我们，当样本容量足够大时，$f \approx p$。我们可以使用这些样本估算概率分布。

然而，如果我想通过抽样(sampling)求一个条件概率，而这个条件事件发生的概率很低，那就意味着在大量的抽样中只有很少的样本是我们需要的。为此，我们浪费了大量的无关样本，并且为了保证估算准确性，我们又不得不抽取更多样本来使我们的目标样本数量具有一定规模。

<br>

### **Likelihood Weighting**

> Likelihood weighting uses the following steps:
> * Start by fixing the values for evidence variables.
> * Sample the non-evidence variables using conditional probabilities in the Bayesian network.
> * Weight each sample by its likelihood: the probability of all the evidence occurring.

**eg:**

在上述sampling的例子中，如果我想知道在火车没有延误(train on time)的情况下appointment的概率分布

<br>

## **Markov Models**

> So far, we have looked at questions of probability given some information that we observed. In this kind of paradigm, the dimension of time is not represented in any way. However, many tasks do rely on the dimension of time, such as prediction. To represent the variable of time we will create a new variable, X, and change it based on the event of interest, such that Xₜ is the current event, Xₜ₊₁ is the next event, and so on. To be able to predict events in the future, we will use Markov Models.

<br>

### **The Markov Assumption**

> The Markov assumption is an assumption that the current state depends on only a finite fixed number of previous states. 

<br>

### **Markov Chain**

> A Markov chain is a sequence of random variables where the distribution of each variable follows the Markov assumption. That is, each event in the chain occurs based on the probability of the event before it.

**a transition model:**

![](https://s3.bmp.ovh/imgs/2022/02/1ee66d10c5ae5518.png)

在这个例子中，明天的天气的概率分布受到今天的天气的影响。

我们可以使用马尔可夫假设模拟得到马尔可夫链，从而预测天气。

![](https://s3.bmp.ovh/imgs/2022/02/684ac831f4061331.png)

<br>

### **Hidden Markov Models**

> A hidden Markov model is a type of a Markov model for a system with hidden states that generate some observed event. This means that sometimes, the AI has some measurement of the world but no access to the precise state of the world. In these cases, the state of the world is called the hidden state and whatever data the AI has access to are the observations.

例如，现在有一个室内的门禁AI，它并不能直接知道天气的状况，但它装有一个摄像头，可以判断经过的人是否带伞，从而推断室外的天气状况。

![](https://s3.bmp.ovh/imgs/2022/02/57a8324c8f25561b.png)

<br>

### **Sensor Markov Assumption**

> The assumption that the evidence variable depends only on the corresponding state. 

使用隐马尔可夫模型，我们可以得到双层的马尔可夫链。下层代表我们的观察信息，上层代表导致这些观察现象背后的事件。

![](https://s3.bmp.ovh/imgs/2022/02/ef28679b6966f37e.png)

而在实际应用中，我们通常只有下层的一系列观察信息，我们所要做的就是根据隐马尔可夫模型推断得到上层链，也就是现象背后的本质。

<br>

## **结语**

原课程地址
[https://cs50.harvard.edu/ai/2020/weeks/2/](https://cs50.harvard.edu/ai/2020/weeks/2/)