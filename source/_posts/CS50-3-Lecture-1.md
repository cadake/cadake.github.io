---
title: CS50.3 Lecture 1
author: cadake
top: true
hide: false
cover: false
toc: true
mathjax: true
date: 2022-02-22 09:50:15
img:
password:
summary: Humans reason based on existing knowledge and draw conclusions. The concept of representing knowledge and drawing conclusions from it is also used in AI.
tags:
    - AI
    - algorithm
categories: 课程
---

# **Knowledge**

## **引言**

人类根据已有的知识进行推理并得出结论。这种基于已有知识诞生新知识的方法也被用于AI身上。

<br>

Consider the following sentences:

1. If it didn’t rain, Harry visited Hagrid today.
2. Harry visited Hagrid or Dumbledore today, but not both.
3. Harry visited Dumbledore today.
   
基于这三句句子(sentence)，我们就可以回答"did it rain toay?"。即使这三句句子都没有直接告诉我们今天是否下雨，但我们依然可以通过逻辑推理得出答案。

<br>

## **概念引入**

### **Sentence**

> A sentence is an assertion about the world in a knowledge representation language.

例如，$P=true$是一个语句(sentence)

$Q=false$也是一个语句

一般而言，语句承载了知识，或者说，语句就是命题赋了值。

<br>

### **Model**

> The model is an assignment of a truth value to every proposition. 

当“石头”一词出现在我们脑海中时，它先是一个概念，具有一些属性，比如形状，体积，质量，这些属性是未知的，不定的。而当我们看到现实中的石头时，至少在这一时刻，这块石头的形状，体积，质量是已知的，一定的，它就是“石头”这个概念在现实世界的一个模型(model)。

比如，有命题P，Q，命题本身就是概念，它们的真假是未知的，不定的，而当命题的真假一旦确定，它们就成为了现实的模型。

$\{P=true, Q=false\}$是一个模型。

$\{P=false, Q=false\}$也是一个模型。


<br>

### **Knowledge Base (KB)**

> The knowledge base is a set of sentences known by a knowledge-based agent.

可以简单理解为KB就是已知条件。

<br>

### **Entailment (⊨)**

> If α ⊨ β (α entails β), then in any world where α is true, β is true, too.

<br>

### **Inference**

> Inference is the process of deriving new sentences from old ones.

逻辑推理是常用的推理方法，也是本文主要的推理方法。

<br>

## **Model Checking Algorithm**
> A way to infer new knowledge based on existing knowledge.

* To determine if KB ⊨ α (in other words, answering the question: “can we conclude that α is true based on our knowledge base”)
   * Enumerate all possible models.
   * If in every model where KB is true, α is true as well, then KB entails α (KB ⊨ α).

简单地说，模型检查(Model Checking)就是遍历所有可能的模型，首先找到那些使得我们的知识库(KB)为真的模型，在这些模型中再检查$\alpha$是否为真。

eg:
![](https://ftp.bmp.ovh/imgs/2022/02/9bd5d6c661cc4611.png)
<!-- ![](https://raw.githubusercontent.com/cadake/PicGo/main/blog/pictures/20220222165519.png) -->
<!-- $$
P: It \, is\, a\, Tuesday.    \newline
Q: It \, is\, raining. \newline
R: Harry\, will\, go\, for\, a\, run. \newline
Knowledge\, Base(KB): ((P \land \lnot Q) \rightarrow R) \land P \land \lnot Q \newline
Query: R \newline
$$ -->

首先枚举所有可能的模型

P|Q|R|KB
-|-|-|-
true|true|true|
true|true|false|
true|false|true|
true|false|false|
false|true|true|
false|true|false|
false|false|true|
false|false|false|

所有使得KB为真的P，Q，R的组合的行的KB项为true，即需要满足P真，Q假，且有$(P \land \lnot Q) \rightarrow R$，于是填表得

P|Q|R|KB
-|-|-|-
true|true|true|false
true|true|false|false
true|false|true|<b>true<b>|
true|false|false|false
false|true|true|false
false|true|false|false
false|false|true|false
false|false|false|false

在这张表中，使得知识库为真的模型只有一个，并且在这个模型中R也为真。如果在所有使得知识库为真的模型中，R也都为真，那么有$KB ⊨ R$。

于是在这个例子中，我们通过模型检查算法从已有的知识库KB推导出了新的知识R。

<br>

**模型检查(Model Checking)显然是一种低效的算法，因为它枚举了所有的可能性。**

<br>

## **Knowledge and Search Problems**

通过如下定义，推理过程可以转化为搜索问题：

* **Initial state:** starting knowledge base
* **Actions:** inference rules
* **Transition model:** new knowledge base after inference
* **Goal test:** checking whether the statement that we are trying to prove is in the KB
* **Path cost function:** the number of steps in the proof

搜索算法能够使我们通过推理规则从已有的知识库诞生新的知识。

eg:

* To determine if KB ⊨ α:
   * Convert (KB ∧ ¬α) to Conjunctive Normal Form.
   * Keep checking to see if we can use resolution to produce a new clause.
   * If we ever produce the empty clause (equivalent to False), congratulations! We have arrived at a contradiction, thus proving that KB ⊨ α.
   * However, if contradiction is not achieved and no more clauses can be inferred, there is no entailment.

<br>
<br>

## **结语**

原课程地址
[https://cs50.harvard.edu/ai/2020/weeks/1/](https://cs50.harvard.edu/ai/2020/weeks/1/)