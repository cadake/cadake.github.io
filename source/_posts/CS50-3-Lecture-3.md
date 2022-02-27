---
title: CS50.3 Lecture 3
author: cadake
top: true
hide: false
cover: false
toc: true
mathjax: true
date: 2022-02-26 10:09:50
img:
password:
summary:
tags:
    - AI
    - algorithm
categories: 课程
---

# **Optimization**

> Optimization is choosing the best option from a set of possible options.

<br>

## **Local Search**

> Local search is a search algorithm that maintains a single node ands searches by moving to a neighboring node.

通常，本地搜索能够得到一个足够好，但未必是最好的答案。

<br>

考虑这样一个问题，现在在地图上有四所随机分布的房子，我们希望在地图上键两所医院，使得每一所房子到离它最近医院的距离的总和最小：

![](https://s3.bmp.ovh/imgs/2022/02/eeafb542d9dc9c8b.png)

我们通过曼哈顿距离(Manhattan distance)来定义房子和医院之间的距离。状态(state)即位置确定下的房子和医院的一个配置，花费(cost)即距离总和。我们试图找到一个最优的状态，使得cost最小。


我们将问题抽象为下图的状态空间(state-space landscape)：
![](https://s3.bmp.ovh/imgs/2022/02/099cedae440e3cee.png)
每一个长条代表一个状态，长条的高度代表该状态的价值(value)，在上面的房子和医院的例子中，cost越小，value越大，长条越高。

**术语引入：**
* An **Objective Function** is a function that we use to maximize the value of the solution.
* A **Cost Function** is a function that we use to minimize the cost of the solution.
* A **Current State** is the state that is currently being considered by the function.
* A **Neighbor State** is a state that the current state can transition to. 

本地搜索(Local search)的工作原理就是仅观察邻居节点(neighbor node)，从当前状态节点转移到邻居节点，而不必考虑整个状态空间。

<br>

### **Hill Climbing**

> Hill climbing is one type of a local search algorithm. In this algorithm, the neighbor states are compared to the current state, and if any of them is better, we change the current node from the current state to that neighbor state. What qualifies as better is defined by whether we use an objective function, preferring a higher value, or a decreasing function, preferring a lower value.

爬山算法(Hill climbing)伪代码描述如下:

function Hill-Climb(problem):

* current = initial state of problem
* repeat:
   * neighbor = best valued neighbor of current
   * if neighbor not better than current:
       * return current
   * current = neighbor

<br>

如图，已知房子的位置，随机选取医院的位置作为初始状态(initial state)

![](https://gitee.com/cadake/picgo/raw/master/img/20220226171906.png)

查看当前状态节点的邻居节点

![](https://gitee.com/cadake/picgo/raw/master/img/20220226172000.png)

如果有符合状态的邻居节点，状态转移

![](https://gitee.com/cadake/picgo/raw/master/img/20220226172104.png)

重复上述步骤，直到没有邻居节点符合条件

![](https://gitee.com/cadake/picgo/raw/master/img/20220226172155.png)

爬山算法只给出了一个足够好的答案，但显然下图的方案会更好，然而爬山算法在中途就停止了

![](https://gitee.com/cadake/picgo/raw/master/img/20220226172217.png)

<br>

### **Local and Global Minima and Maxima**

虽然全局最小值和全局最大值是我们想要的，但爬山算法常常只能得到局部最小值或局部最大值。

![](https://gitee.com/cadake/picgo/raw/master/img/20220226175400.png)

![](https://gitee.com/cadake/picgo/raw/master/img/20220226175416.png)

<br>

### **Hill Climbing Variants**

* **Steepest-ascent:** choose the highest-valued neighbor. This is the standard variation that we discussed above.
* **Stochastic**: choose randomly from higher-valued neighbors. 
* **First-choice**: choose the first higher-valued neighbor.
* **Random-restart**: conduct hill climbing multiple times. Each time, start from a random state. Compare the maxima from every trial, and choose the highest amongst those.
* **Local Beam Search:** chooses the k highest-valued neighbors. This is unlike most local search algorithms in that it uses multiple nodes for the search, and not just one.

通常来讲，搜索所有的状态是不现实的，对于本地搜索(local search)来说，"good enough is enough"。然而，爬山算法及其变种都容易陷入局部最小值或局部最大值。

<br>

### **Simulated Annealing**

> Annealing is the process of heating metal and allowing it to cool slowly, which serves to toughen the metal. This is used as a metaphor for the simulated annealing algorithm, which starts with a high temperature, being more likely to make random decisions, and, as the temperature decreases, it becomes less likely to make random decisions, becoming more “firm.” This mechanism allows the algorithm to change its state to a neighbor that’s worse than the current state, which is how it can escape from local maxima. 

伪代码描述如下：

function Simulated-Annealing(problem, max):
* current = initial state of problem
* for t = 1 to max:
   * T = Temperature(t)
   * neighbor = random neighbor of current
   * ΔE = how much better neighbor is than current
   * if ΔE > 0:
      * current = neighbor
   * with probability e^(ΔE/T) set current = neighbor
* return current

1. max代表希望执行的循环次数。
2. 温度(T)由一个温度函数(Temperature(t))给出，T随着t的增大而减小，意味着随时间的增加温度逐渐降低。
3. 随机选取当前节点的一个邻居节点。
4. ΔE描述邻居节点和当前节点值(value)的差值，ΔE>0说明比当前节点好，ΔE<0说明比当前节点差。
5. for循环中的最后一个语句使得当邻居节点在比当前节点差的情况下也依然有可能被选择到，这个可能性受ΔE和T所影响。simulated annealing算法希望这种带有风险的投资能给自己带来更大的收益。

<br>

## **Linear Programming**

> Linear programming is a family of problems that optimize a linear equation (an equation of the form y = ax₁ + bx₂ + …).

Linear programming will have the following components:

* A **cost function** that we want to minimize: c₁x₁ + c₂x₂ + … + cₙxₙ. Here, each x₋ is a variable and it is associated with some cost c₋.
* A **constraint** that’s represented as a sum of variables that is either less than or equal to a value (a₁x₁ + a₂x₂ + … + aₙxₙ ≤ b) or precisely equal to this value (a₁x₁ + a₂x₂ + … + aₙxₙ = b). In this case, x₋ is a variable, and a₋ is some resource associated with it, and b is how much resources we can dedicate to this problem.
* **Individual bounds** on variables (for example, that a variable can’t be negative) of the form lᵢ ≤ xᵢ ≤ uᵢ.

<br>

## **Constraint Satisfaction**
> Constraint Satisfaction problems are a class of problems where variables need to be assigned values while satisfying some conditions.

Constraints satisfaction problems have the following properties:

* Set of variables (x₁, x₂, …, xₙ)
* Set of domains for each variable {D₁, D₂, …, Dₙ}
* Set of constraints C

<br>

数独可以被抽象为一个constraint satisfaction问题
* 每一个空格是一个variable
* 空格的domain是数字1-9
* constraints是数独的规则，例如同一个大正方形内不能有重复的数字，整个数独大正方形的一列或一行不能有重复的数字。

<br>

现在考虑另一个问题。有四名学生，每名学生都选了A,B,...,G中的三节课，每一门课都有一次考试，这些考试需要集中安排在周一，周二和周三。我们希望每名考生每天只需要参加一门考试。

在这个问题中，
* variables是这些课程
* domain是可以安排考试的时间的集合
* constraints是某些课程的考试不能安排在同一天，因为我们不希望同时选了这两门课的学生在同一天参加两场考试。

![](https://gitee.com/cadake/picgo/raw/master/img/20220227093944.png)

将问题抽象为无向图，节点表示课程，节点间的边代表这两门课被同一个学生选择，那么它们不能被安排在同一天举行考试。
现在需要为节点赋值，即将节点表示的课程安排在某天举行考试，且邻居节点不能有相同的值。

![](https://gitee.com/cadake/picgo/raw/master/img/20220227094050.png)

<br>

为了解决constraint satisfaction问题，我们需要引入一些额外的概念：
* A **Hard Constraint** is a constraint that must be satisfied in a correct solution.
* A **Soft Constraint** is a constraint that expresses which solution is preferred over others.
* A **Unary Constraint** is a constraint that involves only one variable. In our example, a unary constraint would be saying that course A can’t have an exam on Monday {A ≠ Monday}.
* A **Binary Constraint** is a constraint that involves two variables. This is the type of constraint that we used in the example above, saying that some two courses can’t have the same value {A ≠ B}.

<br>

### **Node Consistency**

> Node consistency is when all the values in a variable’s domain satisfy the variable’s unary constraints.

<br>

### **Arc Consistency**

> Arc consistency is when all the values in a variable’s domain satisfy the variable’s binary constraints. In other words, to make X arc-consistent with respect to Y, remove elements from X’s domain until every choice for X has a possible choice for Y.

使两个节点满足arc consistency的伪代码表述如下：

function Revise(csp, X, Y):

* revised = false
* for x in X.domain:
   * if no y in Y.domain satisfies constraint for (X,Y):
      * delete x from X.domain
      * revised = true
* return revised

<br>

使整个问题的所有节点满足arc consistency的伪代码表述如下：
function AC-3(csp):

* queue = all arcs in csp
* while queue non-empty:
   * (X, Y) = Dequeue(queue)
   * if Revise(csp, X, Y):
      * if size of X.domain == 0:
         * return false
      * for each Z in X.neighbors - {Y}:
         * Enqueue(queue, (Z,X))
* return true

在这个算法中，如果Revise函数返回值为真，说明X通过缩小domain的方式与Y达成了arc consistency。如果X的domain最终缩小为$\emptyset$，说明X和Y无法满足arc consistency，问题无解。否则，X的domain缩小可能导致X与X的其他邻居节点的arc consistency发生变化，因此将它们入队列，在以后的循环中二次判断。

constraint satisfaction问题可以被转化为一个搜索问题：

* **Initial state:** empty assignment (all variables don’t have any values assigned to them).
* **Actions:** add a {variable = value} to assignment; that is, give some variable a value.
* **Transition model:** shows how adding the assignment changes the assignment.
* **Goal test:** check if all variables are assigned a value and all constraints are satisfied.

<br>

### **Backtracking Serach**

> Backtracking search is a type of a search algorithm that takes into account the structure of a constraint satisfaction search problem. In general, it is a recursive function that attempts to continue assigning values as long as they satisfy the constraints.

伪代码如下：

function Backtrack(assignment, csp):

* if assignment complete:
   * return assignment
* var = Select-Unassigned-Var(assignment, csp)
* for value in Domain-Values(var, assignment, csp):
  * if value consistent with assignment:
      * add {var = value} to assignment
      * result = Backtrack(assignment, csp)
      * if result ≠ failure:
         * return result
      * remove {var = value} from assignment
* return failure

初始状态下，所有节点都未分配值
![](https://gitee.com/cadake/picgo/raw/master/img/20220227101900.png)

随机选择一个未分配值的节点，我们从A节点开始，从A的domain中顺序选择一个值赋予A，这里是Mon
![](https://gitee.com/cadake/picgo/raw/master/img/20220227101918.png)

递归调用回溯算法，随机选择一个未分配值的节点，我们选到了B，从B的domain中顺序选择一个值Mon赋予B。B的取值违背了(A, B)的约束条件
![](https://gitee.com/cadake/picgo/raw/master/img/20220227101955.png)

顺序走到B的domain中的下一个值Tue，赋值给B
![](https://gitee.com/cadake/picgo/raw/master/img/20220227102712.png)

重复上述步骤，如遇到下图的情况，节点C将没有值可以选择
![](https://gitee.com/cadake/picgo/raw/master/img/20220227102825.png)

逐步回溯，E也没有其他选择，继续回溯
![](https://gitee.com/cadake/picgo/raw/master/img/20220227103009.png)

回溯至D，D的domain为{Tue, Wed}
![](https://gitee.com/cadake/picgo/raw/master/img/20220227103118.png)

回溯至D说明D选择Mon走不通，顺序选择domain的下一个值Tue，赋值给D，但是违反了(B, D)间的约束条件，继续选择Wed赋值给D，递归调用回溯算法向下搜索
![](https://gitee.com/cadake/picgo/raw/master/img/20220227103433.png)

最终得到一个解
![](https://gitee.com/cadake/picgo/raw/master/img/20220227103525.png)

<br>

### **Inference**

backtracking并不高效，我们希望随着搜索的逐渐深入，信息逐渐增多，我们可以利用这些信息，推断出更好的搜索方向。

<br>

#### **Maintaining Arc-Consistency**

> This algorithm will enforce arc-consistency after every new assignment of the backtracking search. Specifically, after we make a new assignment to X, we will call the AC-3 algorithm and start it with a queue of all arcs (Y,X) where Y is a neighbor of X (and not a queue of all arcs in the problem). 

伪代码描述如下：

function Backtrack(assignment, csp):

* if assignment complete:
   * return assignment
* var = Select-Unassigned-Var(assignment, csp)
* for value in Domain-Values(var, assignment, csp):
   * if value consistent with assignment:
      * **add {var = value} to assignment**
      * **inferences = Inference(assignment, csp)**
      * if inferences ≠ failure:
         * add inferences to assignment
      * result = Backtrack(assignment, csp)
      * if result ≠ failure:
         * return result
      * remove {var = value} **and inferences** from assignment
* return failure

当我们处于下图的状态时，我们希望C和它的邻居节点A，B保持arc consistency
![](https://gitee.com/cadake/picgo/raw/master/img/20220227105101.png)

最终C的domain会缩小
![](https://gitee.com/cadake/picgo/raw/master/img/20220227105217.png)

C的邻居节点E的domain也会缩小
![](https://gitee.com/cadake/picgo/raw/master/img/20220227105241.png)

最终，我们维护了整个问题的arc consistency
![](https://gitee.com/cadake/picgo/raw/master/img/20220227105338.png)

<br>

### **Select Unassigned Var**

随机选择未赋值的节点显然不够高效，我们希望利用已有的信息，通过启发式(heuristic)搜索选择节点。

<br>

#### **Minimum Remaining Values(MRV)**

> The idea here is that if a variable’s domain was constricted by inference, and now it has only one value left (or even if it’s two values), then by making this assignment we will reduce the number of backtracks we might need to do later. 

![](https://gitee.com/cadake/picgo/raw/master/img/20220227110553.png)

For example, after having narrowed down the domains of variables given the current assignment, using the MRV heuristic, we will choose variable C next and assign the value Wednesday to it.

<br>

#### **Degree heuristic**

> The Degree heuristic relies on the degrees of variables, where a degree is how many arcs connect a variable to other variables. By choosing the variable with the highest degree, with one assignment, we constrain multiple other variables, speeding the algorithm’s process.

![](https://gitee.com/cadake/picgo/raw/master/img/20220227110540.png)

For example, all the variables above have domains of the same size. Thus, we should pick a domain with the highest degree, which would be variable E.

<br>

#### **Least Constraining Values heuristic**

> The idea here is that, while in the degree heuristic we wanted to use the variable that is more likely to constrain other variables, here we want this variable to place the least constraints on other variables. That is, we want to locate what could be the largest potential source of trouble (the variable with the highest degree), and then render it the least troublesome that we can (assign the least constraining value to it).

![](https://gitee.com/cadake/picgo/raw/master/img/20220227110458.png)

For example, let’s consider variable C. If we assign Tuesday to it, we will put a constraint on all of B, D, E, and F. However, if we choose Wednesday, we will put a constraint only on B, D, and E. Therefore, it is probably better to go with Wednesday.

<br>

## **结语**

原课程地址
[https://cs50.harvard.edu/ai/2020/weeks/2/](https://cs50.harvard.edu/ai/2020/weeks/2/)