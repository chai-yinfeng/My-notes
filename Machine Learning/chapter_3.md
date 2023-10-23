# 第3章 线性模型

# 3.1 基本形式

机器学习中的线性模型是一种用线性方程来描述数据之间关系的模型。具体来说，给定一组输入特征，线性模型试图预测一个或多个输出，这些预测是输入特征的线性组合。

线性模型的基本形式是这样的：
$$
y=w_{1} x_{1}+w_{2} x_{2}+...+w_{n} x_{n}+b
$$
其中：

- $y$ 是预测值。
- $x_1,x_2,...,x_n$是输入特征。
- $w_1,w_2,...,w_n$是模型的权重。
- $b$ 是偏置项。

以下是线性模型的一些常见类型：

1. **线性回归 (Linear Regression)**: 这可能是最简单和最广泛使用的线性模型。它尝试找到输入特征与连续输出值之间的最佳线性关系。常用于预测数值型的输出。
2. **逻辑回归 (Logistic Regression)**: 尽管名字中有“回归”，但逻辑回归实际上是用于分类任务的。它预测某一类的概率，通常使用sigmoid函数将线性组合的输出映射到[0,1]范围内。
3. **线性判别分析 (Linear Discriminant Analysis, LDA)**: 这是一种分类技术，它不仅关注分类，还考虑了类别间和类别内的方差，以获得最佳的分类。
4. **感知机 (Perceptron)**: 这是一个二分类算法，它的工作原理是调整权重，直到所有数据点都被正确分类或达到最大迭代次数。

线性模型的优点包括简单、易于理解、计算效率高和可解释性强。然而，它们的局限性在于不能捕获复杂的非线性关系，所以在某些任务中可能不如其他复杂模型表现得好。



# 3.2 线性回归

## 一元线性回归

参考课程链接：【【吃瓜教程】《机器学习公式详解》（南瓜书）与西瓜书公式推导直播合集】 https://www.bilibili.com/video/BV1Mh411e7VU/?p=3&share_source=copy_web&vd_source=c22abe8e67e193936015d5ca043a8148

一元线性回归是线性回归的一种特殊情况，它只涉及一个输入特征和一个输出。其目的是找到描述输入和输出之间关系的最佳直线。一元线性回归的模型表示为：
$$
y=wx+b
$$
其中：

- $y$ 是预测值
- $x$ 是输入特征
- $w$ 是权重（也称为斜率）
- $b$ 是偏置（也称为截距）

### ！！公式推导！！

公式推导部分内容的符号和记法参照西瓜书和南瓜书中的内容。

线性回归试图学得：
$$
f(x_i)=wx_i+b,使得f(x_i)\simeq y_i
$$
均方误差是回归任务中最常用的性能度量，因此我们可试图让均方误差最小化，即：
$$
(w^*,b^*)=arg\min_{(w,b)}\sum_{i=1}^{m}(f(x_i)-y_i)^2
=arg\min_{(w,b)}\sum_{i=1}^m(y_i-wx_i-b)^2
$$
基于均方误差最小化来进行模型求解的方法称为“最小二乘法”（least square method）。在线性回归中，最小二乘法就是试图找到一条直线，是所有样本到直线上的欧氏距离值和最小。

除了最小二乘法以外，还可以通过极大似然估计的方式来得到模型求解的方法。

对于线性回归来说，可以假设其为以下模型：
$$
y=wx+b+\epsilon
$$
其中 ϵ 为不受控制的随机误差，通常假设其服从均值为0的正态分布 $\epsilon∼N(0,σ^2) $，所以 $\epsilon$ 的概率密度函数为：
$$
p(\epsilon)=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{\epsilon^2}{2\sigma^2})
$$
若将 $\epsilon $ 用 $ y-(wx+b) $ 等价替换可得：
$$
p(\epsilon)=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y-(wx+b))^2}{2\sigma^2})
$$
上式显然可以看做 $y∼N(wx+b,σ^2)$，下面便可以用极大似然估计来估计w和b的值，似然函数为：
$$
L(w,b)=\prod_{i=1}^mp(y_i)=\prod_{i=1}^m\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y_i-(wx_i+b))^2}{2\sigma^2})\\
\ln L(w,b)=\sum_{i=1}^m\ln\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y_i-wx_i-b)^2}{2\sigma^2})\\
=\sum_{i=1}^m\ln\frac{1}{\sqrt{2\pi}\sigma}+\sum_{i=1}^m\ln \exp(-\frac{(y_i-wx_i-b)^2}{2\sigma^2})\\
=m\ln\frac{1}{\sqrt{2\pi}\sigma}-\frac{1}{2\sigma^2}\sum_{i=1}^m(y_i-wx_i-b)^2
$$
其中 $m,σ$ 均为常数
$$
所以最大化\ln L(w,b)等价于最小化\sum_{i=1}^m(y_i-wx_i-b)^2,也即\\
(w^*,b^*)=arg \max_{(w,b)}\ln L(w,b)=arg\min_{(w,b)}\sum_{i=1}^m(y_i-wx_i-b)^2
$$
等价于最小二乘估计。

将均方误差分别对$w$和$b$求导可以得到：
$$
\frac{\partial E_{(w,b)}}{\partial w}=2(w\sum_{i=1}^mx_i^2-\sum_{i=1}^m(y_i-b)x_i)\\
\frac{\partial E_{(w,b)}}{\partial b}=2(mb-\sum_{i=1}^m(y_i-wx_i))
$$
以下为推导过程：
$$
已知E_{(w,b)}=\sum_{i=1}^m(y_i-wx_i-b)^2,所以\\
\frac{\partial E_{(w,b)}}{\partial w}=\frac{\partial}{\partial w}[\sum_{i=1}^m(y_i-wx_i-b)^2]\\
=\sum_{i=1}^m\frac{\partial}{\partial w}[(y_i-wx_i-b)^2]\\
=\sum_{i=1}^m[2(y_i-wx_i-b)·(-x_i)]\\
=\sum_{i=1}^m[2(wx_i^2-y_ix_i+bx_i)]\\
=2(w\sum_{i=1}^mx_i^2-\sum_{i=1}^my_ix_i+b\sum_{i=1}^mx_i)\\
=2[w\sum_{i=1}^mx_i^2-\sum_{i=1}^m(y_i-b)x_i]
$$

$$
已知E_{(w,b)}=\sum_{i=1}^m(y_i-wx_i-b)^2,所以\\
\frac{\partial E_{(w,b)}}{\partial b}=\frac{\partial}{\partial b}[\sum_{i=1}^m(y_i-wx_i-b)^2]\\
=\sum_{i=1}^m\frac{\partial}{\partial b}[(y_i-wx_i-b)^2]\\
=\sum_{i=1}^m[2(y_i-wx_i-b)·(-1)]\\
=\sum_{i=1}^m[2(b-y_i+wx_i)]\\
=2(\sum_{i=1}^mb-\sum_{i=1}^my_i+\sum_{i=1}^mwx_i)\\
=2[mb-\sum_{i=1}^m(y_i-wx_i)]
$$

分别令二式为零可以得到w和b最优解的闭式解：
$$
w=\frac{\sum_{i=1}^my_i(x_i-\bar x)}{\sum_{i=1}^mx_i^2-\frac{1}{m}(\sum_{i=1}^mx_i^2)}\\
b=\frac{1}{m}\sum_{i=1}^m(y_i-wx_i)\\
其中\bar x=\frac{1}{m}\sum_{i=1}^mx_i为x的均值
$$
以下为推导过程：
$$
令\frac{\partial E_{(w,b)}}{\partial w}=2(w\sum_{i=1}^mx_i^2-\sum_{i=1}^m(y_i-b)x_i)等于0,有\\
0=w\sum_{i=1}^mx_i^2-\sum_{i=1}^m(y_i-b)x_i\\
w\sum_{i=1}^mx_i^2=\sum_{i=1}^my_ix_i-\sum_{i=1}^mbx_i
$$

$$
令\frac{\partial E_{(w,b)}}{\partial b}=2(mb-\sum_{i=1}^m(y_i-wx_i))等于0,可得
b=\frac{1}{m}\sum_{i=1}^m(y_i-wx_i),又因为\frac{1}{m}\sum_{i=1}^my_i=\bar y且\\
\frac{1}{m}\sum_{i=1}^mx_i=\bar x,则b=\bar y-w\bar x,代入上式可得\\
w\sum_{i=1}^mx_i^2=\sum_{i=1}^my_ix_i-\sum_{i=1}^m(\bar y-w\bar x)x_i\\
w\sum_{i=1}^mx_i^2=\sum_{i=1}^my_ix_i-\bar y\sum_{i=1}^mx_i+w\bar x\sum_{i=1}^mx_i\\
w(\sum_{i=1}^mx_i^2-\bar x\sum_{i=1}^mx_i)=\sum_{i=1}^my_ix_i-\bar y\sum_{i=1}^mx_i\\
w=\frac{\sum_{i=1}^my_ix_i-\bar y\sum_{i=1}^mx_i}{\sum_{i=1}^mx_i^2-\bar x\sum_{i=1}^mx_i}\\
将\bar y \sum_{i=1}^mx_i=\frac{1}{m}\sum_{i=1}^my_i\sum_{i=1}^mx_i=\bar x\sum_{i=1}^my_i和
\bar x\sum_{i=1}^mx_i=\frac{1}{m}\sum_{i=1}^mx_i\sum_{i=1}^mx_i=\\
\frac{1}{m}(\sum_{i=1}^mx_i)^2代入上式，即可得式:\\
w=\frac{\sum_{i=1}^my_i(x_i-\bar x)}{\sum_{i=1}^mx_i^2-\frac{1}{m}(\sum_{i=1}^mx_i^2)}
$$

南瓜书中还有将w的表达式向量化的过程，便于在Python中使用Numpy等专门加速矩阵运算的类库进行编写，详情可查看推导过程。

### **算法原理**

一元线性回归的目标是最小化实际输出值 *y* 和预测输出值 *f(x)* 之间的平均平方误差（Mean Squared Error, MSE）。数学表示为：
$$
MSE=\frac{1}{m}\sum_{i=1}^{m}(f( x_{i} )-y_{i})^{2}
$$
其中 *m* 是数据点的数量。

为了最小化MSE，我们需要调整权重 *w* 和偏置 *b*。这通常通过梯度下降算法来实现。

**梯度下降**：

1. 初始化权重 *w* 和偏置 *b* 的值（通常为随机值）。
2. 使用当前的 *w* 和 *b* 计算所有数据点的预测值 *f(x)*。
3. 根据预测值计算MSE。
4. 计算MSE关于 *w* 和 *b* 的梯度。
5. 调整 *w* 和 *b* 的值，使其沿着梯度的反方向移动一个小步。
6. 重复步骤2-5，直到MSE达到一个足够小的值或满足其他终止条件。

在一元线性回归中，MSE关于 *w* 和 *b* 的梯度可以直接计算：
$$
\frac{\partial MSE}{\partial w} =\frac{2}{m}\sum_{i=1}^{m}x_{i}(f( x_{i} )-y_{i})
$$

$$
\frac{\partial MSE}{\partial b} =\frac{2}{m}\sum_{i=1}^{m}(f( x_{i} )-y_{i})
$$

梯度下降算法将使用这些梯度值来更新 *w* 和 *b*，以达到最小化MSE的目标。

注意：对于一元线性回归，还有一个解析解，称为“正规方程”或“最小二乘法”，它可以直接找到权重 *w* 和偏置 *b*，而无需迭代地使用梯度下降。

### 补充：正交回归

正交回归（也称为主轴回归或等距回归）是线性回归的一个变种，用于在两个变量都存在测量误差的情境下估计它们之间的关系。传统的线性回归（如最小二乘法）假设自变量（或解释变量）是无误差的，而只有因变量（或响应变量）存在误差。但在某些实际应用中，这个假设可能不成立。例如，在某些科学实验中，两个被测量的变量都可能受到仪器误差的影响。

正交回归的主要思想是寻找一条直线（或一个超平面，对于多维的情况），使得数据点到这条直线的垂直距离的平方和最小。

数学上，考虑两个变量X和Y。传统的最小二乘回归尝试找到一条直线使得垂直于Y轴（因变量轴）的距离的平方和最小。而正交回归则尝试找到一条直线，使得数据点到这条直线的垂直距离的平方和最小。

从几何的角度看，正交回归考虑了数据点到回归线的直线距离，而不仅仅是垂直距离。

要执行正交回归，通常需要使用迭代方法或优化技术，因为它涉及到的数学不如传统的最小二乘回归那么简单。正交回归在某些应用中很有价值，尤其是当我们有理由相信解释变量和响应变量都受到测量误差的影响时。

### 最小二乘估计与极大似然估计

线性回归的最小二乘估计（Least Squares Estimation，LSE）和极大似然估计（Maximum Likelihood Estimation，MLE）都是用来估计线性模型参数的方法。尽管它们背后的思想和方法不同，但在简单线性回归的情境下，它们实际上可以得到相同的结果。

1. **最小二乘估计 (LSE)**:

   这是线性回归中最常用的参数估计方法。其核心思想是最小化实际输出值与预测输出值之间的平方差。

   给定一个线性模型 
   $$
   y=Xβ+ϵ
   $$
   其中 *y* 是输出向量，*X* 是输入矩阵，β 是参数向量，ϵ 是误差项，LSE的目标是找到参数 *β* 使得 *ϵ**T**ϵ* 最小。换句话说，LSE试图最小化输出的预测值和实际值之间的总平方误差。

   对于一元线性回归，LSE可以得到一个封闭形式的解，即正规方程。

2. **极大似然估计 (MLE)**:

   MLE的目标是找到参数估计值，使得给定这些参数下，观测到实际数据的可能性（似然）最大。从统计学的角度来说，我们通常假设误差项 *ϵ* 是独立同分布的，并遵循正态分布，即 ϵ*∼*N*(0,*σ*2)。

   在这种情况下，线性回归模型的似然函数为：
   $$
   L(\beta \mid y,X)=\prod_{i=1}^{n}\frac{1}{\sqrt{2 \pi \sigma^{2} } }  exp(-\frac{(y_{i}-x_{i}^{T}\beta )^{2}}{2 \sigma^{2}} )
   $$
   对似然函数取对数，我们得到对数似然函数。然后，我们可以最大化对数似然函数来得到 *β* 的MLE。

在线性回归的简单场景下（即误差项为正态分布），LSE和MLE实际上是等价的，即它们会得到相同的参数估计值。然而，这两种方法背后的思想和解释是不同的，LSE侧重于最小化预测误差，而MLE侧重于最大化数据的似然。

### 补充：凸函数

凸函数是一种特殊类型的函数，其图形总是位于其任意两点之间的弦的上方或与之重合。更正式地说，一个函数 *f* : R*n*→R 是凸的，如果对于其定义域内的任意两点 *x* 和 *y* 以及任意 *λ* 满足 0≤*λ*≤1，以下不等式成立：
$$
f(λx+(1−λ)y)≤λf(x)+(1−λ)f(y)
$$
直观地，这意味着函数的图形在任意两点之间的线段上或以下。对于一维函数，你可以想象它为“碗”的形状，而不是“鞍”的形状。

凸函数的一个重要特性是：它的局部最小值也是全局最小值。

### 机器学习三要素

1. **模型（Model）**：

   模型定义了输入与输出之间的关系。它可以是一个简单的线性模型、一个决策树、一个神经网络或任何其他数学结构，用于描述或预测数据。模型的选择通常取决于问题的性质、数据的特征以及特定任务的需求。

2. **目标函数（Objective Function，也称为损失函数或代价函数）**：

   目标函数度量模型的预测值与实际值之间的差异。在监督学习中，目标函数常常是用来评估模型表现的关键指标，如均方误差（对于回归任务）或交叉熵损失（对于分类任务）。机器学习的核心任务之一就是优化（通常是最小化）这个函数。

3. **优化算法（Optimization Algorithm）**：

   优化算法定义了如何更新模型的参数以改进其性能，即减少目标函数的值。常见的优化算法包括梯度下降、随机梯度下降、牛顿法、L-BFGS等。这些算法通过不同的方式调整模型的参数，以寻找目标函数的最小值（或最大值，取决于问题）。

## 多元线性回归

参考课程链接：【【吃瓜教程】《机器学习公式详解》（南瓜书）与西瓜书公式推导直播合集】 https://www.bilibili.com/video/BV1Mh411e7VU/?p=4&share_source=copy_web&vd_source=c22abe8e67e193936015d5ca043a8148

### ！！公式推导！！

公式推导部分内容的符号和记法参照西瓜书和南瓜书中的内容。
$$
更一般的情形是数据集D=\{(\boldsymbol x_1,y_1),(\boldsymbol x_2,y_2),...,(\boldsymbol x_m,y_m)\},\\
其中\boldsymbol x_i=(x_{i1};x_{i2};...;x_{id}),y_i\in\mathbb R,样本由d个属性描述.
此时我们试图学得\\
f(\boldsymbol x_i)=\boldsymbol w^T\boldsymbol x_i+b,使得f(\boldsymbol x_i)\simeq y_i\\
f(\boldsymbol x_i)=
\begin{pmatrix}
w_1 & w_2 & ... & w_d
\end{pmatrix}
\begin{pmatrix}
 x_{i1}
 \\x_{i2}
 \\...
 \\x_{id}
\end{pmatrix}+b\\
f(\boldsymbol x_i)=w_1x_{i1}+w_2x_{i2}+...+w_dx_{id}+b\\
f(\boldsymbol x_i)=w_1x_{i1}+w_2x_{i2}+...+w_dx_{id}+w_{d+1}·1\\
f(\boldsymbol x_i)=
\begin{pmatrix}
w_1 & w_2 & ... & w_d & w_{d+1}
\end{pmatrix}
\begin{pmatrix}
 x_{i1}
 \\x_{i2}
 \\...
 \\x_{id}
 \\1
\end{pmatrix}\\
所以可得新的表示:f(\hat{\boldsymbol x_i} )=\hat{\boldsymbol w}^T\hat{\boldsymbol x_i}
$$
由最小二乘法可得：
$$
E_{\hat w}=\sum_{i=1}^m(y_i-f(\hat{\boldsymbol x_i}))^2=\sum_{i=1}^m(y_i-\hat{\boldsymbol w}^T\hat{\boldsymbol x_i})^2
$$
下面将其向量化以得到同西瓜书完全一致的形式：
$$
E_{\hat w}=\sum_{i=1}^m(y_i-\hat{\boldsymbol w}^T\hat{\boldsymbol x_i})^2=(y_1-\hat{\boldsymbol w}^T\hat{\boldsymbol x_1})^2+(y_2-\hat{\boldsymbol w}^T\hat{\boldsymbol x_2})^2+...+(y_m-\hat{\boldsymbol w}^T\hat{\boldsymbol x_m})^2\\
E_{\hat w}=
\begin{pmatrix}
y_1-\hat{\boldsymbol w}^T\hat{\boldsymbol x_1} & y_2-\hat{\boldsymbol w}^T\hat{\boldsymbol x_2} & ... & y_m-\hat{\boldsymbol w}^T \hat{\boldsymbol x_m}
\end{pmatrix}
\begin{pmatrix}
 y_1-\hat{\boldsymbol w}^T\hat{\boldsymbol x_1}
 \\y_2-\hat{\boldsymbol w}^T\hat{\boldsymbol x_2}
 \\...
 \\y_m-\hat{\boldsymbol w}^T\hat{\boldsymbol x_m}
\end{pmatrix}\\
其中\begin{pmatrix}
 y_1-\hat{\boldsymbol w}^T\hat{\boldsymbol x_1}
 \\y_2-\hat{\boldsymbol w}^T\hat{\boldsymbol x_2}
 \\...
 \\y_m-\hat{\boldsymbol w}^T\hat{\boldsymbol x_m}
\end{pmatrix}=\begin{pmatrix}
 y_1
 \\y_2
 \\...
 \\y_m
\end{pmatrix}-\begin{pmatrix}
 \hat{\boldsymbol w}^T\hat{\boldsymbol x_1}
 \\\hat{\boldsymbol w}^T\hat{\boldsymbol x_2}
 \\...
 \\\hat{\boldsymbol w}^T\hat{\boldsymbol x_m}
\end{pmatrix}=\begin{pmatrix}
 y_1
 \\y_2
 \\...
 \\y_m
\end{pmatrix}-\begin{pmatrix}
 \hat{\boldsymbol x_1}^T\hat{\boldsymbol w}
 \\\hat{\boldsymbol x_2}^T\hat{\boldsymbol w}
 \\...
 \\\hat{\boldsymbol x_m}^T\hat{\boldsymbol w}
\end{pmatrix}\\
\boldsymbol y=\begin{pmatrix}
 y_1
 \\y_2
 \\...
 \\y_m
\end{pmatrix},\begin{pmatrix}
 \hat{\boldsymbol x_1}^T\hat{\boldsymbol w}
 \\\hat{\boldsymbol x_2}^T\hat{\boldsymbol w}
 \\...
 \\\hat{\boldsymbol x_m}^T\hat{\boldsymbol w}
\end{pmatrix}=\begin{pmatrix}
 \hat{\boldsymbol x_1}^T
 \\\hat{\boldsymbol x_2}^T
 \\...
 \\\hat{\boldsymbol x_m}^T
\end{pmatrix}·\hat{\boldsymbol w}=\begin{pmatrix}
 \boldsymbol x_1^T & 1
 \\\boldsymbol x_2^T & 1
 \\...&...
 \\\boldsymbol x_m^T & 1
\end{pmatrix}·\hat{\boldsymbol w}=\boldsymbol X·\hat{\boldsymbol w}\\
所以有:\begin{pmatrix}
 y_1-\hat{\boldsymbol w}^T\hat{\boldsymbol x_1}
 \\y_2-\hat{\boldsymbol w}^T\hat{\boldsymbol x_2}
 \\...
 \\y_m-\hat{\boldsymbol w}^T\hat{\boldsymbol x_m}
\end{pmatrix}=\boldsymbol y-\boldsymbol X \hat{\boldsymbol w}\\
E_{\hat{\boldsymbol w}}=\begin{pmatrix}
y_1-\hat{\boldsymbol w}^T\hat{\boldsymbol x_1} & y_2-\hat{\boldsymbol w}^T\hat{\boldsymbol x_2} & ... & y_m-\hat{\boldsymbol w}^T \hat{\boldsymbol x_m}
\end{pmatrix}
\begin{pmatrix}
 y_1-\hat{\boldsymbol w}^T\hat{\boldsymbol x_1}
 \\y_2-\hat{\boldsymbol w}^T\hat{\boldsymbol x_2}
 \\...
 \\y_m-\hat{\boldsymbol w}^T\hat{\boldsymbol x_m}
\end{pmatrix}=(\boldsymbol y-\boldsymbol X\hat{\boldsymbol w})^T(\boldsymbol y-\boldsymbol X\hat{\boldsymbol w})
$$
对 $\hat w$ 求导可得
$$
\frac{\partial E_{\hat{\boldsymbol w}}}{\partial \hat{\boldsymbol w}}=2\boldsymbol X^T(\boldsymbol X\hat{\boldsymbol w}-\boldsymbol y)
$$
推导过程如下：
$$
将E_{\hat{\boldsymbol w}}=(\boldsymbol y-\boldsymbol X\hat{\boldsymbol w})^T(\boldsymbol y-\boldsymbol X\hat{\boldsymbol w})展开可得\\
E_{\hat{\boldsymbol w}}=\boldsymbol y^T \boldsymbol y-\boldsymbol y^T \boldsymbol X \hat{\boldsymbol w}-\hat{\boldsymbol w}\boldsymbol X^T\boldsymbol y+\hat{\boldsymbol w}^T\boldsymbol X^T\boldsymbol X\hat{\boldsymbol w}\\
对\hat{\boldsymbol w}求导可得\\
\frac{\partial E_{\hat{\boldsymbol w}}}{\partial \boldsymbol w}=\frac{\partial \boldsymbol y^T \boldsymbol y}{\partial \boldsymbol w}-\frac{\partial \boldsymbol{y^T X \hat w}}{\partial \boldsymbol w}-\frac{\partial \boldsymbol {\hat w^T X^T y}}{\partial \boldsymbol w}+\frac{\partial \boldsymbol {\hat w^T X^T X \hat w}}{\partial \boldsymbol w}\\
根据矩阵微分公式\frac{\partial \boldsymbol {x^T a}}{\partial \boldsymbol x}=\frac{\partial \boldsymbol {a^T x}}{\partial \boldsymbol x}=\boldsymbol a,\frac{\partial \boldsymbol {x^T A x}}{\partial \boldsymbol x}=(\boldsymbol A + \boldsymbol A^T) \boldsymbol x可得\\
\frac{\partial E_{\hat{\boldsymbol w}}}{\partial \boldsymbol w}=0-\boldsymbol {X^Ty}-\boldsymbol{X^Ty}+\boldsymbol{(X^TX+X^TX)}\hat{\boldsymbol w}\\
=2\boldsymbol{X^T(X\hat w-y)}\\
令\frac{\partial E_{\hat{\boldsymbol w}}}{\partial \boldsymbol w}=2\boldsymbol{X^T(X\hat w-y)}=0\\
2\boldsymbol{X^TX\hat w}-2\boldsymbol{X^T y}=0\\
2\boldsymbol{X^TX\hat w}=2\boldsymbol{X^T y}\\
\boldsymbol{\hat w}=\boldsymbol{(X^TX)^{-1}X^Ty}
$$






### 1. 定义

多元线性回归是线性回归的扩展，用于预测一个响应变量基于两个或更多的特征。其基本假设是所有这些特征与响应变量之间存在线性关系。

### 2. 数学模型

多元线性回归的数学表示为： 
$$
y=β_{0}+β_{1}x_{1}+β_{2}x_{2}+...+β_{n}x_{n}+ϵ
$$
其中：

- *y* 是响应变量。
- *x*1,*x*2,...*xn* 是解释变量或特征。
- *β*0,*β*1,...*βn* 是模型参数，其中 *β*0 是截距。
- *ϵ* 是误差项。

### 3. 参数估计

为了确定模型的参数（*β* 值），我们通常使用**最小二乘法**来最小化预测和真实值之间的平方误差之和。

### 4. 假设

多元线性回归有以下假设：

- 线性关系：解释变量与响应变量之间存在线性关系。
- 误差的独立性：观测值的误差是独立的。
- 同方差性：误差的方差在所有观测值中是恒定的。
- 无多重共线性：解释变量之间没有完全的线性关系。
- 误差的正态性：对于任何给定的解释变量值组合，误差是正态分布的。

### 5. 诊断和检验

为了确认模型的适用性，我们需要进行一系列的诊断检查。这包括：

- **残差分析**：检查残差（实际值与预测值之间的差异）是否满足正态分布、独立性和同方差性的假设。
- **方差膨胀因子（VIF）**：检查多重共线性。
- **t-测试和F-测试**：评估模型中的单个或所有变量是否对响应变量有显著影响。

### 6. 应用

多元线性回归在许多领域都有应用，包括经济学、生物学、工程学和社会科学等。例如，预测基于多个输入特征（如面积、楼层数、社区等）的房价。

### 7. 限制和挑战

尽管多元线性回归是一个强大的工具，但它也有其限制。数据中的非线性关系、相关性高的解释变量（多重共线性）、异方差和缺乏观测独立性都可能对模型的解释和预测能力产生影响。

# 3.3 对数几率回归

参考课程链接：【【吃瓜教程】《机器学习公式详解》（南瓜书）与西瓜书公式推导直播合集】 https://www.bilibili.com/video/BV1Mh411e7VU/?p=5&share_source=copy_web&vd_source=c22abe8e67e193936015d5ca043a8148

对数几率回归，通常称为**逻辑回归**（Logistic Regression），是统计学和机器学习中的一种分类方法。尽管其名称中含有“回归”，但逻辑回归主要用于二分类问题（也可以扩展到多分类问题）。其基本思想是将线性回归的输出通过某种方法（如 Sigmoid 函数）映射到[0,1]区间，以得到某事件发生的概率。

### 1. 基本数学模型

逻辑回归使用的主要函数是 Sigmoid 函数： 
$$
σ(z)=\frac{1}{1+e^{−z}}
$$
其中，*z* 是输入值。

当我们应用线性回归模型
$$
z=β_{0}+β_{1}x_{1}+β_{2}x_{2}+...+β_{n}x_{n}
$$
 我们将 *z* 值放入 Sigmoid 函数中，得到：
$$
p=\frac{1}{1+e^{-(β_{0}+β_{1}x_{1}+β_{2}x_{2}+...+β_{n}x_{n})}}
$$
这里的 *p* 表示事件发生的概率。

### 2. 对数几率

对数几率是逻辑回归名称的来源，它是事件发生概率与事件不发生概率之比的自然对数： 
$$
ln(\frac{p}{1-p})=β_{0}+β_{1}x_{1}+β_{2}x_{2}+...+β_{n}x_{n}
$$
这里，*p* 是事件发生的概率。

### 3. 参数估计

逻辑回归的参数通常使用**最大似然估计**（MLE）来估计。

### 4. 用途

逻辑回归常用于预测一个结果是两个可能类别中的哪一个，例如：邮件是垃圾邮件还是非垃圾邮件、交易是欺诈还是合法、病患是患病还是健康等。

### 5. 优点与缺点

**优点**：

- 输出是概率得分，可解释性强。
- 计算效率高，容易实现。
- 不需要假设数据的分布，如线性回归需要假设误差服从正态分布。

**缺点**：

- 它假定特征与对数几率之间存在线性关系。
- 对于非线性关系，可能需要额外的工作如特征工程或选择其他模型。
- 容易受到无关特征和高度相关特征的影响。

# 3.4 线性判别分析

参考课程链接：【【吃瓜教程】《机器学习公式详解》（南瓜书）与西瓜书公式推导直播合集】 https://www.bilibili.com/video/BV1Mh411e7VU/?p=6&share_source=copy_web&vd_source=c22abe8e67e193936015d5ca043a8148

线性判别分析（Linear Discriminant Analysis，简称LDA）是模式识别和机器学习领域中的一种经典的分类与降维方法。其核心思想是：为了达到最佳的分类效果，我们可以设计一个投影方向，使得同类数据的投影点尽可能地接近，不同类数据的投影点尽可能地远离。

LDA的基本步骤如下：

1. **计算每个类的均值向量**：首先，对于每一个类别，我们计算该类别中所有样本的特征向量的平均值。
2. **计算类内散度矩阵和类间散度矩阵**：
   - 类内散度矩阵：描述的是同一类别中的样本，它们的特征如何分散。
   - 类间散度矩阵：描述的是不同类别的样本均值之间，它们的特征如何分散。
3. **求解广义特征值问题**：利用类内散度矩阵和类间散度矩阵，计算其特征值和特征向量。这些特征向量即为我们要求的最佳投影方向。
4. **选择最大的k个特征值对应的特征向量**：通常选择的k的值小于原始数据的特征数，用于降维。
5. **数据转换到新的空间**：使用上一步得到的k个特征向量，将原始数据投影到这k个特征向量所定义的新的空间上。

LDA通常用于分类任务，但它也常常作为降维技术，特别是在面部识别等领域中有广泛的应用。需要注意的是，LDA假设数据的每个特征对所有的类别来说都是同样地、近似正态分布的，当这个假设不成立时，LDA的性能可能会受到影响。

# 3.5 多分类学习

处理多分类学习任务有多种策略和方法。以下是常见的方法：

1. **直接法**:
   - 有些算法天然地支持多分类任务，例如决策树、随机森林、朴素贝叶斯等。
   - 神经网络也可以直接用于多分类任务，通常在输出层使用softmax函数并配合交叉熵损失函数进行训练。
2. **“一对所有”（One-vs-All，OvA）策略**:
   - 对于每一个类别，训练一个二分类模型，将该类作为正类，其他所有类作为负类。
   - 最后，所有模型对一个新实例进行分类，选择给出最高置信度的那个模型的类别。
   - 适用于如SVM、逻辑回归等不直接支持多分类的算法。
3. **“一对一”（One-vs-One，OvO）策略**:
   - 对于每一对类别，训练一个二分类模型。
   - 对于K个类别，需要训练K(K-1)/2个分类器。
   - 对一个新实例进行分类时，大多数分类器投票决定类别。
   - 训练复杂度较高，但在测试时通常比OvA更快。
   - SVM在多分类问题上常用这种策略。
4. **层次分类**:
   - 在有些情境下，类别之间可能有层次或树状结构，可以首先分类到大的类别，然后再细分到小的类别。
5. **错误校正码**:
   - 为每个类分配一个二进制码，然后为每个位训练一个二分类器。
6. **基于输出编码的策略**:
   - 将多分类问题转化为多个二分类问题，并对这些二分类问题的结果进行编码，最后使用解码器得到最终的多分类结果。
7. **集成方法**:
   - 可以使用集成学习的策略，例如Boosting或Bagging，将多个二分类或多分类模型结合起来处理多分类问题。
8. **数据扩增和迁移学习**:
   - 在多分类问题中，有时某些类别的数据量可能很小。这种情况下，可以使用数据扩增技术增加这些类别的样本数量，或者使用迁移学习从其他相关任务中借鉴知识。

处理多分类任务时，选择合适的策略取决于具体问题的性质、数据分布、所使用的模型以及计算资源的限制等因素。

# 3.6 类别不平衡问题

类别不平衡问题是指在分类任务中，不同的类别之间的样本数量存在明显的差异。例如，正常邮件和垃圾邮件分类中，正常邮件可能远多于垃圾邮件。处理不平衡类别的方法有很多，以下是一些常见的策略：

1. **重采样**:
   - **上采样（Oversampling）**：增加少数类的样本。可以通过复制少数类样本或生成新的少数类样本（例如，使用SMOTE算法）来实现。
   - **下采样（Undersampling）**：减少多数类的样本。简单的策略包括随机选择多数类的子集。
2. **合成新样本**:
   - 使用**SMOTE**（Synthetic Minority Over-sampling Technique）或其变体来生成新的少数类样本。
3. **使用不同的评价标准**:
   - 替代传统的准确率评估，使用如F1分数、Matthews相关系数、AUC-ROC曲线等更加关注少数类的评价指标。
4. **修改算法权重**:
   - 很多算法允许为不同的类别设置不同的权重，从而让算法更加关注少数类。例如，在SVM或逻辑回归中，可以为少数类设置较高的权重。
5. **集成方法**:
   - **Bagging和Boosting**：如Random UnderSampling Boosting（RUSBoost）、Balanced Random Forest等都是为不平衡数据设计的。
   - 使用EasyEnsemble或BalanceCascade技术进行多次下采样并建立集成模型。
6. **使用异常检测方法**:
   - 把问题看作是一个异常检测问题，其中少数类被看作是异常。
7. **代价敏感学习**:
   - 修改算法使其考虑到不同类别的不同代价。例如，在决策树中，可以调整节点分裂的标准，使其考虑分类错误所带来的代价。
8. **使用不同的算法**:
   - 一些算法在处理不平衡数据上表现得比其他算法更好。例如，决策树或其变体（如随机森林）通常在不平衡数据上都有不错的表现。
9. **数据层次方法**:
   - 将数据分为多个集群或子集，并分别进行分类，然后再结合这些子集的结果。
10. **数据扩增**:

- 使用各种数据增强技巧，如旋转、缩放、裁剪等，来人为地增加少数类的样本。

处理类别不平衡问题时，可能需要尝试多种方法或它们的组合，以确定最适合具体问题的策略。

## SMOTE算法

SMOTE（Synthetic Minority Over-sampling Technique）是一个广泛使用的过采样方法，主要用于解决类别不平衡问题。其基本思想是为少数类别生成合成样本，而不是简单地复制已有的样本。

以下是SMOTE算法的基本步骤：

1. **选择样本**：从少数类别的样本中随机选择一个样本，记作*x*.

2. **找到邻居**：计算该样本�*x*在特征空间中的k个最近邻居。

3. **生成合成样本**：

   - 从这k个最近邻居中随机选择一个，记作
     $$
     x_{neighbor}
     $$
     
   - 对每个特征，计算差值：
     $$
     difference=x_{neighbor}-x
     $$
     
   - 乘以一个介于0到1之间的随机数：
     $$
     gap=randomnumber()*difference
     $$
     
   - 生成合成样本：
     $$
     x_{new}=x+gap
     $$
     
   
4. **重复步骤**：根据需要的合成样本数量，重复上述过程。

这种方法的主要优点是它可以生成在原始特征空间中与真实数据相似但并非完全相同的样本，从而增加了少数类别的多样性，减少了模型过拟合的风险。

但是，SMOTE也有其局限性：

1. **噪声放大**：如果原始数据中存在噪声，SMOTE可能会放大这些噪声。
2. **生成的数据可能并不总是有意义**：在某些情况下，特征空间中的线性插值可能不适用或不具有实际意义。
3. **可能引入类别重叠**：合成的样本可能导致不同类别的特征空间重叠，从而使分类任务变得更加困难。

由于这些局限性，SMOTE经常与其他技术（如下采样）结合使用，以获得更好的结果。此外，还有许多基于SMOTE的变体，如Borderline-SMOTE、SMOTE-NC（用于处理包含分类特征的数据）等，以解决其固有的一些问题。

## EasyEnsemble算法

EasyEnsemble是一种针对不平衡数据集的集成方法，它利用Adaboost的框架，并结合了多个下采样子集来创建一个更为健壮的分类器。简单地说，EasyEnsemble通过从多数类中多次随机下采样并与少数类结合，来创建多个平衡的子数据集，并在每个子数据集上训练一个基分类器。

以下是EasyEnsemble的基本步骤：

1. **生成子数据集**：
   - 对于T次迭代：
     1. 从多数类中随机下采样，获得与少数类相同数量的样本。
     2. 将这些下采样的样本与少数类的全部样本结合，形成一个平衡的子数据集。
2. **在子数据集上训练分类器**：
   - 对于每一个平衡的子数据集，训练一个基分类器。这可以是任何一种分类算法，如决策树、逻辑回归等。
3. **集成分类**：
   - 当一个新的样本需要分类时，所有的基分类器对其进行投票。通常情况下，我们会选择得票数最多的类作为最终的类别。

EasyEnsemble的优势在于：

- 通过对多数类进行多次下采样，它增加了分类模型的多样性。
- 由于在多个平衡的子数据集上训练，模型不太容易受到多数类的偏见影响。
- 它很容易并行化，因为每个基分类器都是独立地在一个子数据集上训练的。

但也有一些潜在的缺点：

- 过多地依赖下采样可能会导致丢失一些重要的多数类信息。
- 与其他集成方法相比，需要的计算资源可能会更多，尤其是当基分类器数量较多时。

尽管如此，EasyEnsemble在处理高度不平衡的数据集时仍然是一个有效的策略，它可以与其他方法（如SMOTE）结合使用，以进一步提高分类性能。