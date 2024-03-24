线性回归(Linear Regression)是一种用线性函数拟合数据，再用均方误差 (Mean Square Error ,mse)计算损失(cost)，然后用梯度下降法找到一组使mse最小的权重。

OLS（Ordinary Least Squares，普通最小二乘法），是最经典也是最简单的线性回归方法，它通过最小化残差平方和来估计模型参数。这里使用了OLS对鸢尾植物数据集进行了回归分类。
由于数据集线性关系过于清晰，因此正确率可以达到百分之百。

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/%E9%B8%A2%E5%B0%BE%E8%BE%93%E5%87%BA%E7%BB%93%E6%9E%9C.png)

线性回归还有很多变体，如下图：

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E6%A1%86%E6%9E%B6.png) 

在OLS基础上改变损失函数的有：
1. **岭回归（Ridge regression)**
2. **套索回归（Lasso)**
3. **弹性网络回归(Elastic-Net)**

![](https://cdn.jsdelivr.net/gh/SparKgod1/img/%E6%99%AE%E9%80%9A%E4%BA%8C%E4%B9%98.png)
![](https://cdn.jsdelivr.net/gh/SparKgod1/img/%E5%8F%98%E4%BD%93.png)
![](https://cdn.jsdelivr.net/gh/SparKgod1/img/%E5%BC%B9%E6%80%A7%E7%BD%91.png)

在OLS基础上改变特征选择的有
1. **正交匹配追踪（Orthogonal Matching Pursuit，OMP）**：在每一步中，OMP选择与当前残差最相关的特征，并将其添加到模型中，重复这个过程直到达到预设的特征数量或其他停止条件。
2. **逐步回归（Stepwise Regression）**：前向逐步回归从零开始，逐步添加对模型拟合有利的自变量；后向逐步回归从包含所有自变量的模型开始，逐步删除对模型拟合不利的自变量。
3. **最小角回归（Least Angle Regression，LARS）**：LARS以每次最小化残差的方向来选择变量，使得模型参数逐步逼近最优解。它会生成一条最小角路径，同时保持模型系数的精确性。