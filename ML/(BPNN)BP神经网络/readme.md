在Python中，误差函数通常指的是均方误差（Mean Squared Error, MSE），它是衡量模型预测值与实际值之间差异的一种常用方法。均方误差的公式是：
\[ MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2 \]
其中：
- \( y_i \) 是第 i 个实际值。
- \( \hat{y_i} \) 是第 i 个预测值。
- n 是数据点的总数。
以下是一个Python函数，用于计算均方误差：
```python
def mean_squared_error(y_true, y_pred):
    n = len(y_true)
    mse = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n
    return mse
# 示例数据
y_true = [3.0, -0.5, 2.5, 1.0]
y_pred = [2.5, 0.0, 2.0, 1.2]
# 计算均方误差
mse = mean_squared_error(y_true, y_pred)
print("Mean Squared Error:", mse)
```

在这个例子中，我们定义了一个名为 `mean_squared_error` 的函数，它接受两个列表作为参数：`y_true`（实际值）和 `y_pred`（预测值）。函数计算了两个列表之间元素差的平方，求和后除以数据点的数量来得到均方误差。然后我们使用了一些示例数据来计算均方误差，并打印了结果。