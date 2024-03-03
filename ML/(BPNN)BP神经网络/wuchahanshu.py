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