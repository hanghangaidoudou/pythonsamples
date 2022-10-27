import math
import random

random.seed(0)

#随机函数
def rand(a, b):
    return (b - a) * random.random() + a

#制作矩阵
def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat

#S函数
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

#S型函数派生的导数
def sigmoid_derivative(x):
    return x * (1 - x)

#定义各种层
class BPNeuralNetwork:
    def __init__(self):
        #输入层个数
        self.input_n = 0
        #隐含层个数
        self.hidden_n = 0
        #输出层个数
        self.output_n = 0
        # 输入层
        self.input_cells = []
        #隐含层
        self.hidden_cells = []
        # 输出层
        self.output_cells = []
        #输入层权重
        self.input_weights = []
        #输出层权重
        self.output_weights = []
        #输入层误差
        self.input_correction = []
        #输出层误差
        self.output_correction = []
#建立参数
    def setup(self, ni, nh, no):
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no
        # init cells 初始化神经元
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        # init weights 初始化权重 矩阵
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
        # random activate 随机激活
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)
        # init correction matrix 初始化矩阵误差
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)
 #预测
    def predict(self, inputs):
        # activate input layer 激活输入层
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        # activate hidden layer 激活隐藏层
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
                #利用S函数计算隐藏层神经元
            self.hidden_cells[j] = sigmoid(total)

        # activate output layer 激活输出层
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
                #利用S函数计算输出层神经元
            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:]
 #反向传播
    def back_propagate(self, case, label, learn, correct):
        # feed forward 前馈
        self.predict(case)
        # get output layer error 获取输出层误差
        output_deltas = [0.0] * self.output_n #初始化二维的输出偏导数
        for o in range(self.output_n):
            #误差
            error = label[o] - self.output_cells[o]
            #偏导数
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
        # get hidden layer error 获取隐含层误差
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error
        # update output weights 更新输出层权重 和输出层误差
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # update input weights 更新输入层权重
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # get global error 获取全局误差
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error
   #训练 训练次数 limit 精度 learn 纠正步长correct
    def train(self, cases, labels, limit=10000, learn=0.05, correct=0.1):
        count =1;
        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)
                count+=1
                #print(error)

                #print(count)



    def test(self):
        #输入向量，三个字段
        cases = [
            [0, 0,2],
            [0, 1,3],
            [1, 0,4],
            [1, 1,5],
        ]
        #期望输出向量
        labels = [[0], [1], [1], [0]]
        #3个输入层 5 个隐含层 1 个输出层 神经元
        self.setup(3, 5, 1)
        #训练10000次 学习速率 0.05 误差允许  0.1
        self.train(cases, labels, 10000, 0.05, 0.1)
        for case in cases:
            print(self.predict(case))


if __name__ == '__main__':
    nn = BPNeuralNetwork()
    nn.test()
