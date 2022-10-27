from numpy import *
import matplotlib.pyplot as plt
import operator



class KNNClassifier():
    def __init__(self):
        self.dataSet = []
        self.labels = []

    def loadDataSet(self,filename):
        fr = open(filename)
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataLine = list()
            for i in lineArr:
                dataLine.append(float(i))
            label = dataLine.pop() # pop the last column referring to  label
            self.dataSet.append(dataLine)
            self.labels.append(int(label))
            result = self.dataSet
          #  plt.plot(result)
           # print(result)
          #  plt.show()

    def setDataSet(self, dataSet, labels):
        self.dataSet = dataSet
        self.labels = labels

    def kNN(x, dataSet, labels, k):
        dataSetSize = dataSet.shape[0]
        distance1 = tile(x, (dataSetSize, 1)) - dataSet  # 欧氏距离计算开始
        distance2 = distance1 ** 2  # 每个元素平方
        distance3 = distance2.sum(axis=1)  # 矩阵每行相加
        distance4 = distance3 ** 0.5  # 欧氏距离计算结束
        sortedIndex = distance4.argsort()  # 返回从小到大排序的索引
        classCount = {}
        for i in range(k):  # 统计前k个数据类的数量
            label = labels[sortedIndex[i]]
            classCount[label] = classCount.get(label, 0) + 1
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  # 从大到小按类别数目排序
        return sortedClassCount[0][0]
  #预测
    def predict(self, data, k):
        #设置训练数据
        self.dataSet = array(self.dataSet)
        #设置训练数据的标签
        self.labels = array(self.labels)
        self._normDataSet()
        dataSetSize = self.dataSet.shape[0]
        # get distance 获取距离
        diffMat = tile(data, (dataSetSize,1)) - self.dataSet #欧氏距离计算开始
        print (">>>>>>>>>>>>>>>>>>欧氏距离计算开始")
        print (diffMat)
        sqDiffMat = diffMat**2# 每个元素平方
        print (">>>>>>>>>>>>>>>>>>每个元素平方")
        print (sqDiffMat)
        distance3 = sqDiffMat.sum(axis=1) # 矩阵每行相加

        distances = distance3 ** 0.5  # 欧氏距离计算结束

        print (">>>>>>>>>>>>>>>>>>矩阵每行相加后开方，算出距离")
        print (distances)
        # get K nearest neighbors 获取最临近的数据K
        sortedDistIndicies = distances.argsort()# 返回从小到大排序的索引
        print(">>>>>>>>>>>>>>>>>>排下序 返回从小到大排序的索引，是，你没看错，是索引")
        print(sortedDistIndicies)
        classCount= {}
        for i in range(k):# 统计前k个数据类的数量()
            voteIlabel = self.labels[sortedDistIndicies[i]]#已经是排序好的顺序，不过是按照排序顺序找到其索引
            print(">>>>>>>>>>>>>>>>>>第"+str(i)+"个"+str(voteIlabel))
            print(sortedDistIndicies[i])
            classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1# classCount 计数，前k个数据属于的类别
        # get fittest label 获取测试的label
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) #数量 从大到小按类别数目排序
        print(">>>>>>>>>>>>>>>>>>排后的类别数量")
        print (sortedClassCount)
        return sortedClassCount[0][0]
# 初始化数据集
    def _normDataSet(self):
        minVals = self.dataSet.min(0)
        maxVals = self.dataSet.max(0)
        ranges = maxVals - minVals
        normDataSet = zeros(shape(self.dataSet))
        m = self.dataSet.shape[0]
        normDataSet = self.dataSet - tile(minVals, (m,1))
        normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
        self.dataSet = normDataSet

    def test(self):
        self.dataSet = array([[1.0,1.1],[1.0,1.0],[0.9,0.9],[0,0],[0,0.1],[0,0.2]])
        self.labels = [1,1,1,2,2,2]
        print(">>>>>>>>>>>>>>>>>>测试 属于的类别")
        print(self.predict([1.0,1.1], 2))
        result =   self.dataSet
        plt.plot([1.0,1.1], 'rs')
        plt.plot(result,'bs')
        print(result)
        plt.show()

if __name__ == '__main__':
    KNN = KNNClassifier()
    KNN.loadDataSet('testData.txt')
    KNN.test()
    #print(KNN.predict([72011, 4.932976, 0.632026], 5))




