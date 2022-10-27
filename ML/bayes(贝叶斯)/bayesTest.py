#-*- coding: utf-8 -*-  #添加中文注释
from numpy import *

#过滤网站的恶意留言
#样本数据    
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    #类别标签：1侮辱性文字，0正常言论
    classVec = [0,1,0,1,0,1]     
    #返回文档向量，类别向量
    return postingList,classVec

#创建词汇表
#输入：dataSet已经经过切分处理
#输出：包含所有文档中出现的不重复词的列表                          
def createVocabList(dataSet):
    #构建set集合，会返回不重复词表
    vocabSet = set([])
    #遍历每篇文档向量，扫描所有文档的单词 
    for document in dataSet:
        #通过set(document)，获取document中不重复词列表
        vocabSet = vocabSet | set(document) #求并集
    return list(vocabSet)

#***词集模型：只考虑单词是否出现
#vocabList：词汇表
#inputSet ：某个文档向量
def setOfWords2Vec(vocabList, inputSet):
    #创建所含元素全为0的向量
    returnVec = [0]*len(vocabList)
    #依次取出文档中的单词与词汇表进行对照，若在词汇表中出现则为1
    for word in inputSet:
        if word in vocabList:
        #单词在词汇表中出现，则记为1 
            returnVec[vocabList.index(word)] = 1 #词集模型
        #若测试文档的单词，不在词汇表中，显示提示信息，该单词出现次数用0表示
        else: print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec

#====训练分类器,原始的朴素贝叶斯，没有优化=====
#输入trainMatrix：词向量数据集
#输入trainCategory：数据集对应的类别标签
#输出p0Vect：词汇表中各个单词在正常言论中的类条件概率密度
#输出p1Vect：词汇表中各个单词在侮辱性言论中的类条件概率密度
#输出pAbusive：侮辱性言论在整个数据集中的比例
def trainNB00(trainMatrix,trainCategory):
    #numTrainDocs训练集总条数
    numTrainDocs = len(trainMatrix)
    #训练集中所有不重复单词总数
    numWords = len(trainMatrix[0])
    #侮辱类的概率(侮辱类占总训练数据的比例)
    pAbusive = sum(trainCategory)/float(numTrainDocs) 
    #*正常言论的类条件概率密度 p(某单词|正常言论)=p0Num/p0Denom
    p0Num = zeros(numWords); #初始化分子为0
    #*侮辱性言论的类条件概率密度 p(某单词|侮辱性言论)=p1Num/p1Denom    
    p1Num = zeros(numWords)  #初始化分子为0
    #初始化分母置为0   
    p0Denom = 0; 
    p1Denom = 0               
    #遍历训练集数据    
    for i in range(numTrainDocs):
        #若为侮辱类
        if trainCategory[i] == 1:
            #统计侮辱类所有文档中的各个单词总数
            p1Num += trainMatrix[i]
            #p1Denom侮辱类总单词数
            p1Denom += sum(trainMatrix[i])

        #若为正常类
        else:
            #统计正常类所有文档中的各个单词总数
            p0Num += trainMatrix[i]
            #p0Denom正常类总单词数
            p0Denom += sum(trainMatrix[i])   
    #词汇表中的单词在侮辱性言论文档中的类条件概率    
    p1Vect = p1Num/p1Denom        
    #词汇表中的单词在正常性言论文档中的类条件概率 
    p0Vect = p0Num/p0Denom
    return p0Vect,p1Vect,pAbusive


#=====训练分类器，优化处理=====
#输入trainMatrix：词向量数据集
#输入trainCategory：数据集对应的类别标签
#输出p0Vect：词汇表中各个单词在正常言论中的类条件概率密度
#输出p1Vect：词汇表中各个单词在侮辱性言论中的类条件概率密度
#输出pAbusive：侮辱性言论在整个数据集中的比例
def trainNB0(trainMatrix,trainCategory):
    #训练集总条数：行数
    numTrainDocs = len(trainMatrix)
    #训练集中所有单词总数：词向量维度
    numWords = len(trainMatrix[0])
    #侮辱类的概率(侮辱类占总训练数据的比例)
    pAbusive = sum(trainCategory)/float(numTrainDocs)    
    #*拉普拉斯平滑防止类条件概率为0，初始化分子为1，分母为2
    #正常类向量置为1
    p0Num = ones(numWords); #初始化分子为1
    #侮辱类向量置为1    
    p1Num = ones(numWords)  #初始化分子为1
    #初始化分母置为2    
    p0Denom = 2.0; 
    p1Denom = 2.0               
    #遍历训练集每个样本   
    for i in range(numTrainDocs):
        #若为侮辱类
        if trainCategory[i] == 1:
            #统计侮辱类所有文档中的各个单词总数
            p1Num += trainMatrix[i] #向量
            #p1Denom侮辱类总单词数
            p1Denom += sum(trainMatrix[i])

        #若为正常类
        else:
            #统计正常类所有文档中的各个单词总数
            p0Num += trainMatrix[i]
            #p0Denom正常类总单词数
            p0Denom += sum(trainMatrix[i])   
    #数据取log，即单个单词的p(x1|c1)取log，防止下溢出        
    p1Vect = log(p1Num/p1Denom)         
    p0Vect = log(p0Num/p0Denom) 
    return p0Vect,p1Vect,pAbusive

#vec2Classify：待分类文档 
#p0Vect:词汇表中每个单词在训练样本的正常言论中的类条件概率密度
#p1Vect:词汇表中每个单词在训练样本的侮辱性言论中的类条件概率密度
#pClass1：侮辱性言论在训练集中所占的比例
def classifyNB(vec2Classify, p0Vect, p1Vect, pClass1):
    #在对数空间中进行计算，属于哪一类的概率比较大就判为哪一类
    #print'0p1=',sum(vec2Classify * p0Vect) #查看结果
    #print'0p0=',sum(vec2Classify * p0Vect)
    p1 = sum(vec2Classify * p1Vect) + log(pClass1)    
    p0 = sum(vec2Classify * p0Vect) + log(1.0 - pClass1)
    #print'p1=',p1
    #print'p0=',p0
    if p1 > p0:
        return 1
    else: 
        return 0

def testingNB():
    #获得训练数据，类别标签
    listOPosts,listClasses = loadDataSet()
    #创建词汇表
    myVocabList = createVocabList(listOPosts)
    #构建矩阵，存放训练数据
    trainMat=[]

    #遍历原始数据，转换为词向量，构成数据训练矩阵
    for postinDoc in listOPosts:
        #数据转换后存入数据训练矩阵trainMat中
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    #训练分类器
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))

    #===测试数据（1）
    testEntry = ['love', 'my', 'dalmation']
    #测试数据转为词向量
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    #输出分类结果
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

    #===测试数据（2）
    testEntry = ['stupid', 'garbage']
    #测试数据转为词向量
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    #输出分类结果
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)    )



#***词袋模型：考虑单词出现的次数
#vocabList：词汇表
#inputSet ：某个文档向量
def bagOfWords2VecMN(vocabList, inputSet):
    #创建所含元素全为0的向量
    returnVec = [0]*len(vocabList)
    #依次取出文档中的单词与词汇表进行对照，统计单词在文档中出现的次数
    for word in inputSet:
        if word in vocabList:
            #单词在文档中出现的次数
            returnVec[vocabList.index(word)] += 1
        #若测试文档的单词，不在词汇表中，显示提示信息，该单词出现次数用0表示
        else:  print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec


#准备数据，按空格切分出词 
#单词长度小于或等于2的全部丢弃
def textParse(bigString):    
    import re
    listOfTokens = re.split(r'\W*', bigString)
    #tok.lower() 将整个词转换为小写
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 

def spamTest():
    #文章按篇存放
    docList=[]; 
    #存放文章类别
    classList = [];
    #存放所有文章内容    
    fullText =[]
    for i in range(1,26):
        #读取垃圾邮件
        #wordList = textParse(open('D:/work/python/email/spam/%d.txt' % i).read())   
        wordList = textParse(open('E:/大数据神经网络/深度学习资料/学习基础/具体实践/机器学习/bayes(贝叶斯)/email/spam/%d.txt' % i,encoding='gbk').read())
        #docList按篇存放文章
        docList.append(wordList)
        #fullText邮件内容存放到一起
        fullText.extend(wordList)
        #垃圾邮件类别标记为1
        classList.append(1)

        #读取正常邮件
        #wordList = textParse(open('D:/work/python/email/ham/%d.txt' % i).read())
        wordList = textParse(open('E:/大数据神经网络/深度学习资料/学习基础/具体实践/机器学习/bayes(贝叶斯)/email/ham/%d.txt' % i,encoding='utf-8' ).read())
        docList.append(wordList)
        fullText.extend(wordList)
        #正常邮件类别标记为0
        classList.append(0)

    #创建词典    
    vocabList = createVocabList(docList)
    #训练集共50篇文章
    trainingSet = list(range(50));
    #创建测试集
    testSet=[]
    #随机选取10篇文章为测试集，测试集中文章从训练集中删除    
    for i in range(10):
        #0-50间产生一个随机数
        randIndex = int(random.uniform(0,len(trainingSet)))
        #从训练集中找到对应文章，加入测试集中
        testSet.append(trainingSet[randIndex])
        #删除对应文章
        del(trainingSet[randIndex])  

    #准备数据，用于训练分类器    
    trainMat=[]; #训练数据
    trainClasses = [] #类别标签

    #遍历训练集中文章数据
    for docIndex in trainingSet:
        #每篇文章转为词袋向量模型，存入trainMat数据矩阵中
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        #trainClasses存放每篇文章的类别
        trainClasses.append(classList[docIndex])
    #训练分类器
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))

    #errorCount记录测试数据出错次数
    errorCount = 0
    #遍历测试数据集，每条数据相当于一条文本
    for docIndex in testSet:
        #文本转换为词向量模型    
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        #模型给出的分类结果与本身类别不一致时，说明模型出错，errorCount数加1
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            #输出出错的文章
            print ("classification error",docList[docIndex])

    #输出错误率，即出错次数/总测试次数
    print ('the error rate is: ',float(errorCount)/len(testSet))


    #return vocabList,fullText

if __name__ == "__main__":

###**********************留言板数据：观察参数值start
###    #获取数据
    listOPosts,listClasses = loadDataSet()  
#    #构建词汇表
    myVocabList = createVocabList(listOPosts)
    print ('myVocabList=',myVocabList)
    print ('result=',setOfWords2Vec(myVocabList, listOPosts[0]))
    trainMat = []
    for postinDoc in listOPosts:
        #构建训练矩阵
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0Vect,p1Vect,pAbusive = trainNB0(trainMat, listClasses)
    print ('p0Vect=')
    print (p0Vect)
    print ('p1Vect=')
    print (p1Vect)
    print ('pAbusive=')
    print (pAbusive)
    print ('trainMatrix=')
    print (trainMat)
    print ('listClasses=',listClasses)
###**********************留言板数据：观察参数值end    

##    #测试留言板文档
    print ('===================================')
    testingNB()

#***********************垃圾邮件    
##    #垃圾邮件分类
    print ('=======spam filtering=============')
    spamTest()
