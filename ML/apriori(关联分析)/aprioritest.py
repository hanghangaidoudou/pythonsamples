def loadDataSet():
    return [[1,2,5],[2,4],[2,3],[1,2,4],[1,3],[2,3],[1,3],[1,2,3,5],[1,2,3]]
#1.构建候选1项集C1
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])

    C1.sort()
    return list(map(frozenset, C1))

#将候选集Ck转换为频繁项集Lk
#D：原始数据集
#Cn: 候选集项Ck
#minSupport:支持度的最小值
def scanD(D, Ck, minSupport):
    #候选集计数
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt.keys(): ssCnt[can] = 1
                else: ssCnt[can] += 1

    numItems = float(len(D))
    Lk= []     # 候选集项Cn生成的频繁项集Lk
    supportData = {}    #候选集项Cn的支持度字典
    #计算候选项集的支持度, supportData key:候选项， value:支持度
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            Lk.append(key)
        supportData[key] = support
    return Lk, supportData

#连接操作，将频繁Lk-1项集通过拼接转换为候选k项集
def aprioriGen(Lk_1, k):
    Ck = []
    lenLk = len(Lk_1)
    for i in range(lenLk):
        L1 = list(Lk_1[i])[:k - 2]
        L1.sort()
        for j in range(i + 1, lenLk):
            #前k-2个项相同时，将两个集合合并
            L2 = list(Lk_1[j])[:k - 2]
            L2.sort()
            if L1 == L2:
                Ck.append(Lk_1[i] | Lk_1[j])

    return Ck

def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    L1, supportData = scanD(dataSet, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Lk_1 = L[k-2]
        Ck = aprioriGen(Lk_1, k)
        print("ck:",Ck)
        Lk, supK = scanD(dataSet, Ck, minSupport)
        supportData.update(supK)
        print("lk:", Lk)
        L.append(Lk)
        k += 1

    return L, supportData





#将频繁Lk-1项集转换为候选k项集
def aprioriGen(Lk_1, k):
    Ck = []
    lenLk = len(Lk_1)
    for i in range(lenLk):
        L1 = Lk_1[i]
        for j in range(i + 1, lenLk):
            L2 = Lk_1[j]
            if len(L1 & L2) == k - 2:
                L1_2 = L1 | L2
                if L1_2 not in Ck:
                    Ck.append(L1 | L2)
    return Ck

dataset = loadDataSet()
L, supportData = apriori(dataset, minSupport=0.2)


def scanD(D, Ck, minSupport):
    #候选集计数
    ssCnt = {}
    #数据集过滤
    D2 = [item for item in D if len(item) >= len(Ck[0])]
    for tid in D2:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt.keys(): ssCnt[can] = 1
                else: ssCnt[can] += 1

    return Ck, supportData


#生成关联规则
#L: 频繁项集列表
#supportData: 包含频繁项集支持数据的字典
#minConf 最小置信度
def generateRules(L, supportData, minConf=0.7):
    #包含置信度的规则列表
    bigRuleList = []
    #从频繁二项集开始遍历
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


# 计算是否满足最小可信度
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []
    #用每个conseq作为后件
    for conseq in H:
        # 计算置信度
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            # 元组中的三个元素：前件、后件、置信度
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)

    #返回后件列表
    return prunedH


# 对规则进行评估
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H, m + 1)
       # print(1,H, Hmp1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 0):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)