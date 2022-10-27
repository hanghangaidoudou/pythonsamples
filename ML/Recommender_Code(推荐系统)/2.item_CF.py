# -*-coding:utf-8 -*-
# @Time: 2022/7/8 19:15
# @Author: zou wei
# @File: 2.item_CF.py
# @Contact: visio@163.com
# @Software: PyCharm

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.metrics import jaccard_score
from faker import Faker
import json


if __name__ == '__main__':
    fake = Faker(locale='zh')
    Faker.seed(2022)
    users = np.array(list(set(fake.name() for i in range(100))))
    items = ['手机', '运营商', '数码电脑', '办公', '家居', '家具', '家装', '厨具', '男装', '女装', '童装', '内衣', '美妆',
             '个护清洁', '宠物', '女鞋', '箱包', '钟表', '珠宝', '男鞋', '运动', '户外', '房产', '汽车', '汽车用品', '母婴',
             '玩具', '乐器', '食品', '酒类', '生鲜', '特产', '艺术', '礼品', '鲜花', '农资', '绿植', '医药保健', '计生情趣',
             '图书', '文娱', '教育', '电子书', '机票', '酒店', '旅游', '生活', '理财', '众筹', '白条', '保险', '安装', '维修', '清洗']
    # users = np.array([fake.name() for i in range(5)])
    # items = ['手机', '运营商', '数码电脑', '办公']
    items = np.array(items)
    user_size, item_size = users.size, items.size
    print(items, item_size)
    print(users, user_size)

    # 用户购买记录数据集
    np.random.seed(2022)
    user_item = pd.DataFrame(data=(np.random.rand(user_size, item_size) > 0.7).astype(bool),
                             index=users, columns=items)
    item_similar = pd.DataFrame(data=1 - pairwise_distances(user_item.T.values, metric='jaccard'), index=items, columns=items)
    print('商品两两Jaccard相似度：\n', item_similar)

    similar_items = {}
    for item in items:
        similars = item_similar[item].drop(item).sort_values(ascending=False)
        similar_items[item] = list(similars.index[:2])
    print('相似商品：', similar_items)

    result = {}
    for user in users:
        already = items[user_item.loc[user]]
        candidate = set()
        for i in already:
            candidate = candidate.union(similar_items[i])
        result[user] = list(candidate.difference(already))
    print('推荐结果：', result)
    result_str = json.dumps(result, ensure_ascii=False, indent=2, separators=('，', '：'))
    with open('result_item.json', 'w', encoding='GBK') as f:
        f.write(result_str.replace('"', ''))
