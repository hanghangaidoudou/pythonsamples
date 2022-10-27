# -*-coding:utf-8 -*-
# @Time: 2022/7/7 22:03
# @Author: zou wei
# @File: 3.movielens_userCF.py
# @Contact: visio@163.com
# @Software: PyCharm

import os
import pandas as pd
import numpy as np


def load_data(data_path):
    cache_file_path = os.path.join(cache_path, 'ratings_matrix.cache')
    if os.path.exists(cache_file_path):
        print('加载用户-商品矩阵...')
        return pd.read_pickle(cache_file_path)
    print('加载并缓存用户-商品矩阵...')
    # Coordinate Format (COO)：row,col,value
    ratings_coo = pd.read_csv(data_path, header=0, usecols=range(3))
    data = ratings_coo.pivot_table(index='userId', columns='movieId', values='rating')
    data.to_pickle(cache_file_path)
    return data


def compute_pearson_similarity(user_item, based='user'):
    if based == 'user':
        user_similarity_path = os.path.join(cache_path, 'user_similarity.cache')
        if os.path.exists(user_similarity_path):
            print('正在加载用户相似度矩阵...')
            # similarity = pd.read_excel(user_similarity_path, index_col=0, header=0)
            similarity = pd.read_pickle(user_similarity_path)
        else:
            print('开始计算用户相似度矩阵...')
            similarity = user_item.T.corr()
            # similarity.to_excel(user_similarity_path)
            similarity.to_pickle(user_similarity_path)
    else:   # elif based == 'item':
        item_similarity_path = os.path.join(cache_path, 'item_similarity.cache')
        if os.path.exists(item_similarity_path):
            print('正在加载商品相似度矩阵...')
            # similarity = pd.read_excel(item_similarity_path)
            similarity = pd.read_pickle(item_similarity_path)
        else:
            print('开始计算商品相似度矩阵...')
            similarity = user_item.corr()
            # similarity.to_excel(item_similarity_path)   # 数据过大，直接保存明文慢，改成Pickle方式
            similarity.to_pickle(item_similarity_path)
    return similarity


# uid:用户ID；iid:商品ID；user_item:用户对商品的打分；user_similar:用户相似度矩阵
def userCF_predict(uid, iid, user_item, user_similar):
    similar_users = user_similar[uid].drop(uid).dropna()    # uid用户的相似用户
    similar_users = similar_users[similar_users > 0]        # 筛选出正相关的用户
    if similar_users.empty:     # 没有相似用户
        return 0
    # 从相似用户筛选出对iid物品有评分的用户
    index = set(user_item[iid].dropna().index).intersection(set(similar_users.index))
    if not index:   # 相似用户没有对iid的评分
        return 0
    similar_users = similar_users.loc[list(index)]
    # 近邻用户的评分加权：权重为用户相似度
    sum_of_rate_mul_similar = 0    # 评分预测公式的分子部分的值
    sum_of_similar = 0  # 评分预测公式的分母部分的值
    for sim_uid, similarity in similar_users.iteritems():
        r = user_item.loc[sim_uid, iid]             # 相似用户sim_uid对感兴趣电影iid的评分
        sum_of_rate_mul_similar += similarity * r   # 分子
        sum_of_similar += similarity                # 分母
    predict_rate = sum_of_rate_mul_similar/sum_of_similar
    # print('用户“%d”对电影“%d”的预测评分为%0.2f' % (uid, iid, predict_rate))
    return predict_rate


# uid:用户ID；user_item:用户对商品的打分；user_similar:用户相似度矩阵
def userCF_predict_all(uid, user_item, user_similar, candidate_items=None):
    if candidate_items is None:
        candidate_items = user_item.columns
    for iid in candidate_items:   # 遍历所有候选商品
        rating = userCF_predict(uid, iid, user_item, user_similar)
        yield uid, iid, rating


# uid:用户ID；user_item:用户对商品的打分；user_similar:用户相似度矩阵
def userCF_predict_all_with_rule(uid, user_item, user_similar, filter_rule='unhot'):
    candidate_items = None
    if isinstance(filter_rule, str):
        if filter_rule == 'unhot':      # 过滤冷门电影，只保留热门电影
            count = user_item.count()   # 每列非空的个数：即该电影被打分的人数
            candidate_items = count[count > 100].index   # 打分数量超过100的电影IDs
        elif filter_rule == 'rated':    # 过滤用户评分过的电影，即只保留未评分的电影
            user_ratings = user_item.loc[uid]
            candidate_items = user_ratings[user_ratings != user_ratings].index
    elif isinstance(filter_rule, list):
        # 过滤非热门和用户已经评分过的电影
        count = user_item.count()
        index1 = count[count > 10].index
        user_ratings = user_item.loc[uid]
        index2 = user_ratings[user_ratings != user_ratings].index
        candidate_items = set(index1) & set(index2)
    yield from userCF_predict_all(uid, user_item, user_similar, candidate_items)


def userCF_topK(uid, k):
    user_item = load_data(data_path)
    user_similar = compute_pearson_similarity(user_item, based="user")
    results = userCF_predict_all_with_rule(uid, user_item, user_similar, filter_rule=["unhot", "rated"])
    return sorted(results, key=lambda x: x[2], reverse=True)[:k]


def movie_names(result):
    s = ''
    movie_data = pd.read_csv(movie_name_path, header=0, index_col=0)
    for i, r in enumerate(result, start=1):
        movie_id = r[1]
        a = movie_data.loc[movie_id]
        s += '%02d 电影名：%s，主演：%s，得分：%.3f\n' % (i, a['title'], a['genres'], r[2])
    return s


if __name__ == '__main__':
    data_path = './ml-latest-small/ratings.csv'
    movie_name_path = './ml-latest-small/movies.csv'
    cache_path = './ml-latest-small/'
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    # user_item = load_data(data_path)
    # print(user_item)
    # user_similar = compute_pearson_similarity(user_item, based='user')
    # print('用户相似度：\n', user_similar)
    # item_similar = compute_pearson_similarity(user_item, based='item')
    # print('商品相似度：\n', item_similar)
    #
    # # 预测用户1对物品1的评分
    # userCF_predict(1, 1, user_item, user_similar)
    # # 预测用户1对物品2的评分
    # userCF_predict(1, 2, user_item, user_similar)
    #
    # # 使用所有候选电影做预测
    # # a = userCF_predict_all(1, user_similar, user_similar)
    #
    # # 过滤冷门电影、已经评分电影
    # for result in userCF_predict_all_with_rule(1, user_item, user_similar, filter_rule=['rated', 'unhot']):
    #     print(result)

    result = userCF_topK(1, 10)     # 给用户1推荐前10个商品
    print(result)
    names = movie_names(result)
    print(names)
