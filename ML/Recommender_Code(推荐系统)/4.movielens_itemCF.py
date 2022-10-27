# -*-coding:utf-8 -*-
# @Time: 2022/7/10 15:06
# @Author: zou wei
# @File: 4.movielens_itemCF.py
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


# uid:用户ID；iid:商品ID；user_item:用户对商品的打分；user_similar:用户相似度矩阵
def itemCF_predict(uid, iid, user_item, item_similar):
    similar_items = item_similar[iid].drop(iid).dropna()    # iid商品的相似商品
    similar_items = similar_items[similar_items > 0]        # 筛选出正相关的商品
    if similar_items.size == 0:     # 没有相似商品
        return 0
    # 从相似商品筛选出uid对其有评分的商品
    index = list(set(user_item.loc[uid].dropna().index).intersection(set(similar_items.index)))
    if not index:   # uid没有对相似商品进行过评分
        return 0
    similar_items = similar_items[index]
    # 近邻用户的评分加权：权重为用户相似度
    sum_of_rate_mul_similar = 0     # 评分预测公式的分子部分的值
    sum_of_similar = 0              # 评分预测公式的分母部分的值
    for sim_iid in similar_items.index:
        similarity = similar_items[sim_iid]
        r = user_item.loc[uid, sim_iid]             # 用户uid对相似商品sim_iid的评分
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
def itemCF_predict_all(uid, user_item, item_similar, candidate_items=None):
    if candidate_items is None:
        candidate_items = user_item.columns
    for iid in candidate_items:   # 遍历所有候选商品
        rating = itemCF_predict(uid, iid, user_item, item_similar)
        yield uid, iid, rating


def get_candidate_items(uid, filter_rule):
    candidate_items = None
    if isinstance(filter_rule, str):
        if filter_rule == 'unhot':      # 过滤冷门电影，只保留热门电影
            count = user_item.count()   # 每列非空的个数：即该电影被打分的人数
            candidate_items = count[count > 100].index   # 打分数量超过10的电影IDs
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
    return candidate_items


# uid:用户ID；user_item:用户对商品的打分；user_similar:用户相似度矩阵
def userCF_predict_all_with_rule(uid, user_item, user_similar, filter_rule='unhot'):
    candidate_items = get_candidate_items(uid, filter_rule)
    yield from userCF_predict_all(uid, user_item, user_similar, candidate_items)


def itemCF_predict_all_with_rule(uid, user_item, item_similar, filter_rule='unhot'):
    candidate_items = get_candidate_items(uid, filter_rule)
    yield from itemCF_predict_all(uid, user_item, item_similar, candidate_items)


def topK(uid, k, based='user'):
    user_item = load_data(data_path)
    similar_matrix = compute_pearson_similarity(user_item, based=based)
    if based == 'user:':
        results = userCF_predict_all_with_rule(uid, user_item, similar_matrix, filter_rule=["unhot", "rated"])
    else:
        results = itemCF_predict_all_with_rule(uid, user_item, similar_matrix, filter_rule=["unhot", "rated"])
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
    user_item = load_data(data_path)
    # print(user_item)
    item_similar = compute_pearson_similarity(user_item, based='item')
    # print('商品相似度：\n', item_similar)

    # # 预测用户1对物品1的评分
    # itemCF_predict(1, 1, user_item, item_similar)
    # # 预测用户1对物品2的评分
    # itemCF_predict(1, 2, user_item, item_similar)

    # # 使用所有候选电影做预测
    # for r in itemCF_predict_all(1, user_item, item_similar):
    #     print(r)

    # # 过滤冷门电影、已经评分电影
    # for result in itemCF_predict_all_with_rule(1, user_item, item_similar, filter_rule=['rated', 'unhot']):
    #     print(result)

    result = topK(1, 10, 'item')     # 给用户1推荐前10个商品
    print(result)
    names = movie_names(result)
    print(names)
