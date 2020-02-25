import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
import random
from scipy.spatial.distance import cosine
from functools import reduce
from operator import and_
import requests

def get_cor_response(content):
    if len(content)<=2:
        return ''
    data = pd.read_csv('qa_c.csv')[['question','answer']]

    # data=data.head()

    def token(string):
        return re.findall(r'[\d|\w]+',string) #过滤

    def cut(string):
        return ' '.join(jieba.cut(string))

    # token("这是一个测试~！@##@ \n \t")
    news_content = data['question'].tolist()
    news_content = [token(str(n)) for n in news_content]
    news_content = [' '.join(n) for n in news_content]
    news_content = [cut(n) for n in news_content]

    vectorized = TfidfVectorizer(max_features = 40000)

    X = vectorized.fit_transform(news_content)
    X_array = X.toarray()

    vectorized.vocabulary_   # explain this word2id
    len(vectorized.vocabulary_  )
    np.where(X[0].toarray())

    def get_distance(v1,v2):
        return cosine(v1,v2)

    # get_distance(X[15].toarray()[0],X[14].toarray()[0])

    # y = X[15].toarray()[0]
    # np.where(X[14].toarray())

    transposed_x = X.transpose().toarray()
    word2id = vectorized.vocabulary_
    id2word = {d:w for w,d in word2id.items()}
    set(np.where(transposed_x[1217])[0])  #解释 意思 表示含有该词的新闻
    #word2id['存款']
    usa_force = set(np.where(transposed_x[4727])[0])


    def search_engine(query):
        words = query.split()

        query_vec = vectorized.transform([' '.join(words)]).toarray()[0]

        candidate_ids = [word2id[w] for w in words]


        # 问题中的词进行匹配，求完全的交集
        # merged_documents = reduce(and_, documents_ids)

        # 去除常见的无意义词
        list_w = [1991, 10726, 1728, ]
        try:
            for v in list_w:
                candidate_ids.remove(v)
        except:
            pass

        documents_ids = [
            set(np.where(transposed_x[_id])[0]) for _id in candidate_ids
        ]

        #只要有词存在就好，匹配度越高越好，设置阈值，低于该匹配度就不从语料中取回复
        # 此处设置只要有两个词匹配上就算话题匹配上了
        if reduce(and_, documents_ids) != set():
            sorted_documents_id = sorted(reduce(and_, documents_ids), key=lambda i: get_distance(query_vec, X[i].toarray()))
        else:
            for n in range(len(documents_ids)):
                if set(documents_ids[n]) & set(documents_ids[n+1]) != set() and n<=len(documents_ids)-2:
                    sorted_documents_id = sorted(set(documents_ids[n]) & set(documents_ids[n+1]), key=lambda i: get_distance(query_vec, X[i].toarray()))
                    break
                else:
                    continue

        return sorted_documents_id

    # sen = search_engine('存款 现金')

    content = [content]
    content = [token(str(n)) for n in content]
    content = [' '.join(n) for n in content]
    content = [cut(n) for n in content]

    try:
        sen = search_engine(content[0])
    except:
        output = ''
        return output
    if sen == []:
        output = ''
    else:
        output = data.loc[sen[0],'answer']
    return output



if(__name__=="__main__"):
    content = '美味佳肴'
    output = get_cor_response(content)
    print(output)
