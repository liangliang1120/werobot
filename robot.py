# -*- coding: utf-8 -*-
from werobot import WeRoBot
import config as cfg
import requests
import pandas as pd
import json
import random
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
from scipy.spatial.distance import cosine
from functools import reduce
from operator import and_
import q_a

# from chatterbot import ChatBot
# from chatterbot.trainers import ChatterBotCorpusTrainer
# global chatbot


myrobot = WeRoBot(token=cfg.token)
myrobot.config["APP_ID"] = cfg.appid
myrobot.config['ENCODING_AES_KEY'] = cfg.aeskey


# chatbot = ChatBot("ChineseChatBot")
# trainer = ChatterBotCorpusTrainer(chatbot)
# trainer.train("chatterbot.corpus.chinese")


@myrobot.image
def image_repeat(message,session):
    return message.img


@myrobot.text
def test_repeat(message):
    # return message.content
    msg = message.content
    # 匹配语料
    # 没匹配上就用图灵机器人接口

    def get_response(msg):  # 图灵机器人微信好友自动回复
        KEY = '39d912669ee74f349dd312da23938699'
        apiUrl = 'http://www.tuling123.com/openapi/api'
        data = {
            'key': KEY,
            'info': msg,
            'userid': 'wechat-robot',
        }
        try:
            r = requests.post(apiUrl, data=data).json()
            return r.get('text')
        except:
            return 0


    output = q_a.get_cor_response(msg)
    if output != '':
        data = output
        #print(666,str(data))
    else:
        data = get_response(msg)

    return data


# @myrobot.text
# def text_response(message,session):
#     answer = chatbot.get_response(message.content).text
#     return answer
