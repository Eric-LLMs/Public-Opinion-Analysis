# -*- coding: utf-8 -*-
import os
import json
from flask import Flask, request
import numpy as np
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizer import Tokenizer
from bert4keras.bert import build_bert_model
from keras.layers import *
import tensorflow as tf
import albert.weibo_opinions.config as cur_process_config

conf = cur_process_config.create_params()
config_path = os.path.join(conf.model_dir, 'albert_config_small_google.json')
checkpoint_path = os.path.join(conf.model_dir, 'albert_model.ckpt')
dict_path = os.path.join(conf.model_dir, 'vocab.txt')
model_path = os.path.join(conf.model_dir, 'best_model.weights')

config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)
keras.backend.set_session(session)

# 加载预训练模型
bert = build_bert_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='albert',
    with_pool=True,
    return_keras_model=False,
)

output = Dropout(rate=0.1)(bert.model.output)
output = Dense(units=2,
               activation='softmax',
               kernel_initializer=bert.initializer)(output)

model = keras.models.Model(bert.model.input, output)
tokenizer = Tokenizer(dict_path, do_lower_case=True) # 建立分词器
model.load_weights(model_path)
# 编码测试
token_ids, segment_ids = tokenizer.encode(u'服了[失望]怎么在行政学院旁边也算超区[悲伤]差点扣我二十块钱#哈罗单车#  ​')
print('\n ===== predict test =====\n')
print(model.predict([np.array([token_ids]), np.array([segment_ids])]))
print('\n ===== predict test done =====\n')


app = Flask(__name__)
@app.route("/isValidOpinion", methods=['Get', 'POST'])
def isValidOpinion():
    # http://127.0.0.1:8330/isValidOpinion?query=为什么，走了二十分钟的路都没给我遇上电动的哈啰，路上停的全是普通的，别人手里拉的全是电动的？？？擦擦擦感觉我弟要砍我了说好45到的[doge]
    if request.method == 'POST':
        content = request.form['query']
    else:
        content = request.args.get('query')
    # 编码测试
    with session.as_default():
        with session.graph.as_default():
            token_ids, segment_ids = tokenizer.encode(content)
            pre_score = model.predict([np.array([token_ids]), np.array([segment_ids])])
            ret = pre_score.argmax(axis=1)
            score = pre_score[0]
            ret_dic = {"isValid": str(ret[0]), "score": str(score), "query": content}
            result = json.dumps(ret_dic, ensure_ascii=False, indent=4)
            print(result)
            return result
            # 编码测试
            # token_ids, segment_ids = tokenizer.encode(
            #     u'为什么，走了二十分钟的路都没给我遇上电动的哈啰，路上停的全是普通的，别人手里拉的全是电动的？？？擦擦擦感觉我弟要砍我了说好45到的[doge]')
            # print('\n ===== predicting =====\n')
            # print(model.predict([np.array([token_ids]), np.array([segment_ids])]))

    # stences = weibo_opinions_monitoring(content)
    # result = json.dumps(stences, encoding="UTF-8", ensure_ascii=False, indent=4)
    # result = json.dumps(stences, ensure_ascii=False, indent=4)
    # print (result)
    # return result
app.run(host='0.0.0.0',port=8330)