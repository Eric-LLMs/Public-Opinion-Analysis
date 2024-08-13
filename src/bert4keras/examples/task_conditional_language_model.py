#! -*- coding: utf-8 -*-
# bert做conditional language model任务
# 按类随机生成文本，这个demo的类别是情感极性（正／负）

from __future__ import print_function
import glob
import numpy as np
from tqdm import tqdm
import os, json, codecs, re
from bert4keras.backend import keras, K
from bert4keras.bert import build_bert_model
from bert4keras.tokenizer import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import uniout  # 打印中文
from keras.layers import *


# 模型配置
maxlen = 128
batch_size = 32
num_classes = 2
epochs = 20

# bert配置
config_path = '/root/e/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/e/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/e/bert/chinese_L-12_H-768_A-12/vocab.txt'

# 加载并精简词表，建立分词器
_token_dict = load_vocab(dict_path)  # 读取词典
token_dict, keep_words = {}, []  # keep_words是在bert中保留的字表

for t in ['[PAD]', '[UNK]', '[CLS]', '[SEP]']:
    token_dict[t] = len(token_dict)
    keep_words.append(_token_dict[t])

for t, _ in sorted(_token_dict.items(), key=lambda s: s[1]):
    if t not in token_dict:
        if len(t) == 3 and (Tokenizer._is_cjk_character(t[-1])
                            or Tokenizer._is_punctuation(t[-1])):
            continue
        token_dict[t] = len(token_dict)
        keep_words.append(_token_dict[t])

tokenizer = Tokenizer(token_dict, do_lower_case=True)  # 建立分词器


def load_data(filenames):
    D = []
    for filename in filenames:
        with codecs.open(filename, encoding='utf-8') as f:
            for l in f:
                text, label = l.strip().split('\t')
                if len(text) <= maxlen - 2:
                    D.append((text, int(label)))
                else:
                    tmp = ''
                    for t in re.findall(u'.*?[。！\n]', text):
                        if tmp and len(tmp) + len(t) > maxlen - 2:
                            D.append((tmp, int(label)))
                            tmp = ''
                        tmp += t
                    if tmp:
                        D.append((tmp, int(label)))
    return D


# 加载数据集
data = load_data([
    'datasets/sentiment/sentiment.train.data',
    'datasets/sentiment/sentiment.valid.data',
    'datasets/sentiment/sentiment.test.data',
])


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            text, label = self.data[i]
            token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids, batch_labels], None
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


c_in = Input(shape=(1, ))
c = Embedding(2, 128)(c_in)
c = Reshape((128, ))(c)

# Bert模型
model = build_bert_model(
    config_path,
    checkpoint_path,
    application='lm',
    keep_words=keep_words,  # 只保留keep_words中的字，精简原字表
    layer_norm_cond=c,
    additional_input_layers=c_in,
)

model.summary()

# 交叉熵作为loss，并mask掉输入部分的预测
y_in = model.input[0][:, 1:]  # 目标tokens
y_mask = model.get_layer('Sequence-Mask').output_mask[:, 1:]  # 目标mask
y = model.output[:, :-1]  # 预测tokens，预测与目标错开一位
cross_entropy = K.sparse_categorical_crossentropy(y_in, y)
cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

model.add_loss(cross_entropy)
model.compile(optimizer=Adam(1e-5))


def random_generate(c=0, n=1, topk=5):
    """随机采样生成
    每次从最高概率的topk个token中随机采样一个
    """
    label_ids = [[c] for _ in range(n)]
    target_ids = [[tokenizer._token_cls_id] for _ in range(n)]
    R = []
    for i in range(maxlen):
        segment_ids = [[0] * len(t) for t in target_ids]
        # 下面直接忽略[PAD], [UNK], [CLS]
        _probas = model.predict([target_ids, segment_ids, label_ids])[:, -1, 3:]
        for i, p in enumerate(_probas):
            p_arg_topk = p.argsort()[::-1][:topk]
            p_topk = p[p_arg_topk]
            p = p_topk / sum(p_topk)
            idx = np.random.choice(len(p), p=p)
            target_ids[i].append(p_arg_topk[idx] + 3)
        for t in target_ids:
            if t[-1] == 3:
                R.append(tokenizer.decode(t))
        target_ids = [t for t in target_ids if t[-1] != 3]
        if len(target_ids) == 0:
            break
    return R


def just_show():
    print(u'正面采样:')
    print(random_generate(1, 5, 5), '\n')
    print(u'负面采样:')
    print(random_generate(0, 5, 5), '\n')


class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model.weights')
        # 演示效果
        just_show()


if __name__ == '__main__':

    evaluator = Evaluate()
    train_generator = data_generator(data, batch_size)

    model.fit_generator(train_generator.forfit(),
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        callbacks=[evaluator])

else:

    model.load_weights('./best_model.weights')
