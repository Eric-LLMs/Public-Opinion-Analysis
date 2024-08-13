#! -*- coding: utf-8 -*-
# bert做language model任务，小说生成

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


lm_config = 'lm_config.json'
min_count = 8
maxlen = 256
batch_size = 16
steps_per_epoch = 1000
epochs = 10000

# bert配置
config_path = '/root/e/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/e/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/e/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'


novels = []

for txt in glob.glob('/root/金庸/*/*.txt'):
    txt = open(txt).read()
    txt = txt.decode('gbk', 'ignore')
    txt = txt.replace('\r', '').replace('\n', '')
    txt = txt.replace(u'整理制作，并提供下载', '')
    txt = re.sub(u'www.*?com', '', txt)
    txt = txt.replace(u'\u3000', ' ')
    sents = []
    for t in txt.split('  '):
        for s in re.findall(u'.*?。', t):
            if len(s) <= maxlen - 2:
                sents.append(s)
    novels.append(sents)


_token_dict = load_vocab(dict_path)  # 读取词典
_tokenizer = Tokenizer(_token_dict, do_lower_case=True)  # 建立临时分词器

if os.path.exists(lm_config):
    tokens = json.load(open(lm_config))
else:
    tokens = {}
    for novel in novels:
        for s in novel:
            for t in _tokenizer.tokenize(s):
                tokens[t] = tokens.get(t, 0) + 1
    tokens = [(i, j) for i, j in tokens.items() if j >= min_count]
    tokens = sorted(tokens, key=lambda t: -t[1])
    tokens = [t[0] for t in tokens]
    json.dump(tokens,
              codecs.open(lm_config, 'w', encoding='utf-8'),
              indent=4,
              ensure_ascii=False)

token_dict, keep_words = {}, []  # keep_words是在bert中保留的字表

for t in ['[PAD]', '[UNK]', '[CLS]', '[SEP]']:
    token_dict[t] = len(token_dict)
    keep_words.append(_token_dict[t])

for t in tokens:
    if t in _token_dict and t not in token_dict:
        token_dict[t] = len(token_dict)
        keep_words.append(_token_dict[t])

tokenizer = Tokenizer(token_dict, do_lower_case=True)  # 建立分词器


data = []
pbar = tqdm(desc=u'构建语料中', total=sum(len(n) for n in novels))

for novel in novels:
    s = u''
    for i in range(len(novel)):
        for j in range(len(novel) - i):
            if len(s) + len(novel[i + j]) > maxlen - 2:
                data.append(s)
                s = u''
                break
            else:
                s += novel[i + j]
        pbar.update(1)
        if i + j >= len(novel):
            break
    if s:
        data.append(s)

pbar.close()
np.random.shuffle(data)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids = [], []
        for i in idxs:
            text = self.data[i]
            token_ids, segment_ids = tokenizer.encode(text)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


model = build_bert_model(
    config_path,
    checkpoint_path,
    application='lm',
    keep_words=keep_words,  # 只保留keep_words中的字，精简原字表
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


def random_generate(s, n=1, topk=5):
    """随机采样生成
    每次从最高概率的topk个token中随机采样一个
    """
    token_ids, segment_ids = tokenizer.encode(s)
    token_ids, segment_ids = token_ids[:-1], segment_ids[:-1]
    target_ids = [[] for _ in range(n)]
    R = []
    for i in range(maxlen):
        _target_ids = [token_ids + t for t in target_ids]
        _segment_ids = [segment_ids + [0] * len(t) for t in target_ids]
        # 下面直接忽略[PAD], [UNK], [CLS]
        _probas = model.predict([_target_ids, _segment_ids])[:, -1, 3:]
        for i, p in enumerate(_probas):
            p_arg_topk = p.argsort()[::-1][:topk]
            p_topk = p[p_arg_topk]
            p = p_topk / sum(p_topk)
            idx = np.random.choice(len(p), p=p)
            target_ids[i].append(p_arg_topk[idx] + 3)
        for t in target_ids:
            if t[-1] == 3:
                R.append(tokenizer.decode(token_ids + t))
        target_ids = [t for t in target_ids if t[-1] != 3]
        if len(target_ids) == 0:
            break
    for t in target_ids:
        R.append(tokenizer.decode(token_ids + t))
    return R


def just_show():
    s1 = u'Training large deep neural networks on massive datasets is computationally very challenging.'
    s2 = u'Training large deep neural networks on massive datasets is computationally very challenging.Training large deep neural networks on massive datasets is computationally very challenging.。'
    s3 = u'Training large deep neural networks on massive datasets is computationally very challenging.Training large deep neural networks on massive datasets is computationally very challenging.Training large deep neural networks on massive datasets is computationally very challenging.。'
    for s in [s1, s2, s3]:
        t = random_generate(s)
        print(u'输入: %s' % s)
        print(u'结果: %s\n' % ('\n'.join(t)))


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
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        callbacks=[evaluator])

else:

    model.load_weights('./best_model.weights')

"""
效果：

输入:  
结果: 

输入:  
结果:

输入:  
结果:

"""
