#! -*- coding: utf-8 -*-
# bert做Seq2Seq任务，采用UNILM方案

from __future__ import print_function
import glob
import numpy as np
from tqdm import tqdm
import os, json, codecs
from bert4keras.backend import keras, K
from bert4keras.bert import build_bert_model
from bert4keras.tokenizer import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import parallel_apply, sequence_padding
from bert4keras.snippets import DataGenerator


seq2seq_config = 'seq2seq_config.json'
min_count = 128
maxlen = 256
batch_size = 16
steps_per_epoch = 1000
epochs = 10000

# bert配置
config_path = '/root/e/bert/chinese_wwm_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/e/bert/chinese_wwm_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/e/bert/chinese_wwm_L-12_H-768_A-12/vocab.txt'

# 训练样本。THUCNews数据集，每个样本保存为一个txt。
txts = glob.glob('/root/thuctc/THUCNews/*/*.txt')


_token_dict = load_vocab(dict_path)  # 读取词典
_tokenizer = Tokenizer(_token_dict, do_lower_case=True)  # 建立临时分词器

if os.path.exists(seq2seq_config):

    tokens = json.load(open(seq2seq_config))

else:

    def _batch_texts():
        texts = []
        for txt in txts:
            text = codecs.open(txt, encoding='utf-8').read()
            texts.append(text)
            if len(texts) == 100:
                yield texts
                texts = []
        if texts:
            yield texts

    def _tokenize_and_count(texts):
        _tokens = {}
        for text in texts:
            for token in _tokenizer.tokenize(text):
                _tokens[token] = _tokens.get(token, 0) + 1
        return _tokens

    tokens = {}

    def _total_count(result):
        for k, v in result.items():
            tokens[k] = tokens.get(k, 0) + v

    # 10进程来完成词频统计
    parallel_apply(
        func=_tokenize_and_count,
        iterable=tqdm(_batch_texts(), desc=u'构建词汇表中'),
        workers=10,
        max_queue_size=100,
        callback=_total_count,
        # dummy=True,  # 如果在Windows跑，请设置dummy=True
    )

    tokens = [(i, j) for i, j in tokens.items() if j >= min_count]
    tokens = sorted(tokens, key=lambda t: -t[1])
    tokens = [t[0] for t in tokens]
    json.dump(tokens,
              codecs.open(seq2seq_config, 'w', encoding='utf-8'),
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


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids = [], []
        for i in idxs:
            txt = self.data[i]
            text = codecs.open(txt, encoding='utf-8').read()
            text = text.split('\n')
            if len(text) > 1:
                title = text[0]
                content = '\n'.join(text[1:])
                token_ids, segment_ids = tokenizer.encode(content,
                                                          title,
                                                          max_length=maxlen)
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
    application='seq2seq',
    keep_words=keep_words,  # 只保留keep_words中的字，精简原字表
)

model.summary()

# 交叉熵作为loss，并mask掉输入部分的预测
y_in = model.input[0][:, 1:]  # 目标tokens
y_mask = model.input[1][:, 1:]
y = model.output[:, :-1]  # 预测tokens，预测与目标错开一位
cross_entropy = K.sparse_categorical_crossentropy(y_in, y)
cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

model.add_loss(cross_entropy)
model.compile(optimizer=Adam(1e-5))


def gen_sent(s, topk=2, title_maxlen=32):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    content_maxlen = maxlen - title_maxlen
    token_ids, segment_ids = tokenizer.encode(s, max_length=content_maxlen)
    target_ids = [[] for _ in range(topk)]  # 候选答案id
    target_scores = [0] * topk  # 候选答案分数
    for i in range(title_maxlen):  # 强制要求输出不超过title_maxlen字
        _target_ids = [token_ids + t for t in target_ids]
        _segment_ids = [segment_ids + [1] * len(t) for t in target_ids]
        _probas = model.predict([_target_ids, _segment_ids
                                 ])[:, -1, 3:]  # 直接忽略[PAD], [UNK], [CLS]
        _log_probas = np.log(_probas + 1e-6)  # 取对数，方便计算
        _topk_arg = _log_probas.argsort(axis=1)[:, -topk:]  # 每一项选出topk
        _candidate_ids, _candidate_scores = [], []
        for j, (ids, sco) in enumerate(zip(target_ids, target_scores)):
            # 预测第一个字的时候，输入的topk事实上都是同一个，
            # 所以只需要看第一个，不需要遍历后面的。
            if i == 0 and j > 0:
                continue
            for k in _topk_arg[j]:
                _candidate_ids.append(ids + [k + 3])
                _candidate_scores.append(sco + _log_probas[j][k])
        _topk_arg = np.argsort(_candidate_scores)[-topk:]  # 从中选出新的topk
        target_ids = [_candidate_ids[k] for k in _topk_arg]
        target_scores = [_candidate_scores[k] for k in _topk_arg]
        best_one = np.argmax(target_scores)
        if target_ids[best_one][-1] == 3:
            return tokenizer.decode(target_ids[best_one])
    # 如果title_maxlen字都找不到结束符，直接返回
    return tokenizer.decode(target_ids[np.argmax(target_scores)])


def just_show():
    s1 = u' '
    s2 = u' '
    for s in [s1, s2]:
        print(u'生成标题:', gen_sent(s))
    print()


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
    train_generator = data_generator(txts, batch_size)

    model.fit_generator(train_generator.forfit(),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        callbacks=[evaluator])

else:

    model.load_weights('./best_model.weights')
