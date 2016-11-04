# -*- encoding:utf-8 -*-
from qacnn import QACNN
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import operator
import insurance_qa_data_helpers

# Config函数
class Config(object):
    def __init__(self, vocab_size):
        # 输入序列(句子)长度
        self.sequence_length = 200
        # 循环数
        self.num_epochs = 100000
        # batch大小
        self.batch_size = 100
        # 词表大小
        self.vocab_size = vocab_size
        # 词向量大小
        self.embedding_size = 100
        # 不同类型的filter,相当于1-gram,2-gram,3-gram和5-gram
        self.filter_sizes = [1, 2, 3, 5]
        # 隐层大小
        self.hidden_size = 80
        # 每种filter的数量
        self.num_filters = 512
        # L2正则化,未用,没啥效果
        # 论文里给的是0.0001
        self.l2_reg_lambda = 0.
        # 弃权,未用,没啥效果
        self.keep_prob = 1.0
        # 学习率
        # 论文里给的是0.01
        self.lr = 0.01
        # margin
        # 论文里给的是0.009
        self.m = 0.05
        # 设定GPU的性质,允许将不能在GPU上处理的部分放到CPU
        # 设置log打印
        self.cf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # 只占用20%的GPU内存
        self.cf.gpu_options.per_process_gpu_memory_fraction = 0.2


print 'Loading Data...'


# 词映射ID
vocab = insurance_qa_data_helpers.build_vocab()
# 只记录train里的回答
alist = insurance_qa_data_helpers.read_alist()
# raw语料,记录所有train里的raw数据
raw = insurance_qa_data_helpers.read_raw()

testList, vectors = insurance_qa_data_helpers.load_test_and_vectors()
print 'Loading Data Done!'

# 测试目录
val_file = 'insuranceQA/test1'

# 配置文件
config = Config(len(vocab))


# 开始训练和测试
with tf.device('/gpu:0'):
    with tf.Session(config=config.cf) as sess:
        # 建立CNN网络
        cnn = QACNN(config, sess)
        # 训练函数
        def train_step(x_batch_1, x_batch_2, x_batch_3):
            feed_dict = {
                cnn.q: x_batch_1,
                cnn.aplus: x_batch_2,
                cnn.aminus: x_batch_3,
                cnn.keep_prob: config.keep_prob
            }
            _, step, loss, accuracy = sess.run(
                [cnn.train_op, cnn.global_step, cnn.loss, cnn.accu],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print "{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy)
        # 测试函数
        def dev_step():
            scoreList = list()
            i = 0
            while True:
                x_test_1, x_test_2, x_test_3 = insurance_qa_data_helpers.load_data_val_6(testList, vocab, i, config.batch_size)
                feed_dict = {
                    cnn.q: x_test_1,
                    cnn.aplus: x_test_2,
                    cnn.aminus: x_test_3,
                    cnn.keep_prob: 1.0
                }
                batch_scores = sess.run([cnn.q_ap_cosine], feed_dict)
                for score in batch_scores[0]:
                    scoreList.append(score)
                i += config.batch_size
                if i >= len(testList):
                    break
            sessdict = {}
            index = 0
            for line in open(val_file):
                items = line.strip().split(' ')
                qid = items[1].split(':')[1]
                if not qid in sessdict:
                    sessdict[qid] = list()
                sessdict[qid].append((scoreList[index], items[0]))
                index += 1
                if index >= len(testList):
                    break
            lev1 = .0
            lev0 = .0
            for k, v in sessdict.items():
                v.sort(key=operator.itemgetter(0), reverse=True)
                score, flag = v[0]
                if flag == '1':
                    lev1 += 1
                if flag == '0':
                    lev0 += 1
            # 回答的正确数和错误数
            print '回答正确数 ' + str(lev1)
            print '回答错误数 ' + str(lev0)
            print '准确率 ' + str(float(lev1)/(lev1+lev0))

        # 每5000步测试一下
        evaluate_every = 5000
        # 开始训练和测试
        sess.run(tf.initialize_all_variables())
        for i in range(config.num_epochs):
            # 18540个训练样本
            # 20000+个预训练词向量,此处没有用,不过可以加进去
            x_batch_1, x_batch_2, x_batch_3 = insurance_qa_data_helpers.load_data_6(vocab, alist, raw, config.batch_size)
            train_step(x_batch_1, x_batch_2, x_batch_3)
            if (i+1) % evaluate_every == 0:
                # 共20个问题,每个问题500个,对应1到2个正确答案,499到498个错误答案
                # 相当于从一个size=500的pool里选出正确答案
                print "\n测试{}:".format((i+1)/evaluate_every)
                dev_step()
                print

