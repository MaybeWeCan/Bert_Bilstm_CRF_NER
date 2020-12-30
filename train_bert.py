# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

import os
import csv
import random
import collections
import pickle
import codecs
import logging
from tensorflow.contrib.layers.python.layers import initializers

import modeling
import optimization
import tokenization
from lstm_crf_layer import BLSTM_CRF
from arguments import *


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


# Bert提供的类
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[1])
                else:
                    if len(contends) == 0:
                        l = ' '.join([label for label in labels if len(label) > 0])
                        w = ' '.join([word for word in words if len(word) > 0])
                        lines.append([l, w])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
            return lines

class SelfProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self, files):
        labels = []
        with codecs.open(files, 'r', encoding='utf-8') as fd:
            for line in fd:
                labels.append(line.strip())

        return labels

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    def _read_data(self, input_file):

        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []


            for line in f:

                contends = line.strip()
                tokens = contends.split("\t")

                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[-1])
                else:
                    if len(contends) == 0 and len(words) > 0:
                        label = []
                        word = []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                word.append(w)
                        lines.append([' '.join(label), ' '.join(word)])
                        words = []
                        labels = []
                        continue
            return lines

def write_tokens(tokens, output_dir, mode):
    """
    将序列解析结果写入到文件中
    只在mode=test的时候启用
    :param tokens:
    :param mode:
    :return:
    """
    if mode == "test":
        path = os.path.join(output_dir, "token_" + mode + ".txt")
        wf = codecs.open(path, 'a', encoding='utf-8')
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + '\n')
        wf.close()


# 转变为InputFeatures 类
def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode):

    """
        将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
        :param ex_index: index
        :param example: 一个样本
        :param label_list: 标签列表
        :param max_seq_length:
        :param tokenizer:
        :param output_dir
        :param mode:
        :return:
        """

    label_map = {}
    for (i, label) in enumerate(label_list, 0):
        label_map[label] = i

    # 保存label->index 的map
    if not os.path.exists(os.path.join(output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    textlist = example.text.split(' ')
    labellist = example.label.split(' ')

    tokens = []
    labels = []

    for i, word in enumerate(textlist):
        # 分词，如果是中文，就是分字,但是对于一些不在BERT的vocab.txt中得字符会被进行WordPice处理（例如中文的引号），可以将所有的分字操作替换为list(input)
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]

        # 可能被word_pice
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:  # 一般不会出现else
                labels.append("X")

    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
        labels = labels[0:(max_seq_length - 2)]

    ntokens = []
    segment_ids = []
    label_ids = []

    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    label_ids.append(label_map["O"])


    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])

    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)

    label_ids.append(label_map["O"])

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式

    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:

        # Padding
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

        label_ids.append(label_map["O"])
        ntokens.append("**NULL**")


    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length


    # 打印部分样本数据信息
    if ex_index < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))


    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )

    return feature

def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, output_dir, mode=None):

    """
        将数据转化为TF_Record 结构，作为模型数据输入
        :param examples:  样本
        :param label_list:标签list
        :param max_seq_length: 预先设定的最大序列长度
        :param tokenizer: tokenizer 对象
        :param output_file: tf.record 输出路径
        :param mode:
        :return:
        """

    writer = tf.python_io.TFRecordWriter(output_file)

    # 遍历训练数据
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        # 对于每一个训练样本,
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)

        # features["label_mask"] = create_int_feature(feature.label_mask)
        # tf.train.Example/Feature 是一种协议，方便序列化？？？
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

    writer.close()


# 为Estimator服务
def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = arg_dic['train_batch_size']  # params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=300)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        # 提前从数据集中取出若干数据放到内存中，这样可以使在gpu计算时，cpu通过处理数据，从而提高训练的速度
        d = d.prefetch(buffer_size=4)

        return d

    return input_fn

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings,
                 dropout_rate=1.0, lstm_size=1, cell='lstm', num_layers=1):

    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # 获取对应的embedding 输入数据[batch_size, seq_length, embedding_size]
    embedding = model.get_sequence_output()
    max_seq_length = embedding.shape[1].value
    # 算序列真实长度
    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度
    # 添加CRF output layer

    blstm_crf = BLSTM_CRF(embedded_chars=embedding, hidden_unit=lstm_size, cell_type=cell, num_layers=num_layers,
                          dropout_rate=dropout_rate, initializers=initializers, num_labels=num_labels,
                          seq_length=max_seq_length, labels=labels, lengths=lengths, is_training=is_training)
    rst = blstm_crf.add_blstm_crf_layer(crf_only=True)

    return rst


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, ):


    def model_gpu(features, labels, mode, params):  # pylint: disable=unused-argument


        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # 按照原模型重构结构
        total_loss, logits, trans, pred_ids = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, False, arg_dic["dropout_rate"], arg_dic["lstm_size"], arg_dic["cell"],
            arg_dic["num_layers"])

        tvars = tf.trainable_variables()

        # initialized_variable_names = {}

        # 加载原有参数做初始化
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            init_checkpoint)

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        output_spec = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            # train_op = optimizer.optimizer(total_loss, learning_rate, num_train_steps)
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, False)

            hook_dict = {}
            hook_dict['train_loss'] = total_loss
            # hook_dict['global_steps'] = tf.train.get_or_create_global_step()

            # 以日志的形式输出一个或多个 tensor 的值。
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict,
                every_n_iter=arg_dic["save_summary_steps"])


            # # summary_hook
            # tf.summary.scalar('train_loss',total_loss)
            #
            # summary_op = tf.summary.merge_all()
            #
            # summary_hook = tf.train.SummarySaverHook(
            #     save_steps=arg_dic["save_summary_steps"],
            #     output_dir=arg_dic["train_summary_dir"],
            #     summary_op=summary_op,
            # )

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook])

        elif mode == tf.estimator.ModeKeys.EVAL:

            # 针对NER ,进行了修改
            # eval的是对应的指标,是否在这里计算F1值？

            # "eval_loss": tf.metrics.mean_squared_error(labels=label_ids, predictions=pred_ids),

            def metric_fn(label_ids, pred_ids):
                return {
                    'precision': tf.metrics.precision(label_ids, pred_ids),
                    'f1_score': tf_metrics.f1(labels, pred_ids, num_labels),
                }

            # 标量
            eval_metrics = metric_fn(label_ids, pred_ids)

            # 这样写原理是什么？因为这样也会tensroboard,
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics,
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=pred_ids
            )
        return output_spec

    return model_gpu

def main():

    ''' PrePare and check file'''

    # 检查checkpoint配置的准确性
    tokenization.validate_case_matches_checkpoint(arg_dic['do_lower_case'], arg_dic['init_checkpoint'])

    if not arg_dic['do_train'] and not arg_dic['do_eval'] and not arg_dic['do_predict']:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")


    # 导入Bert配置
    bert_config = modeling.BertConfig.from_json_file(arg_dic['bert_config_file'])



    if arg_dic['max_seq_length'] > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (arg_dic['max_seq_length'], bert_config.max_position_embeddings))


    ''' Estimator Config '''

    processors = {
        "ner": SelfProcessor
    }

    processor = processors[arg_dic["ner"]]()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=arg_dic["vocab_file"], do_lower_case=arg_dic["do_lower_case"])

    '''
        配置tf.Session的运算方式:
        
            log_device_placement: 打印出TensorFlow使用了那种操作
            inter_op_parallelism_threads: 设置线程一个操作内部并行运算的线程数，比如矩阵乘法，如果设置为０，则表示以最优的线程数处理
            intra_op_parallelism_threads: 设置多个操作并行运算的线程数，比如 c = a + b，d = e + f . 可以并行运算
            allow_soft_placement: 那么当运行设备不满足要求时，会自动分配GPU或者CPU
    '''
    session_config = tf.ConfigProto(
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)

    '''
        Estimator Config:
            
            model_dir: 存储模型参数，graph等的路径
            save_summary_steps: 每隔这么多步骤保存摘要
            save_checkpoints_steps: 每隔多少个step就存一次checkpoint
            
    '''

    run_config = tf.estimator.RunConfig(
        model_dir=arg_dic["ckpt_dir"],
        save_summary_steps=arg_dic["save_summary_steps"],
        save_checkpoints_steps=arg_dic["save_checkpoints_steps"],
        session_config=session_config
    )


    train_examples = None
    eval_examples = None
    num_train_steps = None
    num_warmup_steps = None


    ''' Load Data and Model about train and eval '''
    if arg_dic["do_train"] and arg_dic["do_eval"]:

        # train
        train_examples = processor.get_train_examples(arg_dic["data_dir"])

        num_train_steps = int(
            len(train_examples) *1.0 / arg_dic["train_batch_size"] * arg_dic["num_train_epochs"])

        if num_train_steps < 1:
            raise AttributeError('training data is so small...')

        num_warmup_steps = int(num_train_steps * arg_dic["warmup_proportion"])

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", arg_dic["train_batch_size"])
        logger.info("  Num steps = %d", num_train_steps)

        # eval
        eval_examples = processor.get_dev_examples(arg_dic["data_dir"])

        # 打印验证集数据信息
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", arg_dic["train_batch_size"])


    label_list = processor.get_labels(arg_dic["data_dir"]+"label.txt")

    ''' Model of Estimator'''
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=arg_dic["init_checkpoint"],
        learning_rate=arg_dic["learning_rate"],
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    params = {
        'batch_size': arg_dic["train_batch_size"]
    }

    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)


    ''' Train of Estimator'''
    if arg_dic["do_train"] and arg_dic["do_eval"]:

        '''data input_fn'''
        # 1. 将数据转化为tf_record 数据
        train_file = os.path.join(arg_dic["tfrecord_dir"], "train.tf_record")

        # 如果不存在train_record则生成
        if not os.path.exists(train_file):
            filed_based_convert_examples_to_features(train_examples,
                                                     label_list,
                                                     arg_dic["max_seq_length"],
                                                     tokenizer,
                                                     train_file,
                                                     arg_dic["tfrecord_dir"])

        # 2.读取record 数据，组成batch
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=arg_dic["max_seq_length"],
            is_training=True,
            drop_remainder=True)


        # 1. eval
        eval_file = os.path.join(arg_dic["tfrecord_dir"], "eval.tf_record")


        if not os.path.exists(eval_file):
            filed_based_convert_examples_to_features(
                eval_examples, label_list, arg_dic["max_seq_length"], tokenizer, eval_file, arg_dic["tfrecord_dir"])

        # 2. eval read
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=arg_dic["max_seq_length"],
            is_training=False,
            drop_remainder=False)

        '''estimator train'''

        '''
            max_steps_without_increase:如果没有增加的最大长是多少，如果超过了这个最大步长metric还是没有增加那么就会停止。
            eval_dir：默认是使用estimator.eval_dir目录，用于存放评估的summary file。
            run_every_secs：表示多长时间调用一次should_stop_fn
        '''

        early_stopping_hook = tf.estimator.experimental.stop_if_no_decrease_hook(
            estimator=estimator,
            metric_name='loss',
            max_steps_without_decrease=num_train_steps,
            eval_dir=None,
            min_steps=0,
            run_every_secs=None,
            run_every_steps=arg_dic["save_checkpoints_steps"])


        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                            max_steps=num_train_steps,
                                            hooks=[early_stopping_hook])

        '''
            throttle_secs：多少秒后又开始评估，如果没有新的 checkpoints 产生，则不评估，所以这个间隔是最小值。
        '''
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                          throttle_secs=arg_dic["eval_model_steps"])

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


    # 进行预测
    if arg_dic["do_predict"]:

        token_path = os.path.join(arg_dic["output_dir"], "token_test.txt")

        if os.path.exists(token_path):
            os.remove(token_path)

        with codecs.open(os.path.join(arg_dic["tfrecord_dir"], 'label2id.pkl'), 'rb') as rf:

            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        # 数据
        predict_examples = processor.get_test_examples(arg_dic["data_dir"])

        predict_file = os.path.join(arg_dic["output_dir"], "predict.tf_record")

        filed_based_convert_examples_to_features(predict_examples,
                                                 label_list,
                                                 arg_dic["max_seq_length"],
                                                 tokenizer,
                                                 predict_file,
                                                 arg_dic["output_dir"],
                                                 mode="test")

        logger.info("***** Running prediction*****")
        logger.info("  Num examples = %d", len(predict_examples))
        logger.info("  Batch size = %d", arg_dic["train_batch_size"])

        predict_drop_remainder = False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=arg_dic["max_seq_length"],
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn,checkpoint_path=".\output\ckpt\model.ckpt-30")

        output_predict_file = os.path.join(arg_dic["output_dir"], "label_test.txt")

        def result_to_pair(writer):
            print("********")
            print(predict_examples)
            for predict_line, prediction in zip(predict_examples, result):


                idx = 0
                line = ''
                line_token = str(predict_line.text).split(' ')
                label_token = str(predict_line.label).split(' ')

                len_seq = len(label_token)

                if len(line_token) != len(label_token):
                    logger.info(predict_line.text)
                    logger.info(predict_line.label)
                    break
                for id in prediction:
                    if idx >= len_seq:
                        break
                    if id == 0:
                        continue
                    curr_labels = id2label[id]
                    if curr_labels in ['[CLS]', '[SEP]']:
                        continue
                    try:
                        line += line_token[idx] + ' ' + label_token[idx] + ' ' + curr_labels + '\n'
                    except Exception as e:
                        logger.info(e)
                        logger.info(predict_line.text)
                        logger.info(predict_line.label)
                        line = ''
                        break
                    idx += 1
                writer.write(line + '\n')

        # 预测结果写入文件
        with codecs.open(output_predict_file, 'w', encoding='utf-8') as writer:
            result_to_pair(writer)

        import conlleval

        # predict的项
        eval_result = conlleval.return_report(output_predict_file)

        print(''.join(eval_result))

        # 写结果到文件中
        with codecs.open(os.path.join(arg_dic["output_dir"], 'predict_score.txt'), 'a', encoding='utf-8') as fd:
            fd.write(''.join(eval_result))




if __name__ == "__main__":

    # 在re train 的时候，才删除上一轮产出的文件，在predicted 的时候不做clean
    if arg_dic["clean"] and arg_dic["do_train"]:

        if os.path.exists(arg_dic["output_dir"]):

            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)

            try:
                # clean output dir
                del_file(arg_dic["output_dir"])

                # clean log dir
                del_file(arg_dic["log_dir"])

                print("Remove "+arg_dic["output_dir"] + " and " + arg_dic["log_dir"] +" file" + " Successfully! ")

            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and data.conf')
                exit(-1)

    #check output dir exists
    if not os.path.exists(arg_dic["output_dir"]):
        os.mkdir(arg_dic["output_dir"])

    if not os.path.exists(arg_dic["log_dir"]):
        os.mkdir(arg_dic["log_dir"])

    if not os.path.exists(arg_dic["ckpt_dir"]):
        os.mkdir(arg_dic["ckpt_dir"])

    if not os.path.exists(arg_dic["tfrecord_dir"]):
        os.mkdir(arg_dic["tfrecord_dir"])

    if not os.path.exists(arg_dic["train_summary_dir"]):
        os.mkdir(arg_dic["train_summary_dir"])


    '''loggging set'''

    #  创建一个logger,默认为root logger
    logger = logging.getLogger()
    # 调整终端输出级别,设置全局log级别为INFO。注意全局的优先级最高
    logger.setLevel(logging.INFO)

    # 日志格式： 日志时间，日志信息，设置时间输出格式
    formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    #  创建一个终端输出的handler
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)

    #  创建一个文件记录日志的handler
    fhlr = logging.FileHandler(arg_dic["log_file_path"])
    fhlr.setFormatter(formatter)

    logger.addHandler(chlr)
    logger.addHandler(fhlr)

    main()
