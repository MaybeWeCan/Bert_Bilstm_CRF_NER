# 一、介绍



> **简介**本代码是Bert-Bilstm-CRF-Ner的一个简单实现，主要架构基于Estimator
>
> **环境**tensorflow 1.14



# 二、简单运行

（1）下载Bert模型

​			Bert模型[BERT-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip): Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M 



（2）修改配置文件（arguments.py）

​		 	根据需要，修改对应参数。



（3）运行（main）

​		**运行前需要修改（clean , do_train, do_test, do_predict）这四个参数，根据运行目的自行修改。**

       * 运行train_bert.py文件
       * 注：train_bert_new.py为自己尝试添加某个组件但是失败的副本。



# 三、高级

## 3.1 数据

​    （1）样本格式(一般都是这种)

```
就	O
从	O
这	O
```



​    （2）实体标签（./data/label.xtxt）

​		根据自己的修改

```
O
I-LOC
B-LOC
I-ORG
B-ORG
I-PER
B-PER
```



## 3.2 日志+tensorboard

> 需要知道的一点是，estimator会默认将 eval 和 train 的loss打印出来，并且画到tensorboard里。如果需要其他信息可以采用下面的方法自己添加，下面只是举例子



  	在训练和预测时我们需要可视化很多信息，可以打印在日志里，也可以tensorboard显示。主要原理是基于tf.estimator.EstimatorSpec里的hook参数。

（1）train打印train_loss到日志里

```
hook_dict = {}
hook_dict['train_loss'] = total_loss

# 以日志的形式输出一个或多个 tensor 的值。
logging_hook = tf.train.LoggingTensorHook(
    hook_dict,
    every_n_iter=arg_dic["save_summary_steps"])

output_spec = tf.estimator.EstimatorSpec(
    mode=mode,
    loss=total_loss,
    train_op=train_op,
    training_hooks=[logging_hook])
```



**当然也可以替换成下面的代码，画图tensorboard**



```
# summary_hook
tf.summary.scalar('train_loss',total_loss)


summary_op = tf.summary.merge_all()


summary_hook = tf.train.SummarySaverHook(
    save_steps=arg_dic["save_summary_steps"],
    output_dir=arg_dic["train_summary_dir"],
    summary_op=summary_op,
)


output_spec = tf.estimator.EstimatorSpec(
    mode=mode,
    loss=total_loss,
    train_op=train_op,
    training_hooks=[summary_hook])
```





(2) eval时显示各个指标 (下面的做法，这些指标既会打印到log里，也会在tensorboard里显示)

```
def metric_fn(label_ids, pred_ids):
    return {
        "eval_loss": tf.metrics.mean_squared_error(labels=label_ids, predictions=pred_ids),
        'precision': tf.metrics.precision(label_ids, pred_ids),
    }


eval_metrics = metric_fn(label_ids, pred_ids)


# 这样写原理是什么？因为这样也会tensroboard,
output_spec = tf.estimator.EstimatorSpec(
    mode=mode,
    loss=total_loss,
    eval_metric_ops=eval_metrics
)
```



**这里指标的构建可以参考：**     https://blog.csdn.net/pipisorry/article/details/95178966
    

```
metrics = {      
'acc': tf.metrics.accuracy(tf.argmax(labels), tf.argmax(pred_ids)),      
'precision': tf.metrics.precision(tf.argmax(labels), tf.argmax(pred_ids)),      'precision_': tf_metrics.precision(tf.argmax(labels), tf.argmax(pred_ids), num_labels),    'recall': tf.metrics.recall(tf.argmax(labels), tf.argmax(pred_ids)),      
'recall_': tf_metrics.recall(tf.argmax(labels), tf.argmax(pred_ids), num_labels),      'f1_': tf_metrics.f1(tf.argmax(labels), tf.argmax(pred_ids), num_labels),      
'auc': tf.metrics.auc(labels, pred_ids),    }
```





# 四、暂未解决

（1）test只关于实体的F1计算

      这个代码参考的github只是实现了对于预测结果生成的文件，做关于实体的F1_score，自己在test添加时，加不进去，原因是Estimator将构图和sess分开，在用estimator时只需要考虑图于tensor的value取不出来，也不清楚如何触发data.get_nextbatch这个事件，因此无法在eval代码中循环得到数据，各种尝试都报错，暂时没有解决。



（2）代码没有实现单条预测实现（暂时没这个需求）



（3）准备再出一个不用高级API，更加灵活的版本。

























