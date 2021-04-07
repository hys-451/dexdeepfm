"""define train, infer, eval, test process"""
import numpy as np
import os, time, collections
import tensorflow as tf
from IO.iterator import FfmIterator #, DinIterator, CCCFNetIterator

from IO.ffm_cache import FfmCache
from src.exDeepFM import ExtremeDeepFMModel
from src.CIN import CINModel
import utils.util as util
import utils.metric as metric
# from utils.log import Log

# log = Log(hparams)

class TrainModel(collections.namedtuple("TrainModel", ("graph", "model", "iterator", "filenames"))):
    """define train class, include graph, model, iterator"""
    pass


def create_train_model(model_creator, hparams, scope=None):
    graph = tf.Graph()
    with graph.as_default():
        # feed train file name, valid file name, or test file name
        filenames = tf.placeholder(tf.string, shape=[None])
        src_dataset = tf.contrib.data.TFRecordDataset(filenames)
        #src_dataset = tf.data.TFRecordDataset(filenames)

        if hparams.data_format == 'ffm':
            batch_input = FfmIterator(src_dataset)
        elif hparams.data_format == 'din':
            batch_input = DinIterator(src_dataset)
        elif hparams.data_format == 'cccfnet':
            batch_input = CCCFNetIterator(src_dataset)
        else:
            raise ValueError("not support {0} format data".format(hparams.data_format))
        # build model
        model = model_creator(
            hparams,
            iterator=batch_input,
            scope=scope)

    return TrainModel(
        graph=graph,
        model=model,
        iterator=batch_input,
        filenames=filenames)


# run evaluation and get evaluted loss
def run_eval(load_model, load_sess, filename, sample_num_file, hparams, flag):
    # load sample num
    with open(sample_num_file, 'r') as f:
        sample_num = int(f.readlines()[0].strip())
    load_sess.run(load_model.iterator.initializer, feed_dict={load_model.filenames: [filename]})
    preds = []
    labels = []
    while True:
        try:
            _, _, _, step_pred, step_labels = load_model.model.eval(load_sess)#########################5
            preds.extend(np.reshape(step_pred, -1))
            labels.extend(np.reshape(step_labels, -1))
        except tf.errors.OutOfRangeError:
            break
    preds = preds[:sample_num]
    labels = labels[:sample_num]
    hparams.logger.info("data num:{0:d}".format(len(labels)))
    res = metric.cal_metric(labels, preds, hparams, flag)
    return res


# run infer
def run_infer(load_model, load_sess, filename, hparams, sample_num_file):
    # load sample num
    with open(sample_num_file, 'r') as f:
        sample_num = int(f.readlines()[0].strip())
    if not os.path.exists(util.RES_DIR):
        os.mkdir(util.RES_DIR)
    load_sess.run(load_model.iterator.initializer, feed_dict={load_model.filenames: [filename]})
    preds = []
    while True:
        try:
            step_pred = load_model.model.infer(load_sess)
            preds.extend(np.reshape(step_pred, -1))
        except tf.errors.OutOfRangeError:
            break
    preds = preds[:sample_num]
    hparams.res_name = util.convert_res_name(hparams.infer_file)
    # print('result name:', hparams.res_name)
    with open(hparams.res_name, 'w') as out:
        out.write('\n'.join(map(str, preds)))


# cache data
def cache_data(hparams, filename, flag):
    if hparams.data_format == 'ffm':
        cache_obj = FfmCache()
    elif hparams.data_format == 'din':
        cache_obj = DinCache()
    elif hparams.data_format == 'cccfnet':
        cache_obj = CCCFNetCache()
    else:
        raise ValueError(
            "data format must be ffm, din, cccfnet, this format not defined {0}".format(hparams.data_format))
    if not os.path.exists(util.CACHE_DIR):
        os.mkdir(util.CACHE_DIR)
    if flag == 'train':
        hparams.train_file_cache = util.convert_cached_name(hparams.train_file, hparams.batch_size)
        cached_name = hparams.train_file_cache
        sample_num_path = util.TRAIN_NUM
        impression_id_path = util.TRAIN_IMPRESSION_ID
    elif flag == 'eval':
        hparams.eval_file_cache = util.convert_cached_name(hparams.eval_file, hparams.batch_size)
        cached_name = hparams.eval_file_cache
        sample_num_path = util.EVAL_NUM
        impression_id_path = util.EVAL_IMPRESSION_ID
    elif flag == 'test':
        hparams.test_file_cache = util.convert_cached_name(hparams.test_file, hparams.batch_size)
        cached_name = hparams.test_file_cache
        sample_num_path = util.TEST_NUM
        impression_id_path = util.TEST_IMPRESSION_ID
    elif flag == 'infer':
        hparams.infer_file_cache = util.convert_cached_name(hparams.infer_file, hparams.batch_size)
        cached_name = hparams.infer_file_cache
        sample_num_path = util.INFER_NUM
        impression_id_path = util.INFER_IMPRESSION_ID
    else:
        raise ValueError("flag must be train, eval, test, infer")
    print('cache filename:', filename)
    if not os.path.isfile(cached_name):
        print('has not cached file, begin cached...')
        start_time = time.time()
        sample_num, impression_id_list = cache_obj.write_tfrecord(filename, cached_name, hparams)
        util.print_time("caced file used time", start_time)
        print("data sample num:{0}".format(sample_num))
        with open(sample_num_path, 'w') as f:
            f.write(str(sample_num) + '\n')
        with open(impression_id_path, 'w') as f:
            for impression_id in impression_id_list:
                f.write(str(impression_id) + '\n')


def train(hparams, scope=None, target_session=""):
    params = hparams.values()
    for key, val in params.items():
        hparams.logger.info(str(key) + ':' + str(val))

    print('load and cache data...')
    if hparams.train_file is not None:
        cache_data(hparams, hparams.train_file, flag='train')
    if hparams.eval_file is not None:
        cache_data(hparams, hparams.eval_file, flag='eval')
    if hparams.test_file is not None:
        cache_data(hparams, hparams.test_file, flag='test')
    if hparams.infer_file is not None:
        cache_data(hparams, hparams.infer_file, flag='infer')

    if hparams.model_type == 'dexDeepFM':
        print("run dexdeepFM model!")
        model_creator = ExtremeDeepFMModel

    else:
        raise ValueError("model type should be dexdeepFM")

    # define train,eval,infer graph
    # define train session, eval session, infer session
    train_model = create_train_model(model_creator, hparams, scope)
    gpuconfig = tf.ConfigProto()
    gpuconfig.gpu_options.allow_growth = True
    tf.set_random_seed(1234)
    train_sess = tf.Session(target=target_session, graph=train_model.graph, config=gpuconfig)

    train_sess.run(train_model.model.init_op)
    #load model from checkpoint ####################hparams.load_model_name==None###############
    if not hparams.load_model_name is None:
        checkpoint_path = hparams.load_model_name
        try:
            train_model.model.saver.restore(train_sess, checkpoint_path)
            print('load model', checkpoint_path)
        except:
            raise IOError("Failed to find any matching files for {0}".format(checkpoint_path))
    ####################################################################################################
    print('total_loss = data_loss+regularization_loss-diversity_loss, data_loss = {rmse or logloss ..}')
    writer = tf.summary.FileWriter(util.SUMMARIES_DIR, train_sess.graph)
    last_eval = 0
    for epoch in range(hparams.epochs):
        step = 0
        train_sess.run(train_model.iterator.initializer, feed_dict={train_model.filenames: [hparams.train_file_cache]})
        epoch_loss = 0
        train_start = time.time()
        train_load_time = 0
        while True:
            try:
                t1 = time.time()
                step_result = train_model.model.train(train_sess)
                t3 = time.time()
                train_load_time += t3 - t1
                (_, step_loss, step_data_loss, step_diversity_loss, step_regular_loss, summary, final_result) = step_result
                #self.update, self.loss, self.data_loss, self.merged
                writer.add_summary(summary, step)
                epoch_loss += step_loss
                step += 1
                if step % hparams.show_step == 0:
                    print('step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}, diversity_loss: {3:.4f}, regular_loss: {4:.4f}' \
                          .format(step, step_loss, step_data_loss, step_diversity_loss, step_regular_loss))
                    #print(final_result)
            except tf.errors.OutOfRangeError:
                print('finish one epoch!')
                break
        train_end = time.time()
        train_time = train_end - train_start
        if epoch % hparams.save_epoch == 0:
            checkpoint_path = train_model.model.saver.save(
                sess=train_sess,
                save_path=util.MODEL_DIR + 'epoch_' + str(epoch))

        train_res = dict()
        train_res["loss"] = epoch_loss / step
        eval_start = time.time()
        # train_res = run_eval(train_model, train_sess, hparams.train_file_cache, util.TRAIN_NUM, hparams, flag='train')
        eval_res = run_eval(train_model, train_sess, hparams.eval_file_cache, util.EVAL_NUM, hparams, flag='eval')
        train_info = ', '.join(
            [str(item[0]) + ':' + str(item[1])
             for item in sorted(train_res.items(), key=lambda x: x[0])])
        eval_info = ', '.join(
            [str(item[0]) + ':' + str(item[1])
             for item in sorted(eval_res.items(), key=lambda x: x[0])])
        if hparams.test_file is not None:
            test_res = run_eval(train_model, train_sess, hparams.test_file_cache, util.TEST_NUM, hparams, flag='test')
            test_info = ', '.join(
                [str(item[0]) + ':' + str(item[1])
                 for item in sorted(test_res.items(), key=lambda x: x[0])])
        eval_end = time.time()
        eval_time = eval_end - eval_start
        if hparams.test_file is not None:
            print('at epoch {0:d}'.format(
                epoch) + ' train info: ' + train_info + ' eval info: ' + eval_info + ' test info: ' + test_info)
            hparams.logger.info('at epoch {0:d}'.format(
                epoch) + ' train info: ' + train_info + ' eval info: ' + eval_info + ' test info: ' + test_info)
        else:
            print('at epoch {0:d}'.format(epoch) + ' train info: ' + train_info + ' eval info: ' + eval_info)
            hparams.logger.info('at epoch {0:d}'.format(epoch) + ' train info: ' + train_info + ' eval info: ' + eval_info)
        print('at epoch {0:d} , train time: {1:.1f} eval time: {2:.1f}'.format(epoch, train_time, eval_time))

        hparams.logger.info('at epoch {0:d} , train time: {1:.1f} eval time: {2:.1f}' \
                    .format(epoch, train_time, eval_time))
        hparams.logger.info('\n')

        if eval_res["auc"] - last_eval < - 0.003:
            break
        if eval_res["auc"] > last_eval:
            last_eval = eval_res["auc"]
    writer.close()
    # after train,run infer
    if hparams.infer_file is not None:
        run_infer(train_model, train_sess, hparams.infer_file_cache, hparams, util.INFER_NUM)
