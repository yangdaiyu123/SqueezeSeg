#!-*- coding=utf-8 -*-
# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Train"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import sys
import time

import math
import numpy as np
from six.moves import xrange
import tensorflow as tf
import threading

from config import *
from imdb import kitti
from utils.util import *
from nets import *
from config import *

from config.project_config import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_set', 'train',
                           """ Can be train, trainval, val, or test""")

tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Maximum number of batches to run.""")

tf.app.flags.DEFINE_integer('checkpoint_step', 5000,
                            """Number of steps to save summary.""")

tf.app.flags.DEFINE_integer('summary_step', 100,
                            """Number of steps to save summary.""")

tf.app.flags.DEFINE_integer('gpu', 4, """gpu id.""")
tf.app.flags.DEFINE_integer('gpu_wing', 2, """gpu id.""")


class train(object):
    
    def __init__(self):
        pass
    
    
    @property
    def mc(self):
        return self._mc

    @property
    def model(self):
        return self._model

    @property
    def imdb(self):
        return self._imdb



    def train_loop(self):
        gpu_idx = [FLAGS.gpu]
    
        tf.reset_default_graph()
        with tf.Graph().as_default():
            
            mc, imdb = self.setup_config()
            global_step = 0 # step for all queue
            
            # for idx, gpu in enumerate(gpu_idx):
            model = SqueezeSeg(mc, gpu_id=FLAGS.gpu)
            self._model = model
            
            start_step = self.train_initialize()
            global_step = start_step
            
            self.train_action(self._sess)
            self.step_loop(start_step)

    def setup_config(self):
    
        mc = alibaba_squeezeSeg_config()
        mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
        imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)
        
        
        print("\n data path: " + imdb._ali_path)
        print("pretrained model path :" + mc.PRETRAINED_MODEL_PATH)
        print("train_dir :" + FLAGS.train_dir + '\n')
    
        self._imdb = imdb
        self._mc = mc
        
        return mc, imdb

    def train_initialize(self):
        
        model = self._model
    
        # saver = tf.train.Saver(tf.all_variables())
        saver = tf.train.Saver(model.model_params)
        # Op是利用tensor对象执行运算的节点
        # 计算完毕后，会返回０个或多个张量
        # 可在以后为数据流图的其他Op所使用
        
        # 初始化op
        # vars = tf.initialize_all_variables()
        vars = tf.global_variables_initializer()
        
        # 将所有数据合并到一个op中去
        # 将所有summary全部保存到磁盘
        summary_op = tf.summary.merge_all()
        
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[-1])
        else:
            initial_step = 0

        print("start step is %d" % initial_step)
    
        # 用来保存图
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        # 初始化Op
        sess.run(vars)
        
        self._sess = sess
        self._saver = saver
        self._summary_op = summary_op
        self._summary_writer = summary_writer
        
        return initial_step

    
    def queue_train(self):
        mc = self._mc
        sess = self._sess
        
        def enqueue(sess, coord):
            with coord.stop_on_exception():
                while not coord.should_stop():
                    self.train_action(sess)
    
        coord = tf.train.Coordinator() # 创建一个协调器，管理线程
        enq_threads = []
        for _ in range(mc.NUM_ENQUEUE_THREAD):
            eqth = threading.Thread(target=enqueue, args=[sess, coord])
            eqth.start()
            enq_threads.append(eqth)
            
        return coord
    
    def train_action(self, sess):
        model = self._model
        imdb = self._imdb
        mc = self._mc
        
        # read batch input
        lidar_per_batch, \
        lidar_mask_per_batch, \
        label_per_batch, \
        weight_per_batch = imdb.read_batch()
    
        feed_dict = {
            model.ph_keep_prob: mc.KEEP_PROB,
            model.ph_lidar_input: lidar_per_batch,
            model.ph_lidar_mask: lidar_mask_per_batch,
            model.ph_label: label_per_batch,
            model.ph_loss_weight: weight_per_batch,
        }
        sess.run(model.enqueue_op, feed_dict=feed_dict)
        

    def step_loop(self, start_step, max_step=FLAGS.max_steps):
    
        mc = self._mc
        model = self._model
        
        sess = self._sess
        saver = self._saver
        summary_op = self._summary_op
        summary_writer = self._summary_writer
        
        run_options = tf.RunOptions(timeout_in_ms=60000)
    
        for step in range(start_step, max_step):
            start_time = time.time()
        
            if step % FLAGS.summary_step == 0 or step == FLAGS.max_steps - 1:
                op_list = [
                    model.lidar_input,
                    model.lidar_mask,
                    model.label,
                    model.train_op,
                    model.loss,
                    model.pred_cls,
                    summary_op
                ]
            
                lidar_per_batch, lidar_mask_per_batch, label_per_batch, \
                _, loss_value, pred_cls, summary_str \
                    = sess.run(op_list, options=run_options)
            
                label_image = visualize_seg(label_per_batch[:6, :, :], mc)
                pred_image = visualize_seg(pred_cls[:6, :, :], mc)
            
                # Run evaluation on the batch
                ious, _, _, _ = evaluate_iou(
                    label_per_batch, pred_cls * np.squeeze(lidar_mask_per_batch),
                    mc.NUM_CLASS)
            
                feed_dict = {}
                # Assume that class-0 is the background class
                for i in range(1, mc.NUM_CLASS):
                    feed_dict[model.iou_summary_placeholders[i]] = ious[i]
            
                iou_summary_list = sess.run(model.iou_summary_ops[1:], feed_dict)
            
                # Run visualization
                viz_op_list = [model.show_label, model.show_depth_img, model.show_pred]
                viz_summary_list = sess.run(
                    viz_op_list,
                    feed_dict={
                        model.depth_image_to_show: lidar_per_batch[:6, :, :, [4]],
                        model.label_to_show: label_image,
                        model.pred_image_to_show: pred_image,
                    }
                )
            
                # Add summaries
                summary_writer.add_summary(summary_str, step)
            
                for sum_str in iou_summary_list:
                    summary_writer.add_summary(sum_str, step)
            
                for viz_sum in viz_summary_list:
                    summary_writer.add_summary(viz_sum, step)
            
                # force tensorflow to synchronise summaries
                summary_writer.flush()
        
            else:
                # ?
                _, loss_value = sess.run([model.train_op, model.loss], options=run_options)
        
            duration = time.time() - start_time
        
            assert not np.isnan(loss_value), \
                'Model diverged. Total loss: {}, conf_loss: {}, bbox_loss: {}, ' \
                'class_loss: {}'.format(loss_value, conf_loss, bbox_loss,
                                        class_loss)
        
            if step % 10 == 0:
                num_images_per_step = mc.BATCH_SIZE
                images_per_sec = num_images_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f images/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    images_per_sec, sec_per_batch))
                sys.stdout.flush()
        
            # Save the model checkpoint periodically.
            if step % FLAGS.checkpoint_step == 0 or step == FLAGS.max_steps - 1:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)




def main(argv=None):  # pylint: disable=unused-argument
    
    # if tf.gfile.Exists(FLAGS.train_dir):
    #     tf.gfile.DeleteRecursively(FLAGS.train_dir)
    
    tf.gfile.MakeDirs(FLAGS.train_dir)
    
    
    tr = train()
    tr.train_loop()
    

if __name__ == '__main__':
    tf.app.run()
