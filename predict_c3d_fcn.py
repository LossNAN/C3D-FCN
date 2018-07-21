import os
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_test_data
import c3d_model
import fcn_model
import math
import numpy as np
import sys
from utils import placeholder_inputs, _variable_with_weight_decay, fcn_model_loss, tower_acc
# Basic model parameters as external flags.
flags = tf.app.flags
flags.DEFINE_integer('batch_size',1 , 'Batch size.')
flags.DEFINE_boolean('output_to_file', True , 'print outputs to files or to the screen')
FLAGS = flags.FLAGS
pre_model_save_dir = './models/fcn54'

def run_testing():
    with tf.Graph().as_default():
        global_step = tf.get_variable(
                        'global_step',
                        [],
                        initializer=tf.constant_initializer(0),
                        trainable=False
                        )
        with tf.variable_scope('var_name') as var_scope:
            weights = {
                  'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
                  'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
                  'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
                  'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
                  'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
                  'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
                  'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
                  'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
                  'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.0005),
                  'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
                  'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.0005)
                  }
            biases = {
                  'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
                  'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
                  'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
                  'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
                  'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
                  'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
                  'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
                  'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
                  'bd1': _variable_with_weight_decay('bd1', [4096], 0.000),
                  'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
                  'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.000),
                  }
            fcn_weights = {
                  'wconv5': _variable_with_weight_decay('conv5', [3, 7, 7, 512, 512], 0.0005),
                  'wdown5': _variable_with_weight_decay('down5', [1, 7, 7, 512, 512], 0.0005),
                  'wup5': _variable_with_weight_decay('up5', [3, 1, 1, 4096, 512], 0.0005),
                  'wconv4': _variable_with_weight_decay('conv4', [3, 7, 7, 512, 512], 0.0005),
                  'wdown4': _variable_with_weight_decay('down4', [1, 7, 7, 512, 512], 0.0005),
                  'wup4': _variable_with_weight_decay('up4', [3, 1, 1, 4096, 4096], 0.0005),
                  'wup3': _variable_with_weight_decay('up3', [3, 1, 1, fcn_model.NUM_CLASSES, 4096], 0.0005),
                  #'wconv3': _variable_with_weight_decay('dconv3', [3, 7, 7, 256,256 ], 0.0005),
                  #'wdown3': _variable_with_weight_decay('down3', [1, 7, 7, 256,4096 ], 0.0005),
                  #'wup3': _variable_with_weight_decay('up3', [2, 1, 1, fcn_model.NUM_CLASSES, 4096], 0.0005),
                  }
            fcn_biases = {
                  'bconv5': _variable_with_weight_decay('bconv5', [512], 0.000),
                  'bdown5': _variable_with_weight_decay('bdown5', [512], 0.000),
                  'bup5': _variable_with_weight_decay('bup5', [4096], 0.000),
                  'bconv4': _variable_with_weight_decay('bconv4', [512], 0.000),
                  'bdown4': _variable_with_weight_decay('bdown4', [512], 0.000),
                  'bup4': _variable_with_weight_decay('bup4', [4096], 0.000),
                  'bup3': _variable_with_weight_decay('bup3', [fcn_model.NUM_CLASSES], 0.000),
                  #'bconv3': _variable_with_weight_decay('bconv3', [256], 0.000),
                  #'bdown3': _variable_with_weight_decay('bdown3', [4096], 0.000),
                  #'bup3': _variable_with_weight_decay('bup3', [fcn_model.NUM_CLASSES], 0.000),
                  }
        with tf.name_scope('inputs'):
            images_placeholder, labels_placeholder, keep_pro = placeholder_inputs( FLAGS.batch_size )

        feature_map = c3d_model.inference_c3d(
                            images_placeholder,
                            keep_pro,
                            FLAGS.batch_size,
                            weights,
                            biases
                            )

        logit=fcn_model.inference_fcn5(
                            feature_map,
                            keep_pro,
                            FLAGS.batch_size,
                            fcn_weights,
                            fcn_biases
                            )
        loss = fcn_model_loss(
                            logit,
                            labels_placeholder,
                            FLAGS.batch_size
                            )

        accuracy = tower_acc(logit, labels_placeholder, FLAGS.batch_size)
        predictions = tf.nn.top_k(logit, 1)
        # Create a saver for writing training checkpoints.
        new_saver = tf.train.Saver(weights.values() + biases.values()+ fcn_weights.values() + fcn_biases.values())
        init = tf.global_variables_initializer()
        # Create a session for running Ops on the Graph.
        sess = tf.Session(
                        config=tf.ConfigProto(allow_soft_placement=True)
                        )
        sess.run(init)
    ckpt = tf.train.get_checkpoint_state(pre_model_save_dir)  
    if ckpt and ckpt.model_checkpoint_path:  
        print "loading checkpoint,waiting......"
        new_saver.restore(sess, ckpt.model_checkpoint_path)
        print "load complete!"
     
    if FLAGS.output_to_file:
        # all output will be stored in 'output.txt'
        print('outputs will be stored in test.txt')
        sys.stdout = open( 'test.txt', 'a', 1)
    predict_list = []
    label_list = []
    for i in xrange(3358):
        start_time = time.time()
        test_images, test_labels, _, _, _, _ = input_test_data.read_clip_and_label(
            filename='annotation/test.list',
            batch_size=1,
            start_pos=-1,
            num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
            crop_size=c3d_model.CROP_SIZE,
            video_list=[]
        )

        acc, predict = sess.run([accuracy, predictions], feed_dict={
            images_placeholder: test_images,
            labels_placeholder: test_labels,
            keep_pro: 1
        })
        print ('acc: {}'.format(acc))
        print ('predict: {}'.format(np.reshape(predict[1], [32])))
        predict_list.append(np.reshape(predict[1], [32]))
        print ('labels: {}'.format(np.reshape(test_labels, [32])))
        label_list.append(np.reshape(test_labels, [32]))
    np.save('./test/predict', predict_list)
    np.save('./test/label', label_list)
def main(_):
    run_testing()

if __name__ == '__main__':
    tf.app.run()
