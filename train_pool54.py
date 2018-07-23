import os
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_train_data
import input_test_data
import c3d_model
import fcn_model
import numpy as np
from utils import placeholder_inputs, _variable_with_weight_decay, fcn_model_loss, tower_acc
flags = tf.app.flags
flags.DEFINE_float('learning_rate',1e-4, 'Initial learning rate.')
flags.DEFINE_integer('max_steps',2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size',16 , 'Batch size.')
FLAGS = flags.FLAGS
MOVING_AVERAGE_DECAY = 0.9999
model_save_dir = './models/SGD_pool54'

def run_training():
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    use_pretrained_model = True
    model_filename = "./sports1m_finetuning_ucf101.model"
    with tf.Graph().as_default():
        global_step = tf.get_variable(
                        'global_step',
                        [],
                        initializer=tf.constant_initializer(0),
                        trainable=False
                        )
        with tf.variable_scope('var_name') as var_scope:
            weights = {
                  'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.005),
                  'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.005),
                  'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.005),
                  'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.005),
                  'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.005),
                  'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.005),
                  'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.005),
                  'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.005),
                  #'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.005),
                  #'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.005),
                  #'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.005)
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
                  #'bd1': _variable_with_weight_decay('bd1', [4096], 0.000),
                  #'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
                  #'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.000),
                  }
            fcn_weights = {
                  'wconv6': _variable_with_weight_decay('conv6', [1, 4, 4, 512, 512], 0.005),
                  'wconv7': _variable_with_weight_decay('conv7', [1, 7, 7, 512, 512], 0.005),
                  'wup6': _variable_with_weight_decay('up6', [2, 1, 1, 4096, 512], 0.005),
                  'wup7': _variable_with_weight_decay('up7', [2, 1, 1, 4096, 4096], 0.005),
                  'wup8': _variable_with_weight_decay('up8', [2, 1, 1, fcn_model.NUM_CLASSES, 4096], 0.005),
                  }
            fcn_biases = {
                  'bconv6': _variable_with_weight_decay('bconv6', [512], 0.000),
                  'bconv7': _variable_with_weight_decay('bconv7', [512], 0.000),
                  'bup6': _variable_with_weight_decay('bup6', [4096], 0.000),
                  'bup7': _variable_with_weight_decay('bup7', [4096], 0.000),
                  'bup8': _variable_with_weight_decay('bup8', [fcn_model.NUM_CLASSES], 0.000),
                  }
        with tf.name_scope('inputs'):
            images_placeholder, labels_placeholder, keep_pro = placeholder_inputs( FLAGS.batch_size )
        
        varlist1 = list( set(fcn_weights.values() + fcn_biases.values()) )
        varlist2 = list( set(weights.values() + biases.values()) )

        feature_map = c3d_model.inference_c3d(
                            images_placeholder,
                            keep_pro,
                            FLAGS.batch_size,
                            weights,
                            biases
                            )

        logit=fcn_model.inference_pool54(
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
        SGD_cdc = tf.train.GradientDescentOptimizer(1e-4).minimize(loss, var_list = varlist1)
        SGD_c3d = tf.train.GradientDescentOptimizer(1e-5).minimize(loss, var_list = varlist2)
        accuracy = tower_acc(logit, labels_placeholder, FLAGS.batch_size)
        tf.summary.scalar('accuracy', accuracy)

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        train_op = tf.group(SGD_cdc, SGD_c3d, variables_averages_op)
        null_op = tf.no_op()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver(weights.values() + biases.values())
        new_saver = tf.train.Saver(weights.values() + biases.values()+ fcn_weights.values() + fcn_biases.values())
        init = tf.global_variables_initializer()
        # Create a session for running Ops on the Graph.
        sess = tf.Session(
                        config=tf.ConfigProto(allow_soft_placement=True)
                        )
        sess.run(init)
        merged = tf.summary.merge_all()
    if os.path.isfile(model_filename) and use_pretrained_model:
        print 'loading pretrained_model....'
        saver.restore(sess, model_filename)
        print 'complete!'
    # Create summary writter
    train_writer = tf.summary.FileWriter('./visual_logs/SGD_pool54_visual_logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('./visual_logs/SGD_pool54_visual_logs/test', sess.graph)
    video_list = []
    position = -1
    for step in xrange(FLAGS.max_steps+1):
        start_time = time.time()
        train_images, train_labels, _, _, video_list, position = input_train_data.read_clip_and_label(
                              filename='annotation/train.list',
                              batch_size=FLAGS.batch_size,
                              start_pos=position,
                              num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
                              crop_size=c3d_model.CROP_SIZE,
                              video_list=video_list
                              )

        sess.run(train_op, feed_dict={
                          images_placeholder: train_images,
                          labels_placeholder: train_labels,
                          keep_pro: 0.5
                          })
        duration = time.time() - start_time
        print('Batchnum %d: %.3f sec' % (step, duration))

        if (step) %2 == 0 or (step + 1) == FLAGS.max_steps:
            print('Step %d/%d: %.3f sec' % (step, FLAGS.max_steps, duration))
            print('Training Data Eval:')
            summary,loss_train,acc = sess.run(
                            [merged, loss, accuracy],
                            feed_dict={
                                          images_placeholder: train_images,
                                          labels_placeholder: train_labels,
                                          keep_pro: 1
                                })
            print 'loss: %f' % np.mean(loss_train)
            print ("accuracy: " + "{:.5f}".format(acc))
            train_writer.add_summary(summary, step)
        
        if (step) %10 == 0 or (step + 1) == FLAGS.max_steps:

            print('Validation Data Eval:')
            val_images, val_labels, _, _, _, _ = input_train_data.read_clip_and_label(
                            filename='annotation/test.list',
                            batch_size=FLAGS.batch_size,
                            start_pos=-1,
                            num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
                            crop_size=c3d_model.CROP_SIZE,
                            video_list=[]
                            )
            summary,loss_val, acc = sess.run(
                            [merged, loss, accuracy],
                            feed_dict={
                                            images_placeholder: val_images,
                                            labels_placeholder: val_labels,
                                            keep_pro: 1
                                            })
            print 'loss: %f' % np.mean(loss_val)
            print ("accuracy: " + "{:.5f}".format(acc))
            test_writer.add_summary(summary, step)
        # Save the model checkpoint periodically.
        if step > 1 and step % 200 == 0:
            checkpoint_path = os.path.join('./models/SGD_pool54', 'model.ckpt')
            new_saver.save(sess, checkpoint_path, global_step=global_step) 

    print("done")

def main(_):
    run_training()

if __name__ == '__main__':
     tf.app.run()
