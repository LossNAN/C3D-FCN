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
flags.DEFINE_integer('max_steps',40000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size',16 , 'Batch size.')
FLAGS = flags.FLAGS
MOVING_AVERAGE_DECAY = 0.9999
model_save_dir = './models/fcn54_just20'

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
        varlist1 = list( set(fcn_weights.values() + fcn_biases.values()) )
        varlist2 = [weights['wc5a'], biases['bc5a'], weights['wc5b'], biases['bc5b']]
        learning_rate_stable = tf.train.exponential_decay(1e-4, global_step, decay_steps=FLAGS.max_steps/FLAGS.batch_size, decay_rate=0.98, staircase=True)
        learning_rate_finetuning = tf.train.exponential_decay(1e-5, global_step, decay_steps=FLAGS.max_steps/FLAGS.batch_size, decay_rate=0.98,  staircase=True)
        tf.summary.scalar('learning_rate_stable', learning_rate_stable)
        tf.summary.scalar('learning_rate_finetuning', learning_rate_finetuning)
        tf.summary.histogram('wconv5', fcn_weights['wconv5'])
        tf.summary.histogram('wup5', fcn_weights['wup5'])
        tf.summary.histogram('wup4', fcn_weights['wup4'])
        tf.summary.histogram('wup3', fcn_weights['wup3'])
        opt_stable = tf.train.AdamOptimizer(learning_rate_stable)
        opt_finetuning = tf.train.AdamOptimizer(learning_rate_finetuning)

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
        grads_fcn = opt_stable.compute_gradients(loss, varlist1)
        grads_fine = opt_finetuning.compute_gradients(loss, varlist2)

        accuracy = tower_acc(logit, labels_placeholder, FLAGS.batch_size)
        tf.summary.scalar('accuracy', accuracy)
        #grads_fcn = average_gradients(tower_grads1)
        #grads_fine = average_gradients(tower_grads2)

        apply_gradient_stable = opt_stable.apply_gradients(grads_fcn, global_step=global_step)    
        apply_gradient_finetuning = opt_finetuning.apply_gradients(grads_fine, global_step=global_step)

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        train_op = tf.group(apply_gradient_stable,apply_gradient_finetuning, variables_averages_op)
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
    train_writer = tf.summary.FileWriter('./visual_logs/fcn54_just20_visual_logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('./visual_logs/fcn54_just20_visual_logs/test', sess.graph)
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
        
        if (step) %100 == 0 or (step + 1) == FLAGS.max_steps:

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
        if step > 1 and step % 2000 == 0:
            checkpoint_path = os.path.join('./models/fcn54_just20', 'model.ckpt')
            new_saver.save(sess, checkpoint_path, global_step=global_step) 

    print("done")

def main(_):
    run_training()

if __name__ == '__main__':
     tf.app.run()
