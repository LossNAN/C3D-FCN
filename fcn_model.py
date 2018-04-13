
# coding: utf-8

import tensorflow as tf

NUM_CLASSES = 21

def conv3d(name, l_input, w, b,padding=0):
  if padding==1:
    return tf.nn.bias_add(
          tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='VALID'),
          b
          )
  else:
    return tf.nn.bias_add(
          tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
          b
          )

def deconv3d(name,l_input, w, b, output_shape, padding=0):
    # Deconv layer
    if padding == 0:
        deconv = tf.nn.conv3d_transpose(l_input, w, output_shape=output_shape, strides=[1, 2, 1, 1, 1], padding="SAME")
    else:
        deconv = tf.nn.conv3d_transpose(l_input, w, output_shape=output_shape, strides=[1, 2, 1, 1, 1], padding="VALID")
    deconv = tf.nn.bias_add(deconv, b)
    deconv = tf.nn.relu(deconv)
    return deconv


def max_pool(name, l_input, k,m=1):
  return tf.nn.max_pool3d(l_input, ksize=[1, k, 2*m, 2*m, 1], strides=[1, k, 2*m, 2*m, 1], padding='SAME', name=name)

def inference_fcn(list,_dropout,batch_size,_myweights, _mybiases):
    conv3=list[0]#256*L/2*28*28
    conv4=list[1]#512*L/4*14*14
    conv5=list[2]#512*L/8*7*7 
    # Convolution Layer
    conv5 = conv3d('conv5', conv5, _myweights['wconv5'], _mybiases['bconv5'],padding=0)
    conv5 = tf.nn.relu(conv5, 'relu5')
    conv5 = conv3d('conv5', conv5, _myweights['wdown5'], _mybiases['bdown5'],padding=1)
    conv5 = tf.nn.relu(conv5, 'relu5')
    output_shape=conv5.get_shape().as_list()
    output_shape[1] *= 2
    output_shape[4]=_myweights['wup5'].get_shape().as_list()[3]
    conv5=deconv3d('deconv5',conv5,_myweights['wup5'],_mybiases['bup5'],output_shape,padding=0)#512*L/4*1*1
    conv5 = tf.nn.relu(conv5, 'relu5')
    #conv5=deconv3d('deconv5',conv5,_myweights['wup7'],_mybiases['bup7'],output_shape,padding=0)#4096*L*1*1
    #conv5 = tf.nn.dropout(conv5, _dropout)
    print conv5.get_shape()
   
    # Convolution Layer
    #print conv4.get_shape()
    
    pool4 = max_pool('pool4', conv4, k=1,m=1)
    conv4 = conv3d('conv4', pool4, _myweights['wconv4'], _mybiases['bconv4'],padding=0)
    conv4 = tf.nn.relu(conv4, 'relu4')
    conv4 = conv3d('conv4', conv4, _myweights['wdown4'], _mybiases['bdown4'],padding=1)
    conv4 = tf.nn.relu(conv4, 'relu4')#512*L/4*1*1
    print conv4.get_shape()
    output_shape=conv4.get_shape().as_list()
    output_shape[1] *= 2
    output_shape[4]=_myweights['wup4'].get_shape().as_list()[3]
    conv4=deconv3d('deconv5',tf.add_n([conv4,conv5]),_myweights['wup4'],_mybiases['bup4'],output_shape,padding=0)#4096*L/2*1*1
    conv4 = tf.nn.relu(conv4, 'relu4')
    #conv4 = tf.nn.dropout(conv4, _dropout)
    print conv4.get_shape()

    # DEConvolution Layer
    output_shape=conv4.get_shape().as_list()
    output_shape[1] *= 2
    output_shape[4]=_myweights['wup3'].get_shape().as_list()[3]
    conv3=deconv3d('deconv5',conv4,_myweights['wup3'],_mybiases['bup3'],output_shape,padding=0)#K+1*L*1*1
    conv3 = tf.nn.relu(conv3, 'relu3')#4096*L/2*1*1
    conv3 = tf.transpose(conv3, perm=[0,1,4,2,3])
    out = tf.reshape(conv3, [batch_size, conv3.get_shape().as_list()[1],conv3.get_shape().as_list()[2]]) 
    print out.get_shape()
    
    '''
    # Convolution Layer
    #print conv3.get_shape()
    pool3 = max_pool('pool3', conv3, k=1,m=2)
    #print pool3.get_shape()
    conv3 = conv3d('conv3', pool3, _myweights['wconv3'], _mybiases['bconv3'],padding=0)
    conv3 = conv3d('conv3', conv3, _myweights['wdown3'], _mybiases['bdown3'],padding=1)
    conv3 = tf.nn.relu(conv3, 'relu3')#4096*L/2*1*1
    conv3 = tf.nn.l2_normalize(conv3,4)
    output_shape=conv3.get_shape().as_list()
    output_shape[1] *= 2
    output_shape[4]=_myweights['wup3'].get_shape().as_list()[3]
    
    #Output  K+1*L

    conv3=deconv3d('deconv5',tf.add_n([conv3, conv4]),_myweights['wup3'],_mybiases['bup3'],output_shape,padding=0)#K+1*L*1*1
    conv3 = tf.transpose(conv3, perm=[0,1,4,2,3])
    out = tf.reshape(conv3, [batch_size, conv3.get_shape().as_list()[1],conv3.get_shape().as_list()[2]]) 
    #out = tf.nn.l2_normalize(out,2)
    print out.get_shape()
   '''
    return out