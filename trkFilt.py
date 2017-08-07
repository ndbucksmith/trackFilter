from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import argparse
import os.path
import sys
import time
import pdb

import pprint
import pickle
import numpy
import pprint


from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed



import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)
# Basic model parameters as external flags.
FLAGS = None

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
num_epochs = 2

class DataSet(object):

  def __init__(self,
               pv6Inputs,
               pv6trues,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)

    self._pv6inputs = pv6Inputs
    self._pv6trues = pv6trues
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def pv6inputs(self):
    return self._pv6inputs

  @property
  def pv6trues(self):
    return self._pv6trues

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=False):
    """Return the next `batch_size` examples from this data set."""
    #pdb.set_trace()
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._pv6inputs = self.pv6inputs[perm0[0]] 
      self._pv6trues = self.pv6trues[perm0[0]]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      
      #print('self._epochs_completed:' + str(self._epochs_completed))
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      pv6inputs_rest_part = self._pv6inputs[start:self._num_examples]
      pv6trues_rest_part = self._pv6trues[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._pv6inputs = self.pv6inputs[perm[0]]
        self._pv6trues = self.pv6trues[perm[0]]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      pv6inputs_new_part = self._pv6inputs[start:end]
      pv6trues_new_part = self._pv6trues[start:end]
      #avoid numpy concat err
      if pv6inputs_rest_part ==[]:
        xIns = pv6inputs_new_part
      else:
        xIns =  numpy.concatenate((pv6inputs_rest_part, pv6inputs_new_part), axis=0)     
        
      if pv6trues_rest_part ==[]:
         yTrus = pv6trues_new_part
      else:       
        yTrus =  numpy.concatenate((pv6trues_rest_part, pv6trues_new_part), axis=0) 

      return  xIns, yTrus
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      #print('start:' + str(start) + 'end:' + str(end))
      return self._pv6inputs[start:end], self._pv6trues[start:end]

def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None,
                   scale = True):
  numrecordspv6Inputs =0
  numrecordspv6trues = 0
  trainSetIns = []
  trainSetTrues = []
  for ti in range(0, 1000):
    fninp = 'tracks/Inp-' + str(ti) + '.trk'
    fntru = 'tracks/Tru-' + str(ti) + '.trk'
    fncsv = 'tracks/Trk-' + str(ti) + '.csv'
    filinp=open(fninp, 'r')
    filtru=open(fntru, 'r')
    #filcsv = open(fncsv, 'wb')
    x = []
    y=[]
    #
    x = pickle.load(filinp)
    y = pickle.load(filtru)
    #print(x[0][0],y[0][0])
    for pp in range(0,(len(x)-1)):
      trainSetIns.append(x[pp])
      trainSetTrues.append(y[pp])
      rowStr =""
      for qq in range(6,9):
        rowStr = rowStr + str(x[pp][qq]) + ','
      for rr in range(0,3):
        rowStr = rowStr + str(y[pp][rr])  + ','  
      rowStr= rowStr+ '\r\n'
      #filcsv.write(rowStr)
    filinp.close()
    #filcsv.close
    filtru.close()
  #pdb.set_trace()
  numrecordspv6Inputs =len(trainSetIns)
  print('size inputs:' + str(numrecordspv6Inputs))
  numrecordspv6trues = len(trainSetTrues)  
  #print('size truess:' + str(numrecordspv6trues)
  assert numrecordspv6trues == numrecordspv6Inputs
  validation_size = int( 0.2 * len(trainSetIns))
  #print('valdiationsize:' +str(validation_size))
  #now scale to -1 to 1, magic numbers collected in track gen.py
    #492 756157.522271 810653.565324 104353.717046
  #e.g 491 803464.7 787912.7 98608.3
  #now 479 745493.892057 791520.705135 100903.802122
  #pdb.set_trace()
  if scale:   #turn scaling on and off
    for bb in range(0,numrecordspv6Inputs):
      trainSetIns[bb][0] = 0.5 + (trainSetIns[bb][0] / (95000*2))
      trainSetIns[bb][1] = 0.5 + (trainSetIns[bb][1] / (95000*2))
      trainSetIns[bb][2] = trainSetIns[bb][2] / 110000
      trainSetIns[bb][3] = 0.5 + (trainSetIns[bb][3] / (2000*2))
      trainSetIns[bb][4] = 0.5 + (trainSetIns[bb][4] / (2000*2))
      trainSetIns[bb][5] = 0.5 + (trainSetIns[bb][5] / (2000*2))
      trainSetIns[bb][6] = 0.5 + (trainSetIns[bb][6] / (90000*2))
      trainSetIns[bb][7] = 0.5 + (trainSetIns[bb][7] / (90000*2))
      trainSetIns[bb][8] = trainSetIns[bb][8] / 110000
      trainSetIns[bb][9] = 0.5 + (trainSetIns[bb][9] / (2000*2))
      trainSetIns[bb][10] = 0.5 + (trainSetIns[bb][10] / (2000*2))
      trainSetIns[bb][11] = 0.5 + (trainSetIns[bb][11] / (2000*2))
    
      trainSetTrues[bb][0] = 0.5 + (trainSetTrues[bb][0] / (95000*2))
      trainSetTrues[bb][1] = 0.5 + (trainSetTrues[bb][1] / (95000*2))
      trainSetTrues[bb][2] = trainSetTrues[bb][2] / 110000
      trainSetTrues[bb][3] = 0.5 + (trainSetTrues[bb][3] / (2000*2))
      trainSetTrues[bb][4] = 0.5 + (trainSetTrues[bb][4] / (2000*2))
      trainSetTrues[bb][5] = 0.5 + (trainSetTrues[bb][5] / (2000*2))
    
  validation_pv6inputs = trainSetIns[:validation_size]
  validation_pv6trues = trainSetTrues[:validation_size]
  trainSetIns = trainSetIns[validation_size:]
  trainSetTrues = trainSetTrues[validation_size:]
  #print(trainSetIns[0][0], trainSetTrues[0][0])
  train = DataSet(
      trainSetIns, trainSetTrues, dtype=dtype, reshape=reshape, seed=seed)
  train._num_examples= numrecordspv6Inputs - validation_size
  validation = DataSet(
      validation_pv6inputs,
      validation_pv6trues,
      dtype=dtype,
      reshape=reshape,
      seed=seed)
  validation._num_examples=  validation_size
  test = DataSet(
      validation_pv6inputs, validation_pv6trues, dtype=dtype, reshape=reshape, seed=seed)
  test._num_examples=  validation_size
  return base.Datasets(train=train, validation=validation, test=test)

def datasetString(Values, rounder):
  strg = ""
  for kk in range(len(Values)):
    strg = strg +  str(round(Values[kk],rounder)) + ", "
  strg = strg[:-2]  
  return strg

def codeModelTest(scale):  #make a simple postion velocity projector in python and test sum square loss
  nnTrackFilter = read_data_sets('tracks', scale = scale)
  maxSumSq = 0
  if scale:
    formatStr = "{0:.5f} {0:.5f} {0:.5f} {0:.5f} {0:.5f} {0:.5f}"
    rounder = 5
  else:
    formatStr = "{0:.0f} {0:.0f} {0:.0f} {0:.0f} {0:.0f} {0:.0f}"
    rounder = 0
  for bb in range(0,30):
    batch = nnTrackFilter.train.next_batch(100)
  #batch = nnTrackFilter.train.next_batch(100)
    for ff in range(0,100):
        #simple prediction for postion and velocity two seconds ahead
      x_ = (batch[0][ff][6] *2) -batch[0][ff][0] + batch[0][ff][9] 
      y_ = (batch[0][ff][7] *2) -batch[0][ff][1] + batch[0][ff][10]               
      z_ = (batch[0][ff][8] *2) -batch[0][ff][2] + batch[0][ff][11]     
      vx_ = (batch[0][ff][9] *2) - batch[0][ff][3] 
      vy_ = (batch[0][ff][10] *2) - batch[0][ff][4] 
      vz_ = (batch[0][ff][11] *2) - batch[0][ff][5]    
      if ff % 40 == 15:  
        print( 'predictions:' + datasetString([x_,y_,z_,vx_,vy_, vz_], rounder))
        print('trues:       ' + datasetString(batch[1][ff][0:6], rounder))
        print('0:5 in       '  + datasetString(batch[0][ff][0:6],rounder))
        print("6;11 in      " + datasetString(batch[0][ff][6:12],rounder))
        loss = pow((batch[1][ff][0] - x_),2) + pow((batch[1][ff][1]- y_),2) + pow((batch[1][ff][2] -z_),2) + pow((batch[1][ff][3]-vx_),2) + pow((batch[1][ff][4]-vy_),2) + pow((batch[1][ff][5]-vz_),2)
        loss = loss/6
        print('mean-Sq:    {0:.3f}'.format(loss))
        if maxSumSq < loss:
          maxSumSq = loss       
  print('MaxMeanSq: {0:.3f}'.format(maxSumSq))
  print("codemodel complete")

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)



def meanSquare(batch1, trues, idx,scale):
  if scale:
    rounder = 5
  else:
    rounder = 0
  sumSquaresList = [0,0,0,0,0,0]
  meanSquareList = [0,0,0,0,0,0]
  maxMean = 0
  maxMeanIdx = -1
  sumSquares = 0
  meanSumSquares = 0
  assert len(batch1) == 100
  assert len(batch1) == len(trues)
  if idx == -1:
    for aa in range(len(batch1)):
      for bb in range(0,6):
        sumSquaresList[bb] = pow(trues[aa][bb] - batch1[aa][bb],2) + sumSquaresList[bb] 
        sumSquares = pow(trues[aa][bb] - batch1[aa][bb],2) + sumSquares
     #print(str(aa) + '::'   + str(sumSquares)
        #print(sumSquaresList[bb],sumSquares)
        #pdb.set_trace()
  for cc in range(0,6):
    meanSquareList[cc] = sumSquaresList[cc]/len(batch1)  
    if meanSquareList[cc] > maxMean:
      maxMean = meanSquareList[cc]
      maxMeanIdx = cc
  meanSumSquares = sumSquares/(6*len(batch1))
  print('meansquares ' +  datasetString(meanSquareList,rounder))
  #print('max mean ' +  str(round(meanSquareList[maxMeanIdx],rounder) )  + " @idx: " + str(maxMeanIdx))
  print("mean squares  " + str(round(meanSumSquares,rounder)))
  #print("sum squares  " + str(round(sumSquares,rounder)))
    
def main(scale):
  if scale:
    rounder = 5
  else:
    rounder = 0
  nnTrackFilter = read_data_sets('tracks', scale = scale)
  sess = tf.Session()
  global_step = tf.Variable(0, trainable=False)
  measureds_ph = tf.placeholder(tf.float32, shape=(None,12))
  trues_ph = tf.placeholder(tf.float32, shape=(None,6))
  #pdb.set_trace()
  with tf.name_scope('weights'):
    W_h1 = tf.Variable(tf.truncated_normal([12,24]))    #tf.truncated_normal
    variable_summaries(W_h1)
    W_h2 = tf.Variable(tf.truncated_normal([24,20]))
    variable_summaries(W_h2)
    W_h3 = tf.Variable(tf.truncated_normal([20,20]))
    variable_summaries(W_h3)
    W_h4 = tf.Variable(tf.truncated_normal([20,20]))
    variable_summaries(W_h4)
    W_h5 = tf.Variable(tf.truncated_normal([20,20]))
    variable_summaries(W_h5)
    W = tf.Variable(tf.truncated_normal([20,6]))   
    variable_summaries(W)
  with tf.name_scope('biases'):
    b_h1 =    tf.Variable(tf.truncated_normal([24]))   
    variable_summaries(b_h1)  
    b_h2 =    tf.Variable(tf.truncated_normal([20]))  
    variable_summaries(b_h2)      
    b_h3 =    tf.Variable(tf.truncated_normal([20]))  
    variable_summaries(b_h3)  
    b_h4 =    tf.Variable(tf.truncated_normal([20]))  
    variable_summaries(b_h4)  
    b_h5 =    tf.Variable(tf.truncated_normal([20]))  
    variable_summaries(b_h5)  
    b = tf.Variable(tf.truncated_normal([6]))   
    variable_summaries(b)  
  #merged = tf.summary.merge_all()
  #train_writer = tf.summary.FileWriter('tboard', sess.graph)
  sess.run(tf.global_variables_initializer())  
  hidden1 = tf.nn.sigmoid(tf.matmul(measureds_ph, W_h1) + b_h1)
  hidden2 = tf.nn.relu(tf.matmul(hidden1, W_h2) + b_h2)
  hidden3 = tf.nn.sigmoid(tf.matmul(hidden2, W_h3) + b_h3)
  hidden4 = tf.nn.relu(tf.matmul(hidden3, W_h4) + b_h4)
  hidden5 = tf.nn.sigmoid(tf.matmul(hidden4, W_h5) + b_h5)
  y_ = tf.matmul(hidden5,W) + b      
  #tf_loss = tf.losses.mean_squared_error(y_, trues_ph, weights=1.0, scope=None,)
  #tf_loss = tf.losses.mean_squared_error(y_, trues_ph, weights=1.0, scope=None,)
  tf_loss = tf.reduce_mean(tf.square(y_- trues_ph)) # sum of the squares
  #tf_loss = tf.reduce_sum(tf.losses.log_loss(y_, trues_ph))
  starter_learning_rate = 0.000002
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 0.96, staircase=True)
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_loss)    
  tf.logging.set_verbosity(tf.logging.DEBUG) 
  
  #pdb.set_trace()s
  kilobatchMinLoss  =  1000000
  kilobatchMaxLoss = 0
  maxLoss = 0
  maxLossIteration = -1
  minLoss = 1000000000000
  minlossIteration = -1
  breakAtIteration = -1
  tferrcount = 0
  for _ in range(15000000):   #try:
    batch = nnTrackFilter.train.next_batch(100)
 #   except:
    #  tferrcount = tferrcount + 1
      #print("tf err @ iteration " + Str(_))
      #pdb.set_trace()
   #   batch = nnTrackFilter.train.next_batch(100)
      #break
    
    tstep , rmsVal = sess.run([train_step, tf_loss],feed_dict={measureds_ph : batch[0], trues_ph: batch[1]})
    if (_ > 1000000) and rmsVal > (200000):
      y, trus= sess.run([ y_, trues_ph],feed_dict={measureds_ph : batch[0], trues_ph: batch[1]})
      meanSquare(trus, y, -1,scale)
      print(batch[0])
      print(y)
      print(trus)
      break
    if rmsVal < kilobatchMinLoss:
      kilobatchMinLoss = rmsVal
      kilobatchMinIt = _
    if rmsVal > kilobatchMaxLoss:
      kilobatchMaxLoss = rmsVal
      kilobatchMaxIt = _
    #print('iteration: ' + str(_) + '  loss:   {0:.5f}'.format(rmsVal))
    if _ % 100000 == 1  and _ > 10:                  # print(tf.get_default_session().run(W))
      print("kilobatch Min:{0:.5f}".format(kilobatchMinLoss) + " @ " +str(kilobatchMinIt))
      print("kilobatch Max:{0:.5f}".format(kilobatchMaxLoss) + " @ " +str(kilobatchMaxIt))
      kilobatchMaxLoss = rmsVal
      kilobatchMinLoss = rmsVal
      kilobatchMaxIt = _
      kilobatchMinIt = _
      print('min loss: {0:.5f}'.format(minLoss) + '  @ iteration ' + str(minlossIteration))
      #train_step.run( feed_dict={measureds_ph : batch[0], trues_ph: batch[1]})
      #pdb.set_trace()
      if _ == breakAtIteration:
        pdb.set_trace()
        breakAtIteration = 201
        W1, W2, W0 = sess.run([W_h1, W_h2, W])
      #print('batch[1][1]:   ' + datasetString(batch[1][1],rounder)) , merged  , summary
      y, trus= sess.run([ y_, trues_ph],feed_dict={measureds_ph : batch[0], trues_ph: batch[1]})
      #try:
        #train_writer.add_summary(summary, _)
      #except:
         #break     
      #print('iteration: ' + str(_) + '  loss:   {0:.5f}'.format(rmsVal))

      #pisumSq = pow((y[0][0] - trus[0][0]),2) + pow((y[0][1] - trus[0][1]),2) + pow((y[0][2] - trus[0][2]),2) + pow((y[0][3] - trus[0][3]),2) +pow ((y[0][4] - trus[0][4]),2) + pow((y[0][5] - trus[0][5]),2)
    if rmsVal < minLoss:
        minLoss = rmsVal
        minlossIteration = _
    if rmsVal > maxLoss:
        maxLoss = rmsVal
        maxLossIteration = _
      #print('sum Square:',  str(pisumSq))
      #pims = (pisumSq/6)
      #print('pi-mean-Sq: {0:.3f}'.format(pims))
      #print(x)
     # print(y)
    #print(x)
    #with tf.Session() as sesh:
    #  print(sesh.run(tf_loss))
  print('final loss: {0:.5f}'.format(rmsVal) + '  @ iteration ' + str(_)) 
  print('max loss: {0:.5f}'.format(maxLoss) + '  @ iteration ' + str(maxLossIteration)) 
  print('min loss: {0:.5f}'.format(minLoss) + '  @ iteration ' + str(minlossIteration))
  print("tensorflow complete")
  accuracy = tf.reduce_mean(tf.square(y_- trues_ph))
  print(accuracy.eval(feed_dict={measureds_ph : nnTrackFilter.validation[0], trues_ph:nnTrackFilter.validation[1]}))
  
  
if __name__ == "__main__":
   for dd in range(1):
    main(True)
  #codeModelTest(False)

"""
20 million batches

iteration: 19999982  loss:   0.23028
iteration: 19999983  loss:   0.26805
iteration: 19999984  loss:   0.36295
iteration: 19999985  loss:   0.16618
iteration: 19999986  loss:   0.05845
iteration: 19999987  loss:   0.10832
iteration: 19999988  loss:   0.26115
iteration: 19999989  loss:   0.27497
iteration: 19999990  loss:   0.36136
iteration: 19999991  loss:   0.26054
iteration: 19999992  loss:   0.33312
iteration: 19999993  loss:   0.65266
iteration: 19999994  loss:   1.45064
iteration: 19999995  loss:   2.27920
iteration: 19999996  loss:   0.21031
iteration: 19999997  loss:   0.22003
iteration: 19999998  loss:   0.10072
iteration: 19999999  loss:   0.15083
final loss: 0.15083  @ iteration 19999999
max loss: 211251344.00000  @ iteration 1774
min loss: 0.00599  @ iteration 8134017 

random batch error with high loss

teration: 19995317  loss:   0.26882
iteration: 19995318  loss:   0.13679
iteration: 19995319  loss:   0.04436
iteration: 19995320  loss:   210895040.00000
iteration: 19995321  loss:   0.22946


iteration: 19997092  loss:   0.25688
iteration: 19997093  loss:   0.09132
iteration: 19997094  loss:   210895008.00000
iteration: 19997095  loss:   0.88555
iteration: 19997096  loss:   0.22343

  File "trkFilt.py", line 407, in <module>
    main(True)
  File "trkFilt.py", line 358, in main
    batch = nnTrackFilter.train.next_batch(100)
  File "trkFilt.py", line 109, in next_batch


    return numpy.concatenate((pv6inputs_rest_part, pv6inputs_new_part), axis=0) , numpy.concatenate((pv6trues_rest_part, pv6trues_new_part), axis=0)
ValueError: all the input arrays must have same number of dimensions

iteration: 19999904  loss:   0.20500
iteration: 19999905  loss:   0.31319
iteration: 19999906  loss:   0.20337
iteration: 19999907  loss:   0.15201
iteration: 19999908  loss:   0.22272
iteration: 19999909  loss:   0.17528
iteration: 19999910  loss:   0.19620
iteration: 19999911  loss:   0.17449
iteration: 19999912  loss:   0.11097
iteration: 19999913  loss:   0.18484
iteration: 19999914  loss:   0.16977
iteration: 19999915  loss:   0.19189
iteration: 19999916  loss:   0.16953
iteration: 19999917  loss:   0.20327
iteration: 19999918  loss:   0.21833
iteration: 19999919  loss:   0.19008
iteration: 19999920  loss:   0.20630
iteration: 19999921  loss:   0.09273
iteration: 19999922  loss:   0.20800
iteration: 19999923  loss:   0.17844
iteration: 19999924  loss:   0.14656
iteration: 19999925  loss:   0.09878
iteration: 19999926  loss:   0.18845
iteration: 19999927  loss:   0.16315
iteration: 19999928  loss:   0.09006
iteration: 19999929  loss:   0.08358
iteration: 19999930  loss:   0.20110
iteration: 19999931  loss:   0.20126
iteration: 19999932  loss:   0.14910
iteration: 19999933  loss:   0.11258
iteration: 19999934  loss:   0.18462
iteration: 19999935  loss:   0.14709
iteration: 19999936  loss:   0.24407
iteration: 19999937  loss:   0.13956
iteration: 19999938  loss:   0.18188
iteration: 19999939  loss:   0.17433
iteration: 19999940  loss:   0.20541
iteration: 19999941  loss:   0.20837
iteration: 19999942  loss:   0.21422
iteration: 19999943  loss:   0.33184
iteration: 19999944  loss:   0.23295
iteration: 19999945  loss:   0.20810
iteration: 19999946  loss:   0.15342
iteration: 19999947  loss:   0.21541
iteration: 19999948  loss:   0.12583
iteration: 19999949  loss:   0.05958
iteration: 19999950  loss:   0.26927
iteration: 19999951  loss:   0.44983
iteration: 19999952  loss:   1.10761
iteration: 19999953  loss:   2.54100
iteration: 19999954  loss:   1.88901
iteration: 19999955  loss:   0.09317
iteration: 19999956  loss:   0.12837
iteration: 19999957  loss:   0.10149
iteration: 19999958  loss:   0.18510
iteration: 19999959  loss:   0.17733
iteration: 19999960  loss:   0.12423
iteration: 19999961  loss:   0.01901
iteration: 19999962  loss:   0.12834
iteration: 19999963  loss:   0.16585
iteration: 19999964  loss:   0.09141
iteration: 19999965  loss:   0.17648
iteration: 19999966  loss:   0.08462
iteration: 19999967  loss:   0.16205
iteration: 19999968  loss:   0.21487
iteration: 19999969  loss:   0.18580
iteration: 19999970  loss:   0.20264
iteration: 19999971  loss:   0.16818
iteration: 19999972  loss:   0.23614
iteration: 19999973  loss:   0.11995
iteration: 19999974  loss:   0.17997
iteration: 19999975  loss:   0.20676
iteration: 19999976  loss:   0.17369
iteration: 19999977  loss:   0.03121
iteration: 19999978  loss:   0.06648
iteration: 19999979  loss:   0.19389
iteration: 19999980  loss:   0.14734
iteration: 19999981  loss:   0.22006
iteration: 19999982  loss:   0.14954
iteration: 19999983  loss:   0.21362
iteration: 19999984  loss:   0.24585
iteration: 19999985  loss:   0.54927
iteration: 19999986  loss:   1.31744
iteration: 19999987  loss:   0.80702
iteration: 19999988  loss:   0.14248
iteration: 19999989  loss:   0.18895
iteration: 19999990  loss:   0.19018
iteration: 19999991  loss:   0.22531
iteration: 19999992  loss:   0.19311
iteration: 19999993  loss:   0.33080
iteration: 19999994  loss:   0.27917
iteration: 19999995  loss:   0.10400
iteration: 19999996  loss:   0.17440
iteration: 19999997  loss:   0.18982
iteration: 19999998  loss:   0.16505
iteration: 19999999  loss:   0.33884
final loss: 0.33884  @ iteration 19999999
max loss: 211248496.00000  @ iteration 1774
min loss: 0.00278  @ iteration 15279108



"""


