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

#fireTime, max positions:560.0, 979698.13, 969310.95, 111461.39
#max velcoties:  1906.4, 1915.78, 1882.07


trkFilScalers = [98000.0,98000.0,56000.0,2000.0,2000.0,2000.0, 300]

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
                   scale = 'lin'):
  numrecordspv6Inputs =0
  numrecordspv6trues = 0
  trainSetIns = []
  trainSetTrues = []
  for ti in range(0, 8000):
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
      trainSetIns.append(x[pp][0:14])
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

  
  if scale == 'lin':   #turn scaling on and off or between lin and log
    for bb in range(0,numrecordspv6Inputs):
      trainSetIns[bb][0:7] = linearScale (trainSetIns[bb][0:7])
      trainSetIns[bb][8:14] =  linearScale (trainSetIns[bb][8:14])
      trainSetTrues[bb][0:6] = linearScale (trainSetTrues[bb][0:6])
  else:
    for bb in range(0,numrecordspv6Inputs): 
      trainSetIns[bb][0:6] =  unipoleLogScale(trainSetIns[bb][0:6])
      trainSetIns[bb][6:12] =  unipoleLogScale(trainSetIns[bb][6:12])
      trainSetTrues[bb][0:6] =  unipoleLogScale(trainSetTrues[bb][0:6])
  
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
  
def unipoleLogScale(inArray):
  outArray = []
  for ee in range(3):
    newVal = math.log(abs(inArray[ee])+1,10)
    if inArray[ee] < 0:
       newVal = -newVal
    newVal = (math.log(trkFilScalers[ee],10) + newVal)/(2 * math.log(trkFilScalers[ee],10))
    outArray.append(newVal) 
  for rr in range(3):
    newVal = (inArray[rr+3] / (trkFilScalers[rr+3]*2)) + 0.5
    outArray.append(newVal) 
  return outArray
  
def invLogScale(inArray):
  outArray2 = []
  for vv in range(3):
    valtoPow = (inArray[vv] * 2 * math.log(trkFilScalers[vv],10)) - math.log(trkFilScalers[vv],10) 
    if valtoPow > 0:
      newVal2 =pow(10, abs(valtoPow)) -1
    else:
      newVal2 = -pow(10, abs(valtoPow)) +1
    outArray2.append(newVal2)
  for hh in range(3):
    newVal2 = (inArray[hh+3] -0.5) * (trkFilScalers[hh+3]*2)
    outArray2.append(newVal2)   
  return  outArray2   
  
def invLinearScale(inArray):
  outArray1 = []
  for oo in range(len(inArray)):
    if oo == 2:
      newVal1 = inArray[oo]  * (trkFilScalers[oo]*2)     
    else:
      newVal1 = (inArray[oo] -0.5) * (trkFilScalers[oo]*2)
    outArray1.append(newVal1) 
  return outArray1
  
def linearScale(inArray):
  outArray1 = []
  for oo in range(len(inArray)):
    if oo == 2:
       newVal1 = inArray[oo] / (trkFilScalers[oo]*2)     
    else:
      newVal1 = (inArray[oo] / (trkFilScalers[oo]*2)) + 0.5
    outArray1.append(newVal1) 
  return outArray1
   
def datasetString(Values, rounder):
  strg = ""
  for kk in range(len(Values)):
    strg = strg +  str(round(Values[kk],rounder)) + ", "
  strg = strg[:-2]  
  return strg

def codeModelTest(scale):  #make a simple postion velocity projector in python and test sum square loss
  nnTrackFilter = read_data_sets('tracks', scale = scale)
  maxSumSq = 0
  if scale == 'lin':
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
  if scale == 'lin':
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
    
def main(scale, modelName):
  restoreVars = True
  if scale == 'lin':
    rounder = 5
  else:
    rounder = 0
  nnTrackFilter = read_data_sets('tracks', scale = scale)
  sess = tf.Session()
  global_step = tf.Variable(0, trainable=False)
  measureds_ph = tf.placeholder(tf.float32, shape=(None,14))
  trues_ph = tf.placeholder(tf.float32, shape=(None,6))
  #pdb.set_trace()
  with tf.name_scope('weights'):
    W_h1 = tf.Variable(tf.truncated_normal([14,24]))    #tf.truncated_normal
    variable_summaries(W_h1)
    W_h2 = tf.Variable(tf.truncated_normal([24,40]))
    variable_summaries(W_h2)
    W_h3 = tf.Variable(tf.truncated_normal([40,40]))
    variable_summaries(W_h3)
    W_h4 = tf.Variable(tf.truncated_normal([40,40]))
    variable_summaries(W_h4)
    W_h5 = tf.Variable(tf.truncated_normal([40,40]))
    variable_summaries(W_h5)
    W_h6 = tf.Variable(tf.truncated_normal([40,40]))
    variable_summaries(W_h6)
    W_h7 = tf.Variable(tf.truncated_normal([40,40]))
    variable_summaries(W_h7)
    W_h8 = tf.Variable(tf.truncated_normal([40,40]))
    variable_summaries(W_h8)
    W = tf.Variable(tf.truncated_normal([40,6]))   
    variable_summaries(W)
  with tf.name_scope('biases'):
    b_h1 =    tf.Variable(tf.truncated_normal([24]))   
    variable_summaries(b_h1)  
    b_h2 =    tf.Variable(tf.truncated_normal([40]))  
    variable_summaries(b_h2)      
    b_h3 =    tf.Variable(tf.truncated_normal([40]))  
    variable_summaries(b_h3)  
    b_h4 =    tf.Variable(tf.truncated_normal([40]))  
    variable_summaries(b_h4)  
    b_h5 =    tf.Variable(tf.truncated_normal([40]))  
    variable_summaries(b_h5)  
    b_h6 = tf.Variable(tf.truncated_normal([40]))
    variable_summaries(b_h6)
    b_h7 = tf.Variable(tf.truncated_normal([40]))
    variable_summaries(b_h7)
    b_h8 = tf.Variable(tf.truncated_normal([40]))
    variable_summaries(b_h8)
    b = tf.Variable(tf.truncated_normal([6]))   
    variable_summaries(b)  
  saver = tf.train.Saver([W, b, W_h8, b_h8, W_h7, b_h7, W_h6, b_h6, W_h5, b_h5, W_h4, b_h4, W_h3, b_h3, W_h2, b_h2,  W_h1,  b_h1])

  sess.run(tf.global_variables_initializer())  
  if restoreVars == True:
    restoreStr = saver.restore(sess, modelName + scale)
  if modelName ==  '8sig1matmul':
    hidden1 = tf.sigmoid(tf.matmul(measureds_ph, W_h1) + b_h1)
    hidden2 = tf.sigmoid(tf.matmul(hidden1, W_h2) + b_h2)
    hidden3 = tf.sigmoid(tf.matmul(hidden2, W_h3) + b_h3)
    hidden4 = tf.sigmoid(tf.matmul(hidden3, W_h4) + b_h4)
    hidden5 = tf.sigmoid(tf.matmul(hidden4, W_h5) + b_h5)
    hidden6 = tf.sigmoid(tf.matmul(hidden5, W_h6) + b_h6)
    hidden7 = tf.sigmoid(tf.matmul(hidden6, W_h7) + b_h7)
    hidden8 = tf.sigmoid(tf.matmul(hidden7, W_h8) + b_h8)
    y_ = (tf.matmul(hidden8,W) + b)
  elif  modelName ==  '4sig1matmul':   
    hidden1 = tf.sigmoid(tf.matmul(measureds_ph, W_h1) + b_h1)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, W_h2) + b_h2)
    hidden3 = tf.sigmoid(tf.matmul(hidden2, W_h3) + b_h3)
    hidden4 = tf.nn.relu(tf.matmul(hidden3, W_h4) + b_h4)
    y_ = (tf.matmul(hidden4,W) + b)
  elif  modelName ==  '5sigrelu1matmul':   
    hidden1 = tf.sigmoid(tf.matmul(measureds_ph, W_h1) + b_h1)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, W_h2) + b_h2)
    hidden3 = tf.sigmoid(tf.matmul(hidden2, W_h3) + b_h3)
    hidden4 = tf.nn.relu(tf.matmul(hidden3, W_h4) + b_h4)
    hidden5 = tf.sigmoid(tf.matmul(hidden4, W_h5) + b_h5)
    hidden6 = tf.sigmoid(tf.matmul(hidden5, W_h6) + b_h6)
    hidden7 = tf.nn.relu(tf.matmul(hidden6, W_h7) + b_h7)
    hidden8 = tf.sigmoid(tf.matmul(hidden7, W_h8) + b_h8)
    y_ = (tf.matmul(hidden8,W) + b)
  #tf_loss = tf.losses.mean_squared_error(y_, trues_ph, weights=1.0, scope=None,)
  #tf_loss = tf.losses.mean_squared_error(y_, trues_ph, weights=1.0, scope=None,)
  with tf.name_scope('losses'):
    tf_loss = tf.reduce_sum(tf.square(y_- trues_ph)) # sum of the squares
    variable_summaries(tf_loss)  
  #tf_loss = tf.reduce_sum(tf.losses.log_loss(y_, trues_ph))
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter('tboard', sess.graph)
  starter_learning_rate = 0.000002
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 0.96, staircase=True)
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_loss)    
  tf.logging.set_verbosity(tf.logging.DEBUG) 
  
  #pdb.set_trace()s
  kilobatchMinLoss  =  1000000
  kilobatchMinIt = -1
  kilobatchMaxLoss = 0
  kilobatchMinMaxLoss = 1000000
  kilobatchMinMaxIt = -1
  maxLoss = 0
  maxLossIteration = -1
  minLoss = 1000000000000
  minlossIteration = -1
  breakAtIteration = -1
  tferrcount = 0
  for _ in range(1200000):   #try:
    batch = nnTrackFilter.train.next_batch(100)
 #   except:
    #  tferrcount = tferrcount + 1
      #print("tf err @ iteration " + Str(_))
      #pdb.set_trace()
   #   batch = nnTrackFilter.train.next_batch(100)
      #break
    #pdb.set_trace()
    tstep , rmsVal = sess.run([train_step, tf_loss],feed_dict={measureds_ph : batch[0], trues_ph: batch[1]})
    if rmsVal < kilobatchMinLoss:
      kilobatchMinLoss = rmsVal
      kilobatchMinIt = _
    if rmsVal > kilobatchMaxLoss:
      kilobatchMaxLoss = rmsVal
      kilobatchMaxIt = _
    if _ == breakAtIteration:
        pdb.set_trace()
        breakAtIteration = 201
    pdb.set_trace()
    sess.graph.get_operations()
    #print('iteration: ' + str(_) + '  loss:   {0:.5f}'.format(rmsVal))
    if _ % 20000 == 1  and _ > 10:                  # print(tf.get_default_session().run(W))
      print("kBatch Min:{0:.5f}".format(kilobatchMinLoss) + " @ " +str(kilobatchMinIt))
      print("kBatch Max:{0:.5f}".format(kilobatchMaxLoss) + " @ " +str(kilobatchMaxIt))
      if kilobatchMaxLoss < kilobatchMinMaxLoss:
        kilobatchMinMaxLoss =  kilobatchMaxLoss
        kilobatchMinMaxIt = kilobatchMaxIt
      kilobatchMaxLoss = rmsVal
      kilobatchMinLoss = rmsVal
      kilobatchMaxIt = _
      kilobatchMinIt = _
      print('min loss: {0:.5f}'.format(minLoss) + '  @ iteration ' + str(minlossIteration))
      #train_step.run( feed_dict={measureds_ph : batch[0], trues_ph: batch[1]})
      #pdb.set_trace()

      #print('batch[1][1]:   ' + datasetString(batch[1][1],rounder)) , merged  , summary
      y, trus, summary= sess.run([ y_, trues_ph, merged],feed_dict={measureds_ph : batch[0], trues_ph: batch[1]})
      #try:
      train_writer.add_summary(summary, _)
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
   # if (_  -  minlossIteration) > 500000  and  (_  -  kilobatchMinMaxIt) > 500000:
  #    break
      #print('sum Square:',  str(pisumSq))
      #pims = (pisumSq/6)
      #print('pi-mean-Sq: {0:.3f}'.format(pims))
      #print(x)
     # print(y)
    #print(x)
    #with tf.Session() as sesh:
    #  print(sesh.run(tf_loss))
  print('final loss: {0:.5f}'.format(rmsVal) + '  @ iteration ' + str(_)) 
  print('final max loss: {0:.5f}'.format(maxLoss) + '  @ iteration ' + str(maxLossIteration)) 
  print('final min loss: {0:.5f}'.format(minLoss) + '  @ iteration ' + str(minlossIteration))

  accuracy = tf.reduce_mean(tf.square(y_- trues_ph))
  batch = nnTrackFilter.validation.next_batch(10000)
  acc, predicks, troos = sess.run([accuracy,y_, trues_ph,],feed_dict={measureds_ph : batch[0], trues_ph: batch[1]})
  print('*************validation test for ' +  modelName + ' with ' + scale + 'scaling')
  print('accuracy10k:' + str(acc))
  if scale == 'lin':
    for mm in range(10):
      print(datasetString(invLinearScale(predicks[mm]), 0))
      print(datasetString(invLinearScale(troos[mm]), 0))
  else:
    for mm in range(10):         
      print(datasetString(invLogScale(predicks[mm]), 0))
      print(datasetString(invLogScale(troos[mm]), 0))        
  batch = nnTrackFilter.validation.next_batch(1000)
  acc, predicks, troos = sess.run([accuracy,y_, trues_ph],feed_dict={measureds_ph : batch[0], trues_ph: batch[1]})
  print('accuracy1k:' + str(acc))
  filecsvName = modelName + scale + '.csv'
  filecsv = open(filecsvName, 'wb')
  filecsvStr = ''
  if scale =='lin':
    for mm in range(1000):
      filecsvStr = filecsvStr  +  datasetString(invLinearScale(predicks[mm]),0)  + ',' + datasetString(invLinearScale(troos[mm]) , 0) + '\r\n'
  else:  
    for mm in range(1000):
       filecsvStr = filecsvStr  +  datasetString(invLogScale(predicks[mm]),0)  + ',' + datasetString(invLogScale(troos[mm]) , 0) + '\r\n'
  filecsv.write(filecsvStr)
  filecsv.close
  
  batch = nnTrackFilter.validation.next_batch(100)
  acc, predicks, troos = sess.run([accuracy,y_, trues_ph,],feed_dict={measureds_ph : batch[0], trues_ph: batch[1]})
  print('accuracy100:' + str(acc))
  if scale == 'lin':
    for mm in range(10):
      print(datasetString(invLinearScale(predicks[mm]), 0))
      print(datasetString(invLinearScale(troos[mm]), 0))
  else:
    for mm in range(10):
      print(datasetString(invLogScale(predicks[mm]), 0))
      print(datasetString(invLogScale(troos[mm]), 0))
  batch = nnTrackFilter.validation.next_batch(10)
  acc, predicks, troos = sess.run([accuracy,y_, trues_ph,],feed_dict={measureds_ph : batch[0], trues_ph: batch[1]})
  print('accuracy10:' + str(acc))
  if scale == 'lin':
    for mm in range(10):
      print(datasetString(invLinearScale(predicks[mm]), 0))
      print(datasetString(invLinearScale(troos[mm]), 0))
  else:
     print(datasetString(invLogScale(predicks[mm]), 0))
     print(datasetString(invLogScale(troos[mm]), 0))         

  saver.save(sess,modelName + scale)
  
def testLogScale():
  kk = unipoleLogScale([-8000, 60000,45, 80, 90, 100])
  print(kk)
  print(invLogScale(kk))
  
if __name__ == "__main__":
  for dd in range(1):
    main('lin','8sig1matmul')
  #codeModelTest(False)

"""
lin','5sig1matmul'
kBatch Min:0.08303 @ 964648
kBatch Max:117.84129 @ 977063
min loss: 0.05423  @ iteration 549316
final loss: 30.18925  @ iteration 1049317
final max loss: 4845.82324  @ iteration 0
final min loss: 0.05423  @ iteration 549316

1 million batches

100kBatch Min:0.00247 @ 9235524
100kBatch Max:1.52284 @ 9237758
min loss: 0.00245  @ iteration 8392532
100kBatch Min:0.00248 @ 9324260
100kBatch Max:1.51984 @ 9326494
min loss: 0.00245  @ iteration 8392532
100kBatch Min:0.00248 @ 9412996
100kBatch Max:1.51687 @ 9415230
min loss: 0.00245  @ iteration 8392532
100kBatch Min:0.00249 @ 9501732
100kBatch Max:1.51390 @ 9503966
min loss: 0.00245  @ iteration 8392532
100kBatch Min:0.00250 @ 9634836
100kBatch Max:1.50950 @ 9637070
min loss: 0.00245  @ iteration 8392532
100kBatch Min:0.00251 @ 9723572
100kBatch Max:1.50660 @ 9725806
min loss: 0.00245  @ iteration 8392532
100kBatch Min:0.00251 @ 9812308
100kBatch Max:1.50371 @ 9814542
min loss: 0.00245  @ iteration 8392532
final loss: 0.01102  @ iteration 9999999
max loss: 6.93543  @ iteration 340
min loss: 0.00245  @ iteration 8392532
tensorflow complete
accuracy1k:[0.013362693]
accuracy100:[0.011243958]
accuracy10:[0.043907799]
accuracy10k:[0.025602074]



"""


