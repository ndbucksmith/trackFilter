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
      print('self._epochs_completed:' + str(self._epochs_completed))
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
      return numpy.concatenate((pv6inputs_rest_part, pv6inputs_new_part), axis=0) , numpy.concatenate((pv6trues_rest_part, pv6trues_new_part), axis=0)
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
  print(numrecordspv6Inputs)
  numrecordspv6trues = len(trainSetTrues)  
  print(numrecordspv6trues)
  assert numrecordspv6trues == numrecordspv6Inputs
  validation_size = int( 0.2 * len(trainSetIns))
  print(validation_size)
  #now scale to -1 to 1, magic numbers collected in track gen.py
    #492 756157.522271 810653.565324 104353.717046
  #e.g 491 803464.7 787912.7 98608.3
  #now 479 745493.892057 791520.705135 100903.802122
  #pdb.set_trace()
  if scale:   #turn scaling on and off
    for bb in range(0,numrecordspv6Inputs-1):
      trainSetIns[bb][0] = trainSetIns[bb][0] / 95000
      trainSetIns[bb][1] = trainSetIns[bb][1] / 95000
      trainSetIns[bb][2] = trainSetIns[bb][2] / 110000
      trainSetIns[bb][3] = trainSetIns[bb][3] / 2000
      trainSetIns[bb][4] = trainSetIns[bb][4] / 2000
      trainSetIns[bb][5] = trainSetIns[bb][5] / 2000
      trainSetIns[bb][6] = trainSetIns[bb][6] / 90000
      trainSetIns[bb][7] = trainSetIns[bb][7] / 90000
      trainSetIns[bb][8] = trainSetIns[bb][8] / 110000
      trainSetIns[bb][9] = trainSetIns[bb][9] / 2000
      trainSetIns[bb][10] = trainSetIns[bb][10] / 2000
      trainSetIns[bb][11] = trainSetIns[bb][11] / 2000
    
      trainSetTrues[bb][0] = trainSetTrues[bb][0] / 95000
      trainSetTrues[bb][1] = trainSetTrues[bb][1] / 95000
      trainSetTrues[bb][2] = trainSetTrues[bb][2] / 110000
      trainSetTrues[bb][3] = trainSetTrues[bb][3] / 2000
      trainSetTrues[bb][4] = trainSetTrues[bb][4] / 2000
      trainSetTrues[bb][5] = trainSetTrues[bb][5] / 2000
    
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
        #print(sumSquaresList[bb],sumSquares)
        #pdb.set_trace()
  for cc in range(0,6):
    meanSquareList[cc] = sumSquaresList[cc]/len(batch1)  
    if meanSquareList[cc] > maxMean:
      maxMean = meanSquareList[cc]
      maxMeanIdx = cc
  meanSumSquares = sumSquares/(6*len(batch1))
  #print('meansquares ' +  datasetString(meanSquareList,rounder))
  print('max mean ' +  str(round(meanSquareList[maxMeanIdx],rounder) )  + " @idx: " + str(maxMeanIdx))
  #print("mean squares  " + str(round(meanSumSquares,rounder)))
  print("sum squares  " + str(round(sumSquares,rounder)))
    
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
    W_h1 = tf.Variable(tf.truncated_normal([12,24]))
    variable_summaries(W_h1)
    W_h2 = tf.Variable(tf.truncated_normal([24,20]))
    variable_summaries(W_h2)
    W = tf.Variable(tf.truncated_normal([20,6]))   
    variable_summaries(W)
  with tf.name_scope('biases'):
    b_h1 =    tf.Variable(tf.zeros([24]))   
    variable_summaries(b_h1)  
    b_h2 =    tf.Variable(tf.zeros([20]))  
    variable_summaries(b_h2)      
    b = tf.Variable(tf.zeros([6]))   
    variable_summaries(b)  
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter('tboard', sess.graph)
  sess.run(tf.global_variables_initializer())  
  hidden1 = tf.nn.relu(tf.matmul(measureds_ph, W_h1) + b_h1)
  hidden2 = tf.nn.relu(tf.matmul(hidden1, W_h2) + b_h2)
  y_ = tf.matmul(hidden2,W) + b      
  #tf_loss = tf.losses.mean_squared_error(y_, trues_ph, weights=1.0, scope=None,)
  #tf_loss = tf.losses.mean_squared_error(y_, trues_ph, weights=1.0, scope=None,)
  tf_loss = tf.reduce_sum(tf.square(y_- trues_ph)) # sum of the squares

  starter_learning_rate = 0.00001
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 0.96, staircase=True)
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_loss)    
  tf.logging.set_verbosity(tf.logging.DEBUG) 
  
  #pdb.set_trace()s
  maxLoss = 0
  maxLossIteration = -1
  minLoss = 1000000000000
  minlossIteration = -1
  breakAtIteration = 101
  for _ in range(200000):
    try:
      batch = nnTrackFilter.train.next_batch(100)
    except:
      print("tf err")
      break
    #pdb.set_trace()\
    tstep , rmsVal = sess.run([train_step, tf_loss],feed_dict={measureds_ph : batch[0], trues_ph: batch[1]})
    if _ % 100 == 1:                  # print(tf.get_default_session().run(W))
      #train_step.run( feed_dict={measureds_ph : batch[0], trues_ph: batch[1]})
      #pdb.set_trace()
      if _ == breakAtIteration:
        #pdb.set_trace()
        breakAtIteration = 201
      #print('batch[1][1]:   ' + datasetString(batch[1][1],rounder))
      y, trus, summary = sess.run([ y_, trues_ph, merged],feed_dict={measureds_ph : batch[0], trues_ph: batch[1]})
      train_writer.add_summary(summary, _)
      print('iteration: ' + str(_) + '  loss:   {0:.3f}'.format(rmsVal))
      meanSquare(trus, y, -1,scale)
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
  print('max loss: {0:.3f}'.format(maxLoss) + '  @ iteration ' + str(maxLossIteration)) 
  print('min loss: {0:.3f}'.format(minLoss) + '  @ iteration ' + str(minlossIteration))
  print("tensorflow complete")
  
  
  
if __name__ == "__main__":
  main(True)
  #codeModelTest(False)
        