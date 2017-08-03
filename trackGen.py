# Generate track data

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import pdb
import math
import pprint
import pickle

import argparse
import os
import sys

import tensorflow as tf
filetype11 ='pkl'
rocketAccelMax = 32  #fps^2
rocketAccelMin = 16
rocketThrustTimeMin=  30 #seconds
rocketThrusTimetMax = 60
rocketAzimuthMax = 80  #degrees
rocketAzimuthMin = 60

def _float32_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def aim():
  att = rocketAzimuthMin + random.random() * (rocketAzimuthMax-rocketAzimuthMin)
  azi = random.random() * 360
  fireTime = rocketThrustTimeMin + random.random() * (rocketThrusTimetMax-rocketThrustTimeMin)
  accel = rocketAccelMin + random.random()  *  (rocketAccelMax-rocketAccelMin)
  return att, azi, fireTime, accel
  
def fire(trkID):
  x_max, vx_max = 0,0
  y_max ,vy_max = 0,0
  z_max, vz_max= 0,0
  trues = []
  measureds = []
  at, az, ft, ac = aim()
  x,y,z = 0,0,0  #true x,y,z
  xm,ym,zm = 0,0,0
  vx,vy,vz = 0,0,0  #true velocitys
  vxm,vym,vzm = 0,0,0  #measured velocitys
  vx0,vy0,vz0 = 0,0,0  #previous true velocity-
  for ti in range(1, 1000):
    trueSix = []
    measSix = []
    vx0,vy0,vz0 =  vx,vy,vz 
    if ti < ft:   #firing
      vz = vz + math.sin(at/57.295) * ac
      vx = vx +math.sin(az/57.295) *  ac
      vy = vz + math.cos(az/57.295) * ac
    else:   # coasting
      if vz >  - 250:  #terminal velocity
        vz = vz - 32.2
      vx = vx * 0.9999
      vy = vy * 0.9999
    x = x + ((vx + vx0)/2)
    y = y + ((vy + vy0)/2)
    z = z + ((vz + vz0)/2)
    trueSix.append(x)
    trueSix.append(y)
    trueSix.append(z)
    trueSix.append(vx)
    trueSix.append(vy)
    trueSix.append(vz)
    #print(trueSix)
    #print("%.1f" % x,"%.1f" % y,"%.1f" % z, "%.1f" % vx, "%.1f" % vy, "%.1f" % vz)
    xm = x #+ ((random.random()-0.5) * 10)
    ym = y  #+ ((random.random()-0.5) * 10)
    zm = z  #+ ((random.random()-0.5) * 10)
    vxm = vx #*  (1 + ((random.random()-0.5) / 50.0))
    vym = vy #*  (1 + ((random.random()-0.5) / 50.0))  
    vzm = vz # *  (1 + ((random.random()-0.5) / 50.0)) 
    #rint(xm,x)
    measSix.append(xm)
    measSix.append(ym)
    measSix.append(zm)
    measSix.append(vxm)
    measSix.append(vym)
    measSix.append(vzm)
    trues.append(trueSix)
    measureds.append(measSix)
    #print('lenm6:', str(len(measureds)))
    if abs(x) > abs(x_max):
     x_max = abs(x)
    if abs(y) > abs(y_max):
       y_max = abs(y)
    if z > abs(z_max):
       z_max = z 
    if abs(vx) > abs(vx_max):
      vx_max = abs(vx)
    if abs(vy) > abs(vy_max):
      vy_max = abs(vy)
    if abs(vz) > abs(vz_max):
      vz_max = abs(vz) 
    #print("%.1f" % xm,"%.1f" % ym,"%.1f" % zm, "%.1f" % vxm, "%.1f" % vym, "%.1f" % vzm)
    if z < 0:
      print(ti)      
      break
  #pdb.set_trace()    
  #print(x_max,y_max,z_max)
  if (filetype11 == 'pkl') :
    outputData = []
    inputData = []
    for  kk in range(5, ti-3):
      inputSix= measureds[kk-1] + measureds[kk]
      outputSix = trues[kk+2] 
      print(measureds[kk][0], trues[kk+2][0])
      inputData.append(inputSix)
      outputData.append(outputSix)
  #print(outputData)
      #print(ti)          
      #print(int(ft))
      #qq = len(outputData)
      #print(qq)
    fninpStr = 'tracks/Inp-' + str(trkID) + '.trk'
    fntruStr = 'tracks/Tru-' + str(trkID) + '.trk'
    #pdb.set_trace()
    dof=open(fninpStr, 'wb')
    pickle.dump(inputData, dof) 
    dof.flush()
    dof.close()
    dof=open(fntruStr, 'wb')
    pickle.dump(outputData, dof) 
    dof.flush()
    dof.close()
  else: 
    fn = 'tracks/t-' + str(trkID) + '.tfrecords'     
    writer = tf.python_io.TFRecordWriter(fn)
    #pdb.set_trace()
    for  kk in range(5, ti-3):
      example = tf.train.Example(features=tf.train.Features(feature={
              'xm0':   _float32_feature(measureds[kk-1][0]),    
              'ym0':   _float32_feature(measureds[kk-1][1]),                 
              'zm0':   _float32_feature(measureds[kk-1][2]),    
              'vxm0':  _float32_feature(measureds[kk-1][3]),         
              'vym0':  _float32_feature(measureds[kk-1][4]),   
              'vzm0':  _float32_feature(measureds[kk-1][5]),
              'xm':   _float32_feature(measureds[kk][6]),    
              'ym':   _float32_feature(measureds[kk][7]),                 
              'zm':   _float32_feature(measureds[kk][8]),    
              'vxm':  _float32_feature(measureds[kk][9]),         
              'vym':  _float32_feature(measureds[kk][10]),   
              'vzm':  _float32_feature(measureds[kk][11]),         
              'x':     _float32_feature(trues[kk][0]),    
              'y':    _float32_feature(trues[kk][1]),                 
              'z':     _float32_feature(trues[kk][2]),    
              'vx':    _float32_feature(trues[kk][3]),         
              'vy':    _float32_feature(trues[kk][4]),    
              'vz':    _float32_feature(trues[kk][5])}))
      writer.write(example.SerializeToString())
    writer.close()
  return ti, x_max,y_max,z_max, vx_max, vy_max, vz_max
 
def datasetString(Values, rounder):
  strg = ""
  for kk in range(len(Values)):
    strg = strg +  str(round(Values[kk],rounder)) + ", "
  strg = strg[:-2]  
  return strg
  
def main():
  tf_maxmax, x_maxmax, y_maxmax, z_maxmax, vx_maxmax, vy_maxmax, vz_maxmax = 0,0,0,0,0,0,0
  tf, xmx, ymx, zmx,vxmx,vymx,vzmx = 0,0,0,0,0,0,0
  for gg in range(0,1000):
    tf, xmx, ymx, zmx,vxmx,vymx,vzmx  = fire(gg)
    #pdb.set_trace()
    #dont need abd here b/c abs is coerced in fire()
    if xmx > x_maxmax:
      x_maxmax = xmx         
    if ymx > y_maxmax:
      y_maxmax = ymx     
    if zmx > z_maxmax:
      z_maxmax = zmx  
    if tf > tf_maxmax:
      tf_maxmax = tf
    if vxmx > vx_maxmax:
      vx_maxmax = vxmx         
    if vymx > vy_maxmax:
      vy_maxmax = vymx     
    if vzmx > vz_maxmax:
      vz_maxmax = vzmx   
  print("fireTime, max positions:"  +  datasetString([tf_maxmax, x_maxmax, y_maxmax, z_maxmax],2))
  print("max velcoties:  " + datasetString([vx_maxmax, vy_maxmax, vz_maxmax],2))
  
  
#fireTime, max positions:537.0, 918258.21, 896022.98, 104329.4
#max velcoties:  1850.38, 1824.12, 1810.75

#fireTime, max positions:529.0, 827057.68, 901098.26, 103342.07
#max velcoties:  1786.96, 1840.59, 1808.97


      
if __name__ == "__main__":
  main()