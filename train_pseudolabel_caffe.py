#task 1: train full SUN with existing caffe net
#   subtask 1.1: do it from the command line
# ./build/tools/caffe.bin train -solver models/SUN_pseudolabels/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel -gpu 0
#   subtask 1.2: replicate it in python

import os
os.chdir('/media/MartinK3TB/Documents/caffe/Oct2015/caffe')

import sys
sys.path.insert(0, './python')
import caffe

from pylab import *
#matplotlib inline

from caffe import layers as L
from caffe import params as P


#create new model, which is the same as bvlc_reference_caffenet, except for the last layer
#DONE: do this outside python, with a new .prototxt file

#make sure that all layers except last have blobs_lr set to 0
#DONE: do this outside python, with a new .prototxt file

#load bvlc_reference_caffenet.caffemodel weights into it
#can be done with bash:
#bash ./build/tools/caffe train -solver models/finetune_flickr_style/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel -gpu 0
#in python:
caffe_root = '/media/MartinK3TB/Documents/caffe/'
net = caffe.Net(caffe_root + 'models/SUN_pseudolabels/train.prototxt',#this file defines training DATA
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TRAIN)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

#incorporate information that I want to retrain only last layer into loaded model
caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver(caffe_root + 'models/SUN_pseudolabels/solver.prototxt')

#   >>>>>>>>>>>>    ./prepare_SUN_train_test.sh    <<<<<<<<<<<<<

solver.net.forward()  # train net   -- necessary for images to be loaded into solver.net.blobs['data']
solver.test_nets[0].forward()  # test net (there can be more than one)

#show first image of dataset
#imshow(transformer.deprocess('data', solver.net.blobs['data'].data[0]))


#%%time
niter = 20000
test_interval = 25
# losses will also be stored in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))
output = zeros((niter, 8, 10))

# the main solver loop - TODO: I need to change training data here, somehow!
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    
    # history of prediction for the images of the first batch, for later analysis:
    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    #solver.test_nets[0].forward(start='conv1')
    #output[it] = solver.test_nets[0].blobs['ip2'].data[:8]
    
    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['fc8_pseudolabel'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4
        print 'Iteration:', it, 'test accuracy:', test_acc[it // test_interval]

print('training done')

_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')