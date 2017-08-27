from __future__ import print_function

import os
import sys
sys.path.append('..')
import argparse
import time
import mxnet as mx
import cv2
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from mxnet.gluon import Trainer, utils, loss, ndarray
from mxnet import autograd as ag

from model_def.dcgan import DCGAN as dcgan

parser = argparse.ArgumentParser(description="Train DCGAN.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='celeba',
                        help='dataset used for training')
parser.add_argument('--num_layer', type=int, default=3,
                        help='number of intermediate layers')
parser.add_argument('--num_gpu', type=int, default='0',
                        help='number of gpus used. 0 means using cpu')
parser.add_argument('--num_epoch', type=int, default=25,
                        help='max num of epochs')
parser.add_argument('--lr', type=float, default=0.0002,
                        help='initial learning rate')
parser.add_argument('--beta1', type=float, default=0.5,
                        help='Exponential decay rate for the first moment estimates for adam')
parser.add_argument('--wd', type=float, default=0,
                        help='weight decay for adam')
parser.add_argument('--latent_vector_size', type=int, default=100,
                        help='Length of latent vector')
parser.add_argument('--num_image', type=int, default=-1,
                        help='Number of training images')
parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size.')
parser.add_argument('--use_hybrid', type=bool, default=True,
                        help='Whether to use hybrid mode.')
parser.add_argument('--interactive', type=bool, default=False,
                        help='Whether show output images for every 10 batches.')
parser.add_argument('--save_model', type=bool, default=True,
                        help='Whether save model.')

img_dim = (3, 64, 64)

# Utility functions
class RandIter(mx.io.DataIter):
    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('latent_vector', (batch_size, ndim, 1, 1))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        return [mx.random.normal(0, 1.0, shape=(self.batch_size, self.ndim, 1, 1))]

def resize(img, target_wd, target_ht):
    img_arr = cv2.imread(img)
    height, width = img_arr.shape[:2]
    interpolation = cv2.INTER_AREA if height > target_ht and width > target_wd \
        else cv2.INTER_LINEAR
    resized_img = cv2.resize(img_arr, (target_ht, target_wd), interpolation=interpolation)
    return resized_img

def prep_data(target_wd, target_ht, args):
    parent_folder = '../datasets/'
    data_path = parent_folder + args.dataset
    data=[]
    img_files = [os.path.join(data_path, fname) for fname in os.listdir(data_path)]
    num_img = len(img_files) if args.num_image == -1 else args.num_image
    for i, image in enumerate(img_files):
        if i == num_img:
            break
        if not image.endswith('jpg'):
            continue
        resized = resize(image, target_wd, target_ht)
        normalized = np.array(resized/127.5 - 1)
        data.append(np.rollaxis(normalized, 2, 0))
    train_iter = mx.io.NDArrayIter(data=mx.nd.array(data), batch_size=args.batch_size,
                                   label_name='logistic_label')
    return train_iter

def fill_buf(buf, i, img, shape):
    n = buf.shape[0]/shape[1]
    m = buf.shape[1]/shape[0]

    sx = (i%m)*shape[0]
    sy = (i/m)*shape[1]
    buf[sy:sy+shape[1], sx:sx+shape[0], :] = img

def visual(title, X):
    assert len(X.shape) == 4
    X = X.transpose((0, 2, 3, 1))
    X = np.clip((X+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    n = np.ceil(np.sqrt(X.shape[0]))
    buff = np.zeros((int(n*X.shape[1]), int(n*X.shape[2]), int(X.shape[3])), dtype=np.uint8)
    for i, img in enumerate(X):
        fill_buf(buff, i, img, X.shape[1:3])
    buff = cv2.cvtColor(buff, cv2.COLOR_BGR2RGB)
    plt.imshow(buff)
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    ctx = [mx.gpu(int(i)) for i in range(args.num_gpu)] if args.num_gpu > 0 else mx.cpu(0)

    print("Preprocessing data...")
    train_iter = prep_data(img_dim[1], img_dim[2], args)
    rand_iter = RandIter(args.batch_size, args.latent_vector_size)
    label = mx.nd.zeros((args.batch_size, 1))

    print("Building model...")
    dcgan_builder = dcgan(num_layer=args.num_layer)

    # Initialize parameters and set optimizer for generator
    generator = dcgan_builder.make_generator()
    generator.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)
    gen_trainer = Trainer(generator.collect_params(), 'adam',
        {
            'learning_rate': args.lr,
            'wd': args.wd,
            'beta1': args.beta1,
        })

    # Initialize parameters and set optimizer for discriminator
    discriminator = dcgan_builder.make_discriminator()
    dist_loss_func = loss.SigmoidBinaryCrossEntropyLoss()
    discriminator.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)
    dist_trainer = Trainer(discriminator.collect_params(), 'adam',
        {
            'learning_rate': args.lr,
            'wd': args.wd,
            'beta1': args.beta1,
        })

    if args.use_hybrid:
        generator.hybridize()
        discriminator.hybridize()

    # Printing utility function
    def norm_stat(d):
        return mx.nd.norm(d) / np.sqrt(d.size)

    def facc(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return ((pred > 0.5) == label).mean()

    def fentropy(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return -(label * np.log(pred + 1e-12) + (1. - label) * np.log(1. - pred + 1e-12)).mean()

    mG = mx.metric.CustomMetric(fentropy)
    mD = mx.metric.CustomMetric(fentropy)
    mACC = mx.metric.CustomMetric(facc)

    print('Training...')
    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')

    # =============train===============
    start_time = time.time()
    for epoch in range(args.num_epoch):
        train_iter.reset()
        iter_num = 0
        for t, batch in enumerate(train_iter):
            # Draw z_latent vector from normal distribution and generate images.
            rand_batch = rand_iter.next()
            real_img = utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            split_rand_vector = utils.split_and_load(rand_batch.data[0], ctx_list=ctx, batch_axis=0)
            gen_img = []
            for data_slice in split_rand_vector:
                gen_img.append(generator.forward(data_slice))

            # Train discriminator using both fake and real images.
            label[:] = 0
            split_false_label = utils.split_and_load(label, ctx_list=ctx, batch_axis=0)
            label[:] = 1
            split_true_label = utils.split_and_load(label, ctx_list=ctx, batch_axis=0)
            dist_gen_out = []
            dist_real_out = []
            dist_loss = []
            with ag.record():
                for gen_img_slice, real_img_slice, false_label_slice, true_label_slice \
                        in zip(gen_img, real_img, split_false_label, split_true_label):
                    gen_img_out = discriminator.forward(gen_img_slice)
                    real_img_out = discriminator.forward(real_img_slice)
                    dist_gen_out.append(mx.nd.Activation(gen_img_out, act_type='sigmoid'))
                    dist_real_out.append(mx.nd.Activation(real_img_out, act_type='sigmoid'))
                    loss = dist_loss_func(gen_img_out, false_label_slice) + \
                           dist_loss_func(real_img_out, true_label_slice)
                    dist_loss.append(loss)
                # store the loss and do backward after we have done forward
                # on all GPUs for better speed on multiple GPUs.
                for loss in dist_loss:
                    loss.backward()
            dist_trainer.step(batch.data[0].shape[0], ignore_stale_grad=True)
            mD.update(split_false_label, dist_gen_out)
            mD.update(split_true_label, dist_real_out)
            mACC.update(split_false_label, dist_gen_out)
            mACC.update(split_true_label, dist_real_out)

            # Train generator. We only need to update generator.
            rand_batch = rand_iter.next()
            split_rand_vector = utils.split_and_load(rand_batch.data[0], ctx_list=ctx, batch_axis=0)
            dist_out = []
            dist_loss = []
            with ag.record():
                for data_slice, true_label_slice in zip(split_rand_vector, split_true_label):
                    gen_img = generator.forward(data_slice)
                    output = discriminator.forward(gen_img)
                    loss = dist_loss_func(output, true_label_slice)
                    dist_out.append(mx.nd.Activation(output, act_type='sigmoid'))
                    dist_loss.append(loss)
                for loss in dist_loss:
                    loss.backward()
            gen_trainer.step(rand_batch.data[0].shape[0])
            mG.update(split_true_label, dist_out)

            iter_num += 1
            if t % 10 == 0:
                print('epoch:', epoch, 'iter:', iter_num, 'metric:', mACC.get(), mG.get(), mD.get())
                mACC.reset()
                mG.reset()
                mD.reset()

                if args.interactive:
                    latent_vector = mx.random.normal(0, 1.0, shape=(args.batch_size,
                                                                    args.latent_vector_size, 1, 1))
                    gout = generator.forward(latent_vector)
                    visual('gout', gout.asnumpy())
                    visual('data', batch.data[0].asnumpy())
    print("Total training time is %s." % (str(time.time() - start_time)))

    if args.save_model:
        print('Saving model...')
        generator.save_params('%s_G_%s-%04d.params' % (args.dataset, stamp, args.num_epoch))
        discriminator.save_params('%s_D_%s-%04d.params' % (args.dataset, stamp, args.num_epoch))
