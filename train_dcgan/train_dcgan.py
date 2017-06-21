from __future__ import print_function

import os
import sys
sys.path.append('..')
import argparse
import mxnet as mx
import cv2
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

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
parser.add_argument('--batch_size', type=int, default=32,
                        help='The batch size.')
parser.add_argument('--interactive', type=bool, default=False,
                        help='Whether show output images for every 10 batches.')
parser.add_argument('--save_model', type=bool, default=True,
                        help='Whether save model.')

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

def prep_data():
    img_dim = (3, 64, 64)
    parent_folder = '/Users/kevinthesun1/Downloads/'
    data_path = parent_folder + args.dataset
    resized_data_path = data_path + '_resized'
    img_lst = args.dataset + '.lst'
    create_lst_cmd = 'python im2rec.py %s %s --list True' % (args.dataset, data_path)

    #Resize images to 64x64
    os.system(create_lst_cmd)
    if not os.path.isdir(resized_data_path):
        os.system('mkdir %s' % (resized_data_path))
        img_files = [os.path.join(data_path, fname) for fname in os.listdir(data_path)]
        for image in img_files:
            if not image.endswith('jpg'):
                continue
            img_arr = cv2.imread(image)
            resized = cv2.resize(img_arr, (img_dim[1], img_dim[2]), interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(resized_data_path + '/' + os.path.basename(image), resized):
                raise IOError("Error occurred while preprocessing images")


    train_iter = mx.img.ImageIter(batch_size=args.batch_size, data_shape=img_dim,
                                  path_root=resized_data_path, path_imglist=img_lst,
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
    ctx = [mx.gpu(i) for i in range(args.num_gpu)] if args.num_gpu > 0 else mx.cpu(0)

    print("Preprocessing data...")
    train_iter = prep_data()
    rand_iter = RandIter(args.batch_size, args.latent_vector_size)
    label = mx.nd.zeros((args.batch_size,), ctx=ctx)

    print("Building model...")
    dcgan_builder = dcgan(num_layer=args.num_layer)

    #Create generator module
    z_latent = mx.sym.Variable('latent_vector')
    gen_sym = dcgan_builder.make_generator(z_latent=z_latent)

    mod_gen = mx.mod.Module(symbol=gen_sym, data_names=('latent_vector',),
                            label_names=None, context=ctx)
    mod_gen.bind(data_shapes=rand_iter.provide_data)
    mod_gen.init_params(initializer=mx.init.Normal(0.02))
    mod_gen.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': args.lr,
            'wd': args.wd,
            'beta1': args.beta1,
        })
    mods = [mod_gen]

    #Create discriminator module
    data = mx.sym.Variable('data')
    label_dist = mx.sym.Variable('logistic_label')
    dist_sym = dcgan_builder.make_discriminator(data, label_dist)

    mod_dist = mx.mod.Module(symbol=dist_sym, data_names=('data',),
                             label_names=('logistic_label',), context=ctx)
    mod_dist.bind(data_shapes=train_iter.provide_data,
              label_shapes=[('logistic_label', (args.batch_size,))],
              inputs_need_grad=True)
    mod_dist.init_params(initializer=mx.init.Normal(0.02))
    mod_dist.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': args.lr,
            'wd': args.wd,
            'beta1': args.beta1,
        })
    mods.append(mod_dist)

    # Printing utility function
    def norm_stat(d):
        return mx.nd.norm(d) / np.sqrt(d.size)

    mon = mx.mon.Monitor(10, norm_stat, pattern=".*output|d1_backward_data", sort=True)
    if mon is not None:
        for mod in mods:
            mod.install_monitor(mon)

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

    print('Training generator and discriminator...')
    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')

    # =============train===============
    for epoch in range(args.num_epoch):
        train_iter.reset()
        for t, batch in enumerate(train_iter):
            # Draw z_latent vector from normal distribution
            rbatch = rand_iter.next()

            if mon is not None:
                mon.tic()

            mod_gen.forward(rbatch, is_train=True)
            out_gen = mod_gen.get_outputs()

            # Train discriminator on generated fake images
            label[:] = 0
            mod_dist.forward(mx.io.DataBatch(out_gen, [label]), is_train=True)
            mod_dist.backward()
            grad_dist = [[grad.copyto(grad.context) for grad in grads] for \
                         grads in mod_dist._exec_group.grad_arrays]
            mod_dist.update_metric(mD, [label])
            mod_dist.update_metric(mACC, [label])

            # Train discriminator on real images
            label[:] = 1
            batch.label = [label]
            mod_dist.forward(batch, is_train=True)
            mod_dist.backward()
            for gradsr, gradsf in zip(mod_dist._exec_group.grad_arrays, grad_dist):
                for gradr, gradf in zip(gradsr, gradsf):
                    gradr += gradf
            mod_dist.update()
            mod_dist.update_metric(mD, [label])
            mod_dist.update_metric(mACC, [label])

            # Train generator
            label[:] = 1
            mod_dist.forward(mx.io.DataBatch(out_gen, [label]), is_train=True)
            mod_dist.backward()
            diff_dist = mod_dist.get_input_grads()
            mod_gen.backward(diff_dist)
            mod_gen.update()

            mG.update([label], mod_dist.get_outputs())

            if mon is not None:
                mon.toc_print()

            t += 1
            if t % 10 == 0:
                print('epoch:', epoch, 'iter:', t, 'metric:', mACC.get(), mG.get(), mD.get())
                mACC.reset()
                mG.reset()
                mD.reset()

                if args.interactive:
                    visual('gout', out_gen[0].asnumpy())
                    diff = diff_dist[0].asnumpy()
                    diff = (diff - diff.mean()) / diff.std()
                    visual('diff', diff)
                    visual('data', batch.data[0].asnumpy())

        if args.save_model:
            print('Saving model...')
            mod_gen.save_params('%s_G_%s-%04d.params' % (args.dataset, stamp, args.num_epoch))
            mod_gen.save_params('%s_D_%s-%04d.params' % (args.dataset, stamp, args.num_epoch))

    print('Training predictor')

