from mxnet.gluon.nn import Dense, Activation, Conv2D, Conv2DTranspose, \
    BatchNorm, LeakyReLU, Flatten, HybridSequential


class DCGAN(object):
    """DCGAN model definition
    """
    def __init__(self, num_g_filter=64, num_d_filter=64, num_channel=3, num_layer=3):
        self.num_g_filter = num_g_filter
        self.num_d_filter = num_d_filter
        self.num_channel = num_channel
        self.num_layer = num_layer

    def make_generator(self, use_bias=False, scale=False, epsilon=1e-5 + 1e-12):
        ngf = self.num_g_filter
        nc = self.num_channel
        nlayer = self.num_layer
        generator = HybridSequential()

        with generator.name_scope():
            self._trans_conv_factory(generator, kernel=(4, 4), use_bias=use_bias, scale=scale,
                                     epsilon=epsilon, num_filter=ngf * 2 ** nlayer)
            for idx in range(nlayer):
                self._trans_conv_factory(generator, kernel=(4, 4), use_bias=use_bias, scale=scale,
                                         epsilon=epsilon, strides=(2, 2), padding=(1, 1),
                                         num_filter=ngf * 2 ** (nlayer - idx - 1))

            self._trans_conv_factory(generator, kernel=(4, 4), use_bias=use_bias, scale=scale,
                                     epsilon=epsilon, strides=(2, 2), padding=(1, 1),
                                     num_filter=nc, apply_bn=False, act_type='tanh')
            return generator

    def make_discriminator(self, use_bias=False, scale=False, epsilon=1e-5 + 1e-12):
        ndf = self.num_d_filter
        nlayer = self.num_layer
        discriminator = HybridSequential()

        with discriminator.name_scope():
            self._conv_factory(discriminator, kernel=(4, 4), use_bias=use_bias, scale=scale,
                               epsilon=epsilon, num_filter=ndf, strides=(2, 2), padding=(1, 1), 
                               apply_bn=False)
            for idx in range(nlayer):
                self._conv_factory(discriminator, kernel=(4, 4), use_bias=use_bias, scale=scale,
                                   epsilon=epsilon, num_filter=ndf * 2 ** (idx + 1), 
                                   strides=(2, 2), padding=(1, 1))
            discriminator.add(Conv2D(channels=1, kernel_size=(4, 4), use_bias=use_bias))
            discriminator.add(Flatten())
            return discriminator

    def make_predictor(self, num_dim, use_bias=False, scale=False, epsilon=1e-5 + 1e-12):
        ndf = self.num_d_filter
        nlayer = self.num_layer
        predictor = HybridSequential()

        with predictor.name_scope():
            self._conv_factory(predictor, kernel=(4, 4), use_bias=use_bias, scale=scale,
                               epsilon=epsilon, num_filter=ndf, strides=(2, 2), padding=(1, 1), 
                               apply_bn=False)
            for idx in range(nlayer):
                self._conv_factory(predictor, kernel=(4, 4), use_bias=use_bias, scale=scale,
                                   epsilon=epsilon, num_filter=ndf * 2 ** (idx + 1), 
                                   strides=(2, 2), padding=(1, 1))

            predictor.add(Conv2D(channels=1, kernel_size=(4, 4), use_bias=use_bias))
            predictor.add(Flatten())
            predictor.add(Dense(units=num_dim, activation='tanh'))
            return predictor

    def _trans_conv_factory(self, model, kernel, use_bias, scale, epsilon, num_filter,
                            strides=(1, 1), padding=(0, 0), apply_bn=True, act_type='relu'):
        model.add(Conv2DTranspose(channels=num_filter, kernel_size=kernel, strides=strides,
                                  pad=padding, use_bias=use_bias))
        if apply_bn:
            model.add(BatchNorm(scale=scale, epsilon=epsilon))
        model.add(Activation(activation=act_type))

    def _conv_factory(self, model, kernel, use_bias, scale, epsilon, num_filter, 
                      strides=(1, 1), padding=(0, 0), apply_bn=True):
        model.add(Conv2D(channels=num_filter, kernel_size=kernel, strides=strides,
                         padding=padding, use_bias=use_bias))
        if apply_bn:
            model.add(BatchNorm(scale=scale, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))
