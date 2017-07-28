from mxnet.gluon.nn import Dense, Activation, Conv2D, Conv2DTranspose, \
    BatchNorm, LeakyReLU, Flatten, Sequential

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
        self.generator = Sequential()

        self._trans_conv_factory(self.generator, kernel=(4, 4), num_filter=ngf * 2 ** nlayer)
        for idx in range(nlayer):
            self._trans_conv_factory(self.generator, kernel=(4, 4), strides=(2, 2), padding=(1, 1),
                                     num_filter=ngf * 2 ** (nlayer - idx - 1))

        self._trans_conv_factory(self.generator, kernel=(4, 4), strides=(2, 2), padding=(1, 1),
                                 num_filter=nc, apply_bn=False, act_type='tanh')

    def make_discriminator(self, use_bias=False, scale=False, epsilon=1e-5 + 1e-12):
        ndf = self.num_d_filter
        nlayer = self.num_layer
        self.discriminator = Sequential()

        self._conv_factory(self.discriminator, kernel=(4, 4), strides=(2, 2), padding=(1, 1),
                           apply_bn=False)
        for idx in range(nlayer):
            self._conv_factory(self.discriminator, kernel=(4, 4), strides=(2, 2), padding=(1, 1),
                               num_filter=ndf * 2 ** (idx + 1))

        self.discriminator.add(Conv2D(channels=1, kernel=(4, 4), use_bias=use_bias))
        self.discriminator.add(Flatten())

    def make_predictor(self, num_dim, use_bias=False, scale=False, epsilon=1e-5 + 1e-12):
        ndf = self.num_d_filter
        nlayer = self.num_layer
        self.predictor = Sequential()

        self._conv_factory(self.predictor, kernel=(4, 4), strides=(2, 2), padding=(1, 1),
                           apply_bn=False)
        for idx in range(nlayer):
            self._conv_factory(self.predictor, kernel=(4, 4), strides=(2, 2), padding=(1, 1),
                               num_filter=ndf * 2 ** (idx + 1))

        self.predictor.add(Conv2D(channels=1, kernel=(4, 4), use_bias=use_bias))
        self.predictor.add(Flatten())
        self.predictor.add(Dense(units=num_dim, activation='tanh'))

    def _trans_conv_factory(self, model, kernel, strides=(1, 1), padding=(0, 0),
                            num_filter=self.num_d_filter, apply_bn=True, act_type='relu'):
        model.add(Conv2DTranspose(channels=num_filter, kernel=kernel, strides=strides,
                                      pad=padding, use_bias=use_bias))
        if apply_bn:
            model.add(BatchNorm(scale=scale, epsilon=epsilon))
        model.add(Activation(activation=act_type))

    def _conv_factory(self, model, kernel, strides=(1, 1), padding=(0, 0), num_filter=self.num_d_filter,
                      apply_bn=True):
        model.add(Conv2D(channels=num_filter, kernel=kernel, strides=strides,
                                 padding=padding, use_bias=use_bias))
        if apply_bn:
            model.add(BatchNorm(scale=scale, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))

