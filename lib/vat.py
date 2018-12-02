import chainer
import chainer.functions as F
import chainer.links as L
#from chainer import training,cuda
from chainer import training
import cupy as xp
#import numpy as xp
import numpy as np
from scipy.stats import entropy

import six

from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer.training import _updater

class KL_multinominal(chainer.function.Function):
    """ KL divergence between multinominal distributions """

    def __init__(self):
        pass

    def forward_gpu(self, inputs):
        p, q = inputs
        loss = xp.ReductionKernel(
            'T p, T q',
            'T loss',
            'p*(log(p)-log(q))',
            'a + b',
            'loss = a',
            '0',
            'kl'
        )(p, q)
        return loss / xp.float32(p.shape[0]),

    # backward only q-side
    def backward_gpu(self, inputs, grads):
        p, q = inputs
        dq = -xp.float32(1.0) * p / (xp.float32(1e-8) + q) / xp.float32(
            p.shape[0]) * grads[0]
        return xp.zeros_like(p), dq



def kl(p, q):
    return KL_multinominal()(F.softmax(p), F.softmax(q))


def kl(p, q):
    return KL_multinominal()(p, q)


def distance(y0, y1):
    return kl(F.softmax(y0), F.softmax(y1))


def vat(model,distance, x, xi=10, eps=1.0, Ip=1):
    #print("x:{0}".format(x))
    #xp = cuda.cupy
    #print("model:{0}".format(model))
    #y = model.preditctor(x)
    #print("x.shape:{0}".format(x.shape))
    y = model(x)
    #print("y.shape:{0}".format(y.shape))
    y.unchain_backward()
    #print(type(y))
    #print(y.data)
    # calc adversarial direction
    #print(xp)
    d = xp.random.normal(size=x.shape, dtype=np.float32)
    #d = xp.random.normal(size=x.shape)
    sum_axis = [i for i in range(d.ndim) if i]
    #print(sum_axis)
    shape = (x.shape[0],) + (1,) * (d.ndim - 1)
    d = d / xp.sqrt(xp.sum(d ** 2, axis=sum_axis)).reshape((shape))
    #d = d / xp.sqrt(xp.sum(d ** 2,axis=1)).reshape((shape))
    for ip in range(Ip):
        #d_var = Variable(d.astype(numpy.float32))
        d_var = chainer.Variable(d.astype(xp.float32))
        #y2 = model.predictor(x + xi * d_var)
        y2 = model(x + xi * d_var)
        kl_loss = distance(y, y2)
        kl_loss.backward()
        d = d_var.grad
        d = d / xp.sqrt(xp.sum(d ** 2, axis=sum_axis)).reshape((shape))
    #d_var = Variable(d.astype(numpy.float32))
    d_var = chainer.Variable(d.astype(xp.float32))

    # calc regularization
    #y2 = model.predictor(x + eps * d_var)
    y2 = model(x + eps * d_var)
    return distance(y, y2)

class VATLossClassifier(chainer.Chain):
    def __init__(self, predictor,
                 lossfun=None,
                 accfun=None,
                 label_key=-1):
        if not (isinstance(label_key, (int, str))):
            raise TypeError('label_key must be int or str, but is %s' %
                            type(label_key))

        super(VATLossClassifier, self).__init__()
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None
        self.label_key = label_key

        with self.init_scope():
            self.predictor = predictor
    #def __call__(self, x, t=None):
    #def forward(self, *args, **kwargs):
    def forward(self, *args, **kwargs):
        #print("args-before:{0}".format(args))
        #print("args-before-len:{0}".format(len(args)))
        #print("self.label_key:{0}".format(self.label_key))
        #print(type(self.label_key))
        if len(args) == 1:
            args=args
            t = None
        elif isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = 'Label key %d is out of bounds' % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[:self.label_key] + args[self.label_key + 1:]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]
        #print("t:{0}".format(t))
        #print("len(t):{0}".format(len(t)))
        self.y = None
        self.loss = None
        self.accuracy = None
        #print("args-after:{0}".format(args))
        #print(args[0])
        #print(args[1])
        #print(kwargs)
        #print(self)
        if t is not None:
            #h = self.predict(x)
            #print("t is not None")
            h = self.predictor(*args, **kwargs)
            loss = F.softmax_cross_entropy(h, t)
            #print(h,t)
            #print(loss)
            chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
            return loss
        else:
            #print("t is None")
            #return vat(self, distance, x, self.eps)
            h = self.predictor(*args, **kwargs)
            loss = vat(self.predictor, distance, *args)
            #print(h)
            #print(loss)
            #chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
            chainer.report({'loss': loss}, self)
            return loss

class VATUpdater(training.updater.StandardUpdater):

    def __init__(self, iterator,iterator_ul, optimizer, converter=convert.concat_examples,
                 device=None, loss_func=None, loss_scale=None,
                 auto_new_epoch=True):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if not isinstance(optimizer, dict):
            optimizer = {'main': optimizer}
        self._optimizers = optimizer

        if isinstance(iterator_ul, iterator_module.Iterator):
            iterator_ul = {'main': iterator_ul}
        self._iterators_ul = iterator_ul

        if device is not None and device >= 0:
            for optimizer in six.itervalues(self._optimizers):
                optimizer.target.to_gpu(device)

        self.converter = converter
        self.loss_func = loss_func
        self.device = device
        self.iteration = 0

        self.loss_scale = loss_scale
        if loss_scale is not None:
            for optimizer in six.itervalues(self._optimizers):
                optimizer.set_loss_scale(loss_scale)

        self.auto_new_epoch = auto_new_epoch
        if auto_new_epoch:
            for o in six.itervalues(self._optimizers):
                o.use_auto_new_epoch = True

    def first_iter(self):
        self.f_val=True

    def update(self):
        #print("#1:{0}".format(self._iterators['main'].next()))
        '''
        if(self._iterators['main'].is_new_epoch is True or self.f_val is True):
            self.update_core()
            self.f_val=False
        else:
            self.update_vat()
        '''
            #1
        #print("#1:{0}".format(self._iterators['main'].current_position))
        #self.update_core()
        #print("#2:{0}".format(self._iterators['main'].current_position))
        #iterator=self._iterators['main']

        self.update_vat()
        #print("#3:{0}".format(self._iterators['main'].current_position))
        #print(self.iteration)
        #print(iterator.epoch)
        #print(iterator.epoch_detail)
        #print(iterator.previous_epoch_detail)
        self.iteration += 1
        #print(self._iterators['main'].is_new_epoch)


    def update_vat(self):
        iterator=self._iterators['main']
        batch = iterator.next()

        iterator_ul=self._iterators_ul['main']
        batch_ul = iterator_ul.next()
        #print(batch)
        in_arrays = self.converter(batch, self.device)
        in_arrays_ul = self.converter(batch_ul, self.device)
        #print(in_arrays)
        optimizer = self._optimizers['main']
        loss_func = optimizer.target
        #print(in_arrays[0] )
        #optimizer.zero_grads()
        #optimizer.cleargrad()
        #optimizer.update(loss_func, in_arrays[0])


        #print(np.sum(in_arrays[0][0]))
        #print(in_arrays_ul[0].shape)

        if isinstance(in_arrays, tuple):
            #optimizer.update(loss_func, *in_arrays)
            loss_l=loss_func(*in_arrays)
        elif isinstance(in_arrays, dict):
            #optimizer.update(loss_func, **in_arrays)
            loss_l=loss_func(**in_arrays)
        else:
            #optimizer.update(loss_func, in_arrays)
            loss_l=loss_func(in_arrays)

        #print(in_arrays_ul[0])
        #optimizer.update(loss_func, in_arrays_ul[0])
        loss_ul=loss_func(in_arrays_ul[0])
        loss_total=loss_l+loss_ul
        loss_func.cleargrads()
        loss_total.backward()
        #print(self.iteration)
        #print(optimizer.alpha)

        if(self.iteration >= 500 and self.iteration % 500 == 0):
            #print("alpha change")
            optimizer.alpha *= 0.9

        optimizer.update()

        #print(iterator.is_new_epoch)
        if self.auto_new_epoch and iterator.is_new_epoch:
            optimizer.new_epoch(auto=True)
        #optimizer.zero_grads()
        #optimizer.cleargrad()


def vat_loss_function(
        x, t, normalize=True, cache_score=True, class_weight=None,
        ignore_label=-1, reduce='mean', enable_double_backprop=False):
    '''
    if enable_double_backprop:
        return _double_backward_softmax_cross_entropy(
            x, t, normalize, class_weight, ignore_label, reduce)
    else:
        return SoftmaxCrossEntropy(
            normalize, cache_score, class_weight, ignore_label, reduce)(x, t)
    '''
    if t is not None:
        #h = self.predict(x)
        loss = F.softmax_cross_entropy(x, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(x, t)}, self)
        return loss
    else:
        #return vat(self, distance, x, self.eps)
        return vat(self,distance, x, 1.0)
