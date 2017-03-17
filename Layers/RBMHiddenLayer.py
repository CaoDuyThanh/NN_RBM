import theano
import theano.tensor as T
import numpy

class RBMHiddenLayer():
    def __init__(self,
                 rng,
                 theanoRng,
                 input = None,
                 numVisible = 784,
                 numHidden = 500,
                 activation = T.nnet.sigmoid,
                 W = None,
                 h = None,
                 v = None):
        self.Rng = rng
        self.TheanoRng = theanoRng
        self.Input = input
        self.NumVisible = numVisible
        self.NumHidden = numHidden
        self.Activation = activation
        self.W = W
        self.h = h
        self.v = v

        self.createModel()

    def createModel(self):
        if self.W is None:
            wBound = numpy.sqrt(6.0 / (self.NumVisible + self.NumHidden))
            self.W = theano.shared(
                numpy.asarray(
                    self.Rng.uniform(
                        low  = -wBound,
                        high =  wBound,
                        size = (self.NumVisible, self.NumHidden)
                    ),
                    dtype = theano.config.floatX
                ),
                borrow = True
            )

        if self.h is None:
            hBound = numpy.sqrt(6.0 / self.NumHidden)
            self.h = theano.shared(
                numpy.asarray(
                    self.Rng.uniform(
                        low  = -hBound,
                        high =  hBound,
                        size = (self.NumHidden,)
                    ),
                    dtype = theano.config.floatX
                ),
                borrow = True
            )

        if self.v is None:
            vBound = numpy.sqrt(6.0 / self.NumVisible)
            self.v = theano.shared(
                numpy.asarray(
                    self.Rng.uniform(
                        low  = -vBound,
                        high =  vBound,
                        size = (self.NumVisible)
                    ),
                    dtype = theano.config.floatX
                ),
                borrow = True
            )

        self.Params = [self.W, self.h, self.v]
        self.Cost, self.Update = self.getCostUpdate()

    def propup(self, visible):
        preActivation = T.dot(visible, self.W) + self.h
        afterActivation = self.Activation(preActivation)
        return [preActivation, afterActivation]

    def sampleHgivenV(self, input):
        preActivationH1, h1Mean = self.propup(input)
        h1Sample = self.TheanoRng.binomial(size = h1Mean.shape,
                                           n = 1, p = h1Mean,
                                           dtype = theano.config.floatX)
        return [preActivationH1, h1Mean, h1Sample]

    def propdown(self, hidden):
        preActivation = T.dot(hidden, self.W.T) + self.v
        afterActivation = self.Activation(preActivation)
        return [preActivation, afterActivation]

    def sampleVgivenH(self, input):
        preActivationV1, v1Mean = self.propdown(input)
        v1Sample = self.TheanoRng.binomial(size = v1Mean.shape,
                                           n = 1, p = v1Mean,
                                           dtype = theano.config.floatX)
        return [preActivationV1, v1Mean, v1Sample]

    def freeEnergy(self, vSample):
        wxB = T.dot(vSample, self.W) + self.h
        vTemp = T.dot(vSample, self.v)
        hTemp = T.dot(T.log(1 + T.exp(wxB)), axis = 1)
        return -hTemp - vTemp

    def getCostUpdate(self,
                      learningRate,
                      persistent = None,
                      k = 1):
        preActivationph, phMean, phSample = self.sampleHgivenV(self.Input)

        if persistent is None:
            chainStart = phSample
        else:
            chainStart = persistent
        (
            [
                preActivationNvs,
                nvMeans,
                nvSamples,
                preActivationNhs,
                nhMeans,
                nhSamples
            ]
        ) = theano.scan(
            self.gibbsHvh,
            outputs_info = [None, None, None, None, None, chainStart],
            n_steps = k
        )
        chainEnd = nvSamples[-1]

        cost = T.mean(self.freeEnergy(self.Input)) - T.mean(self.freeEnergy(chainEnd))
        grads = T.grad(cost, self.Params, consider_constant=[chainEnd])
        updates = [(param, param - learningRate * grad)
                   for param, grad in zip(self.Params, grads)]

        if persistent is not None:
            updates[persistent] = self.getPseudoLikelihoodCost(updates)
        else:
            monitoringCost = self.getReconstructionCost(preActivationNvs[-1])

        return [monitoringCost, updates]

    def getPseudoLikelihoodCost(self, updates):
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.freeEnergy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.freeEnergy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.numVisible * T.log(self.Activation(fe_xi_flip - fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.NumVisible

        return cost

    def getReconstructioinCost(self, preActivationNv):
        crossEntropy = T.mean(
            T.sum(
                self.Input * T.log(self.Activation(preActivationNv)) + (1 - self.Input) * T.log(1 - self.Activation(preActivationNv)),
                axis = 1
            )
        )
        return crossEntropy