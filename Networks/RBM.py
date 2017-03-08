import theano
import theano.tensor as T
import numpy

class RBM(object):
    def __init__(self,
                 rng,
                 theanoRng,
                 input = None,
                 numVisible = 784,
                 numHidden = 500,
                 W = None,
                 hBias = None,
                 vBias = None):
        # Set parameters
        self.Rng = rng
        self.TheanoRng = theanoRng
        self.Input = input
        self.NumVisible = numVisible
        self.NumHidden = numHidden
        self.W = W
        self.HBias = hBias
        self.VBias = vBias

        self.createModel()

    def createModel(self):
        if self.W is None:
            wBound = 4 * numpy.sqrt(6. / (self.NumHidden + self.NumVisible))
            self.W = theano.shared(
                numpy.asarray(
                    self.Rng.uniform(
                        low = -wBound,
                        high = wBound,
                        size = (self.NumVisible, self.NumHidden)
                    ),
                    dtype = theano.config.floatX
                ),
                borrow = True
            )

        if self.HBias is None:
            self.HBias = theano.shared(
                numpy.zeros(
                    self.NumHidden,
                    dtype = theano.config.floatX
                ),
                borrow = True
            )

        if self.WBias is None:
            selfWBias = theano.shared(
                numpy.zeros(
                    self.NumVisible,
                    dtype = theano.config.floatX
                ),
                borrow = True
            )

        self.Params = [self.W, self.HBias, self.WBias]

    def FreeEnergy(self, vSample):
        wx_b = T.dot(vSample, self.W) + self.HBias
        vBiasTemp = T.dot(vSample, self.VBias)
        hiddenTerm = T.sum(T.log(1 + T.exp(wx_b)), axis = 1)
        return hiddenTerm - vBiasTemp

    def Propup(self, vis):
        preSigmoidActivation = T.dot(vis, self.W) + self.HBias
        return (preSigmoidActivation, T.nnet.sigmoid(preSigmoidActivation))

    def SampleHGivenV(self, v0Sample):
        preSigmoid_h1, h1_Mean = self.Propup(v0Sample)
        h1Sample = self.TheanoRng.binomial(
            size = h1_Mean.shape,
            n = 1, p = h1_Mean,
            dtype = theano.config.floatX
        )
        return (preSigmoid_h1, h1_Mean, h1Sample)

    def Propdown(self, hid):
        preSigmoidActivation = T.dot(hid, self.W.T) + self.VBias
        return (preSigmoidActivation, T.nnet.sigmoid(preSigmoidActivation))

    def SampleVGivenH(self, h0Sample):
        preSigmoidV1, v1_Mean = self.Propdown(h0Sample)
        v1Sample = self.TheanoRng.binomial(
            size = v1_Mean.shape,
            n = 1, p = v1_Mean,
            dtype = theano.config.floatX
        )
        return (preSigmoidV1, v1_Mean, v1Sample)

    def GibbsHvh(self, h0Sample):
        preSigmoidV1, v1Mean, v1Sample = self.SampleVGivenH(h0Sample)
        preSigmoidH1, h1Mean, h1Sample = self.SampleHGivenV(v1Sample)

        return [preSigmoidV1, v1Mean, v1Sample,
                preSigmoidH1, h1Mean, h1Sample]

    def GibbsVhv(self, v0Sample):
        preSigmoidH1, h1Mean, h1Sample = self.SampleHGivenV(v0Sample)
        preSigmoidV1, v1Mean, v1Sample = self.SampleVGivenH(h1Sample)
        return [preSigmoidH1, h1Mean, h1Sample,
                preSigmoidV1, v1Mean, v1Sample]

    def GetCostUpdates(self,
                       learningRate = 0.1,
                       persistent = None,
                       k = 1):
        preSigmoidPh, phMean, phSample = self.SampleHGivenV(self.Input)

        if persistent is None:
            chainStart = phSample
        else:
            chainStart = persistent

        (
            [
                preSigmoidNvs,
                nvMeans,
                nvSamples,
                preSigmoidNhs,
                nhMeans,
                nhSamples
            ],
            updates
        ) = theano.scan(
            self.GibbsHvh,
            outputs_info = [None, None, None, None, None, chainStart],
            n_steps = k
        )
        chainEnd = nvSamples[-1]

        cost = T.mean(self.FreeEnergy(self.Input)) - T.mean(self.FreeEnergy(chainEnd))

        gParams = T.grad(cost, self.params, consider_constant = [chainEnd])

        for gParam, param in zip(gParams, self.Params):
            updates[param] = param - gParam * T.cast(
                learningRate,
                dtype = theano.config.floatX
            )

        if persistent:
            updates[persistent] = nhSamples[-1]
            monitoringCost = self.GetLikelihoodCost(updates)
        else:
            monitoringCost = self.GetReconstructionCost(updates,
                                                        preSigmoidNvs[-1])

        return monitoringCost, updates

    def GetLikelihoodCost(self, updates):
        bitIIdx = theano.shared(value = 0)
        xi = T.round(self.Input)
        feXi = self.FreeEnergy(xi)

        xiFlip = T.set_subtensor(xi[:, bitIIdx], 1 - xi[:, bitIIdx])

        feXiFlip = self.FreeEnergy(xiFlip)

        cost = T.mean(self.NumVisible * T.log(T.nnet.sigmoid(feXiFlip - feXi)))

        updates[bitIIdx] = (bitIIdx + 1)  % self.NumVisible

        return cost

    def GetReconstructionCost(self, updates, preSigmoidNv):
        crossEntropy = T.mean(
            T.sum(
                self.Input * T.log(T.nnet.sigmoid(preSigmoidNv)) +
                (1 - self.Input) * T.log(1 - T.nnet.sigmoid(preSigmoidNv)),
                axis = 1
            )
        )

        return crossEntropy
