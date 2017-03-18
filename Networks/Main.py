import Utils.DataHelper as DataHelper
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from PIL import Image
from Utils.FilterHelper import *

# Import layers
from Layers.RBMHiddenLayer import *


# Hyper parameters
DATASET_NAME = '../Dataset/mnist.pkl.gz'
LEARNING_RATE = 0.1
NUM_EPOCHS = 15
BATCH_SIZE = 20

# NETWORKS HYPER PARAMETERS
NUM_CHAINS = 20
NUM_SAMPLES = 10
NUM_HIDDEN = 500

def RBM():
    #########################################
    #      LOAD DATASET                     #
    #########################################
    # Load dataset from local disk or download from internet
    datasets = DataHelper.LoadData(DATASET_NAME)
    trainSetX = datasets[0][0]
    nTrains = trainSetX.get_value(borrow = True).shape[0]
    nTrainBatchs = nTrains // BATCH_SIZE

    #########################################
    #      CREATE MODEL                     #
    #########################################
    '''
    MODEL ARCHITECTURE
       VISIBLE LAYER            ->         HIDDEN LAYER
    28 X 28 = 576 neurons                   500 neurons
    '''
    # Create random state
    rng = numpy.random.RandomState(12345)
    theanoRng = RandomStreams(rng.randint(2 ** 30))

    # Create shared variable for input
    Index = T.lscalar('Index')
    LearningRate = T.scalar('LearningRate', dtype = 'float32')
    X = T.matrix('X')
    X = X.reshape((BATCH_SIZE, 28 * 28))

    # Construct the RBM class with one hidden layer
    persistentChain = theano.shared(
                            numpy.zeros((BATCH_SIZE, NUM_HIDDEN), dtype=theano.config.floatX),
                            borrow=True
                    )
    rbm = RBMHiddenLayer(
        rng          = rng,
        theanoRng    = theanoRng,
        input        = X,
        numVisible   = 28 * 28,
        numHidden    = NUM_HIDDEN,
        learningRate = LearningRate,
        persistent   = persistentChain,
        kGibbsSample = 15
    )

    #########################################
    #      CREATE TRAIN FUNCTION            #
    #########################################
    # Cost function
    trainFunc = theano.function(
        inputs  = [Index, LearningRate],
        outputs = [rbm.MonitoringCost],
        updates = rbm.Updates,
        givens  = {
            X: trainSetX[Index * BATCH_SIZE : (Index + 1) * BATCH_SIZE]
        }
    )

    #########################################
    #      TRAIN THE MODEL                  #
    #########################################
    for epoch in range(NUM_EPOCHS):
        meanCost = []
        for batchIndex in range(nTrainBatchs):
            cost = trainFunc(batchIndex, LEARNING_RATE)
            meanCost.append(cost)

        print ('Epoch = %d, cost = %f' % (epoch, numpy.mean(meanCost)))

        # Construct and save image of filter
        image = Image.fromarray(
            tile_raster_images(
                X            =  rbm.W.get_value(borrow=True).T,
                img_shape    = (28, 28),
                tile_shape   = (10, 10),
                tile_spacing = (1, 1)
            )
        )
        image.save('filters_at_epoch_%i.png' % (epoch))

if __name__ == '__main__':
    RBM()