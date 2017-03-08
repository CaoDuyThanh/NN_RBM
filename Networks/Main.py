import Utils.DataHelper as DataHelper
from RBM import *
from PIL import Image
from Utils.FilterHelper import *
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# Hyper parameters
DATASET_NAME = '../Dataset/mnist.pkl.gz'
LEARNING_RATE = 0.1
NUM_EPOCHS = 15
BATCH_SIZE = 20

# Networks
NUM_CHAINS = 20
NUM_SAMPLES = 10
NUM_HIDDEN = 500

if __name__ == '__main__':
    def RBM():
        # Load dataset from local disk or download from internet
        datasets = DataHelper.LoadData(DATASET_NAME)
        trainSetX, trainSetY = datasets[0]
        testSetX, testSetY = datasets[2]

        nTrains = trainSetX.get_value(borrow = True).shape[0]
        nTests = testSetX.get_value(borrow = True).shape[0]

        nTrainBatchs = nTrains // BATCH_SIZE
        nTestBatchs  = nTests // BATCH_SIZE

        # Create model
        '''
        MODEL ARCHITECTURE
        INPUT     ->    Convolution      ->        Dropout
        (32x32)        (6, 1, 5, 5)              (6, 14, 14)
                  ->    Convolution      ->        Dropout
                       (16, 6, 5, 5)             (16, 5, 5)
                  ->    Hidden layer
                        (120 neurons)
                  ->    Hidden layer
                        (84 neurons)
                  ->    Output layer (Softmax)
                        (10 neurons)
        '''
        # Create random state
        rng = numpy.random.RandomState(12345)
        theanoRng = RandomStreams(rng.randint(2 ** 30))

        # Create shared variable for input
        Index = T.lscalar('Index')
        X = T.matrix('X')

        # Create storage for the persistance chain
        persistanceChain = theano.shared(
            numpy.zeros(
                (BATCH_SIZE, NUM_HIDDEN),
                dtype = theano.config.floatX
            ),
            borrow = True
        )

        # Construct the RBM class
        rbm = RBM(
            rng = rng,
            theanoRng = theanoRng,
            input = X,
            numVisible = 28 * 28,
            numHidden = NUM_HIDDEN
        )

        # Get the cost and the gradient corresponding to one step of CD-15
        cost, updates = rbm.GetCostUpdates(
            learningRate = LEARNING_RATE,
            persistent = persistanceChain,
            k = 15
        )

        #############################
        #     Training the RBM      #
        #############################
        trainRBM = theano.function(
            inputs = [Index],
            outputs = [cost],
            updates = updates,
            givens = {
                X: trainSetX[Index * BATCH_SIZE : (Index + 1) * BATCH_SIZE]
            }
        )

        plottingTime = 0

        # Training............
        if __name__ == '__main__':
            for epoch in range(NUM_EPOCHS):
                meanCost = []
                for batchIndex in range(nTrainBatchs):
                    meanCost += [trainRBM(batchIndex)]

                print('Training epoch = %d, cost is %f ' % (epoch, numpy.mean(meanCost)))

                image = Image.fromarray(
                    tile_raster_images(
                        X = rbm.W.get_value(borrow = True).T,
                        img_shape = (28, 28),
                        tile_shape = (10, 10),
                        tile_spacing = (1, 1)
                    )
                )
                image.save('filters_at_epoch_%i.png' % (epoch))

        # Testing...............
        # Pick random test examples
        testIdx = rng.randint(nTests - NUM_CHAINS)
        persistentVisChain = theano.shared(
            numpy.asarray(
                testSetX.get_value(borrow = True)[testIdx : testIdx + NUM_CHAINS],
                dtype = theano.config.floatX
            )
        )

        plotEvery = 1000
        (
            [
                presigHigs,
                hidMfs,
                hidSamples,
                presigVis,
                visMfs,
                visSamples
            ],
            updates
        ) = theano.scan(
            rbm.GibbsVhv,
            outputs_info = [None, None, None, None, None, persistentVisChain],
            n_steps = plotEvery
        )

        updates.update({persistentVisChain: visSamples[-1]})

        sampleFn = theano.function(
            [],
            [
                visMfs[-1],
                visSamples[-1]
            ],
            updates = updates
        )

        imageData = numpy.zeros(
            (29 * NUM_SAMPLES + 1, 29 * NUM_CHAINS - 1),
            dtype = 'uint8'
        )
        for idx in range(NUM_SAMPLES):
            visMf, visSample = sampleFn()
            print('... plotting sample %d' %idx)
            imageData[29 * idx : 29 * idx + 28, :] = tile_raster_images(
                X = visMf,
                img_shape = (28, 28),
                tile_shape = (1, NUM_CHAINS),
                tile_spacing = (1, 1)
            )

        image = Image.fromarray(imageData)
        image.save('sample.png')


if __name__ == '__main__':
    RBM()