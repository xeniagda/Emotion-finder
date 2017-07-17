import csv
import numpy as np
import lasagne
import theano
import theano.tensor as T

LEARNING_RATE = 0.01

BATCH_SIZE = 200

VERBOSE = True

if VERBOSE:
    debug = print
else:
    debug = lambda *args, **kwargs: 0

# tweets contains tuples of (id, emotion, user, message)
tweets = []

try:
    debug("Reading tweets...")
    with open("Tweets.csv", "r") as f:
        f.readline()  # Skip the first line, only contains meta-information.

        tweetreader = csv.reader(f)
        for row in tweetreader:
            tweets.append(
                    ( int(row[0])  # Tweet ID
                    , row[1]       # Emotion
                    , row[2]       # Username
                    , row[3]       # Message
                    )
            )
    debug("Done reading tweets")
except FileNotFoundError:
    print("File 'Tweets.cvs' not found. Please download it from https://www.crowdflower.com/wp-content/uploads/2016/07/text_emotion.csv")


emotions = set(tweet[1] for tweet in tweets)  # Collect all emotions

# Indexing from neural network output to emotion and vice versa
ix_to_em = sorted(emotions)  # Sort to be consistent between runs since the set is unsorted
em_to_ix = {em: i for i, em in enumerate(ix_to_em)}

chars = set(ch for tweet in tweets for ch in tweet[3])  # Collect all chars
vocab_size = len(chars)

# Same as em_to_ix & co
ix_to_ch = sorted(chars)
ch_to_ix = {ch: i for i, ch in enumerate(ix_to_ch)}


# The length of all the messages must be the same between all runs, we take the 99th precentile of the lengths of the tweets. Should be ~140
message_lengths = sorted([len(tweet[3]) for tweet in tweets])
TRAIN_LENGTH = message_lengths[int(-len(message_lengths) * 0.01)]


# Take a tweet and give back a training data for the network in the form of a
# one-hot array of chars, and the correct emotion as an ix in a tuple
def tweet_to_data(tweet):  
    text = tweet[3]
    emotion_ix = em_to_ix[tweet[1]]

    text_as_array = np.zeros((len(text), vocab_size))
    for i, ch in enumerate(text):
        text_as_array[i, ch_to_ix[ch]] = 1

    while text_as_array.shape[0] < TRAIN_LENGTH:
        text_as_array = np.concatenate((np.zeros((1, vocab_size)), text_as_array))
    if text_as_array.shape[0] > TRAIN_LENGTH:
        text_as_array = text_as_array[0:TRAIN_LENGTH]
    return (text_as_array, emotion_ix)

class EmotionNetwork:
    def __init__(self, vocab_size, emotion_size, n_hidden, grad_clip, training=True):

        debug("Building network")

        # Input layer. Input is in shape (batch size, text length, chars)
        l_in = lasagne.layers.InputLayer(shape=(None, None, vocab_size))

        # Lstm layer with tanh nonlinearity.
        l_lstm = lasagne.layers.LSTMLayer(
                l_in, n_hidden, grad_clipping=grad_clip,
                nonlinearity=lasagne.nonlinearities.tanh,
                only_return_final=True
                )
        
        # Output layer in form a dense layer, with a one-hot representation of the emotion in the tweet as the result. Output shape is (batch size, emotion)
        self.l_out = lasagne.layers.DenseLayer(
                l_lstm, num_units=emotion_size,
                W=lasagne.init.Normal(),
                nonlinearity=lasagne.nonlinearities.softmax
                )

        # Theano tensor for the target
        target_values = T.ivector(str(id(self)) + "_target_output")
        
        net_output = lasagne.layers.get_output(self.l_out)

        cost = lasagne.objectives.categorical_crossentropy(net_output, target_values)
        cost = cost.mean()

        all_params = lasagne.layers.get_all_params(self.l_out, trainable=True)

        debug("Computing updates")

        updates = lasagne.updates.nesterov_momentum(cost, all_params, LEARNING_RATE, momentum=0.4)

        debug("Compiling functions, cost", end="", flush=True)

        self.cost = theano.function([l_in.input_var, target_values], cost, allow_input_downcast=True)

        if training:
            debug(" done, train", end="", flush=True)

            self.train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)

        debug(" done, eval", end="", flush=True)

        self.eval = theano.function([l_in.input_var], net_output, allow_input_downcast=True)
        
        debug(" done")

        debug("All done!")

    def save(self, filename):
        values = lasagne.layers.get_all_param_values(self.l_out)
        np.savez(filename, *values)

    def load(self, filename):
        debug("Loading from save...")
        try:
            with np.load(filename) as f:
                param_values = [f["arr_%d" % i] for i in range(len(f.files))]
        except Exception as e:
            print(repr(e))
            print("Couldn't read input file!")
            cont = input("Continue anyway? [Y/n] ").lower()
            if cont == "n":
                sys.exit()
            param_values = None
        if param_values != None:
            lasagne.layers.set_all_param_values(self.l_out, param_values)

if __name__ == "__main__":
    network = EmotionNetwork(vocab_size, len(emotions), 100, 100)
    network.load("Emotion.npz")

    epoch = 0
    while True:
        p = 0
        while p < len(tweets) - 1:
            batch = tweets[p:p+BATCH_SIZE]
            batch = map(tweet_to_data, batch)
            batch = list(zip(*batch))
            inputs = np.array(batch[0], dtype=float)
            targets = np.array(batch[1], dtype=int)

            cost = network.train(inputs, targets)

            p += BATCH_SIZE

            print("Epoch %s, cost = %s." % (p / len(tweets) + epoch, cost))

        print("Epoch %s done. Saving." % epoch)
        network.save("Emotion.npz")
        epoch += 1