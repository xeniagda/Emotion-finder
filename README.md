
# Emotion-finder

This is a neural network which tries to guess what emotion a message displaying


It works by using [this](https://www.crowdflower.com/wp-content/uploads/2016/07/text_emotion.csv) dataset of 40000 tweets. The file is contained in this repository.

The file `Emotion.py` is a module with the neural network which. When run it trains the network and saves the resulting network to `Emotion.npz`. In this repository the `Emotion.npz` file is already trained.

The file `Show.py` displays the progress of the neural network by showing random tweets from the dataset together with the guessed and actual emotion. There's also the `Try.py` file which allows you to enter your own text and have it classified

## Running

To use this neural network you will need the following:

* Python 3
* numpy (for arrays)
* theano (for GPU processing)
* lasagne (for the actual network code)
* pygame (for the `Show.py` code)

You can install all of the modules using the package manager `pip`. Note that lasagne must use the bleeding-edge version for everything to work.