import random

from Emotion import *

network = EmotionNetwork(vocab_size, len(emotions), 100, 100, training=False)
network.load("Emotion.npz")

while True:
    tweet = random.choice(tweets)

    (t, e) = tweet_to_data(tweet)
    res = network.eval([t])[0]
    cost = network.cost([t], [e])

    guesses = np.argsort(-res)

    print()
    print("Message: %s" % tweet[3])  # Message
    for i in range(4):
        print("Guess %s: %s (%s)" % (i+1, ix_to_em[guesses[i]], res[guesses[i]]))
    print()
    print("Actual emotion: %s" % ix_to_em[e])
    if np.argmax(res) == e:
        print("✅")
    else:
        print("❌")
    print("Cost: %s" % str(cost))


    input()