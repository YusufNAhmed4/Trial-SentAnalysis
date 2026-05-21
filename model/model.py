'''
This is the model which uses the scraper's data to generate an ML model predicting SC opinions.
'''

import json
# import tensorflow as tf
# from tf import keras
# from tf.keras.models import Sequential
# from tf.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout
import modelhelpers

def train_model() :
    """
    Trains ML model
    """
    with open('output.json', 'r', encoding="utf-8") as file:
        data = json.load(file)
    train, test = modelhelpers.train_test(data)
    train = modelhelpers.make_pairs(train)
    test = modelhelpers.make_pairs(test)

    justice_vocab, results_vocab = modelhelpers.make_vocabs(train)

    train = modelhelpers.prep_justice_result(train, justice_vocab, results_vocab)
    test = modelhelpers.prep_justice_result(test, justice_vocab, results_vocab)
    modelhelpers.scale_years(train)
    print(train[:1])



if __name__ == "__main__":
    train_model()
