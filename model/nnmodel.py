'''
This is the model which uses the scraper's data to generate an ML model predicting SC opinions.
'''
import sys
import json
import numpy as np
import modelhelpers
import sklearn


def train_nn_model(input_file) :
    """
    Trains ML NN model
    """
    with open(input_file, 'r', encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
    train, test = modelhelpers.train_test_temporal(data)
    train = modelhelpers.make_pairs(train)
    test = modelhelpers.make_pairs(test)

    justice_vocab, results_vocab = modelhelpers.make_vocabs(train)
    # print(results_vocab)

    train = modelhelpers.prep_justice_result(train, justice_vocab, results_vocab)
    test = modelhelpers.prep_justice_result(test, justice_vocab, results_vocab)
    modelhelpers.scale_years(train, train)
    modelhelpers.scale_years(test, train)
    # print(train[0])

    title_vectorizer = modelhelpers.make_text_vectorizer(train, "name", output_length=20)
    excerpt_vectorizer = modelhelpers.make_text_vectorizer(train, "excerpt", output_length=200)

    modelhelpers.tokenize_field(train, title_vectorizer, "name")
    modelhelpers.tokenize_field(test, title_vectorizer, "name")

    modelhelpers.tokenize_field(train, excerpt_vectorizer, "excerpt")
    modelhelpers.tokenize_field(test, excerpt_vectorizer, "excerpt")
    #print(train[0][0])

    x_train, y_train = modelhelpers.make_x_y(train)
    x_test, y_test = modelhelpers.make_x_y(test)

    model = modelhelpers.build_model(x_train, len(results_vocab))

    early_stop = modelhelpers.make_early_stopping()

    bad = np.all(x_train["excerpt_tokens"] == 0, axis=1)

    x_train = {
        key: value[~bad]
        for key, value in x_train.items()
    }
    y_train = y_train[~bad]
    # for key, value in x_train.items():
    #     print(key)
    #     print("dtype:", value.dtype)
    #     print("has nan:", np.isnan(value).any() if np.issubdtype(value.dtype, np.number) else "not numeric")
    #     print("has inf:", np.isinf(value).any() if np.issubdtype(value.dtype, np.number) else "not numeric")
    #     print()
    # print(np.unique(y_train, return_counts=True))
    # print(np.unique(y_test, return_counts=True))
    # print("y_train nan:", np.isnan(y_train).any())
    # print("y_test nan:", np.isnan(y_test).any())
    # print(x_train["name_tokens"].min(), x_train["name_tokens"].max())
    # print(x_train["excerpt_tokens"].min(), x_train["excerpt_tokens"].max())

    # for key in ["name_tokens", "excerpt_tokens"]:
    #     print(key)

    #     train_zero = np.all(x_train[key] == 0, axis=1)
    #     test_zero = np.all(x_test[key] == 0, axis=1)

    #     print("train all-zero:", train_zero.sum())
    #     print("test all-zero:", test_zero.sum())

    # x_train_bal, y_train_bal = modelhelpers.oversample_classes(x_train, y_train)

    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=10,
        batch_size=32,
        callbacks=[early_stop]
    )

    loss, accuracy = model.evaluate(x_test, y_test)

    preds = model.predict(x_test)
    print(np.unique(np.argmax(preds, axis=1), return_counts=True))
    preds = np.argmax(model.predict(x_test), axis=1)

    print("Predictions:")
    print(np.unique(preds, return_counts=True))

    print("Truth:")
    print(np.unique(y_test, return_counts=True))

    print("Test loss:", loss)
    print("Test accuracy:", accuracy)
    model.save("sc_model.keras")

    preds = np.argmax(model.predict(x_test), axis=1)

    print(results_vocab)
    print(np.unique(y_test, return_counts=True))
    print(sklearn.metrics.confusion_matrix(y_test, preds))
    print(sklearn.metrics.classification_report(y_test, preds))




if __name__ == "__main__":
    if len(sys.argv) != 2 :
        print("Usage: python nnmodel.py <input_data>")
        sys.exit(1)

    train_nn_model(sys.argv[1])
