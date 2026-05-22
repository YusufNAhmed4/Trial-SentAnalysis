'''
This is the model which uses the scraper's data to generate an ML model predicting SC opinions.
'''

import json
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
    print(results_vocab)

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

    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=10,
        batch_size=32
    )

    loss, accuracy = model.evaluate(x_test, y_test)

    print("Test loss:", loss)
    print("Test accuracy:", accuracy)
    model.save("sc_model.keras")




if __name__ == "__main__":
    train_model()
