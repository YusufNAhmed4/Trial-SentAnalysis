'''
Helper functions for the model
'''

# pylint: disable=no-member
import random
import tensorflow as tf
import numpy as np
import sklearn

def train_test(data, train_ratio=0.8, seed=1832) :
    """
    Splits long json into train-test (80-20)
    """
    data_copy = data.copy()

    random.seed(seed)
    random.shuffle(data_copy)

    print("Amt of data: ", len(data_copy))
    results = [x["result"] for x in data_copy]

    print("Reversed count: ", results.count("reversed"))
    print("affirmed count: ", results.count("affirmed"))
    print("vacated count: ", results.count("vacated"))
    split_idx = int(len(data_copy) * train_ratio)

    train_data = data_copy[:split_idx]
    test_data = data_copy[split_idx:]

    return train_data, test_data

def make_pairs(data):
    """
    Takes a JSON list and makes a list of (input, label) tuples 
    """
    # print(type(data))
    # print((data[0]))
    pairs = []

    for entry in data:
        year = entry["year"]
        justices = entry["justices"]
        name = entry["title"]
        excerpt = entry["excerpt"]
        # print(type(justices))
        x = {
            "year": year,
            "justices": justices,
            "name": name,
            "excerpt": excerpt,
        }

        y = entry["result"]

        pairs.append((x, y))

    return pairs

def make_vocabs(data) :
    """
    Makes sets of unique results and justices
    """
    justices = []
    results = []

    for entry in data :
        results.append(entry[1])
        for justice in entry[0]["justices"] :
            justices.append(justice)
        # break

    justice_vocab = set(justices)
    results_vocab = set(results)
    return justice_vocab, results_vocab

def scale_years(data, train) :
    """
    Takes years and normalizes them between 0-1
    """
    years = [int(x["year"]) for x, _ in train]
    min_year = min(years)
    max_year = max(years)
    for entry, _ in data :
        year = int(entry["year"])
        entry["year"] =  (year - min_year) / (max_year - min_year)


def prep_justice_result(pairs, justice_vocab, results_vocab):
    """
    Turns justices/results into multi-hot numerical encodings
    """
    justice_to_id = {justice: i for i, justice in enumerate(justice_vocab)}
    result_to_id = {result: i for i, result in enumerate(results_vocab)}

    encoded = []

    for x, y in pairs:
        justice_vector = [0] * len(justice_vocab)

        for justice in x["justices"]:
            if justice in justice_to_id:
                justice_vector[justice_to_id[justice]] = 1

        encoded_x = {
            "year": x["year"],
            "justices": justice_vector,
            "name": x["name"],
            "excerpt": x["excerpt"],
        }

        encoded_y = result_to_id[y]

        encoded.append((encoded_x, encoded_y))

    return encoded

def make_text_vectorizer(train, field, max_tokens=20000, output_length=200):
    """
    Fits a TextVectorization layer on one text field.
    train is a list of (input, label) tuples.
    """

    texts = [x.get(field, "") for x, _ in train]

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=output_length
    )

    vectorizer.adapt(texts)

    return vectorizer

def tokenize_field(data, vectorizer, field):
    """
    Takes in a text field and tokenizes it using the inputted vectorizer
    """
    for x, _ in data:
        x[field + "_tokens"] = (
            vectorizer([x[field]])
            .numpy()[0]
        )

def make_x_y(data):
    """
    Turns (input, label) list into X and y, both are np arrays
    """
    x = {
        "justices": np.array([x["justices"] for x, _ in data]),
        "year": np.array([x["year"] for x, _ in data], dtype=np.float32),
        "name_tokens": np.array([x["name_tokens"] for x, _ in data]),
        "excerpt_tokens": np.array([x["excerpt_tokens"] for x, _ in data]),
    }

    y = np.array([label for _, label in data])

    return x, y


def create_inputs(x_train):
    return {
        "justices": tf.keras.Input(
            shape=(x_train["justices"].shape[1],),
            name="justices"
        ),
        "year": tf.keras.Input(shape=(), name="year"),
        "name_tokens": tf.keras.Input(
            shape=(x_train["name_tokens"].shape[1],),
            name="name_tokens"
        ),
        "excerpt_tokens": tf.keras.Input(
            shape=(x_train["excerpt_tokens"].shape[1],),
            name="excerpt_tokens"
        )
    }


def make_text_features(input_layer, embed_size):
    embed = tf.keras.layers.Embedding(
        input_dim=20000,
        output_dim=embed_size,
        mask_zero=False
    )(input_layer)

    pooled = tf.keras.layers.GlobalAveragePooling1D()(embed)
    return tf.keras.layers.Dropout(0.25)(pooled)


def make_other_features(inputs):
    justice_features = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(inputs["justices"])

    justice_features = tf.keras.layers.Dropout(0.25)(justice_features)

    year_features = tf.keras.layers.Reshape((1,))(inputs["year"])

    return justice_features, year_features


def make_classifier(features, num_results):
    x = tf.keras.layers.Concatenate()(features)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    return tf.keras.layers.Dense(
        num_results,
        activation="softmax"
    )(x)


def build_model(x_train, num_results):
    inputs = create_inputs(x_train)

    name_features = make_text_features(inputs["name_tokens"], 32)
    excerpt_features = make_text_features(inputs["excerpt_tokens"], 64)
    justice_features, year_features = make_other_features(inputs)

    output = make_classifier(
        [
            justice_features,
            year_features,
            name_features,
            excerpt_features
        ],
        num_results
    )

    model = tf.keras.Model(inputs=inputs, outputs=output)

    model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4,
        clipnorm=1.0
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

    return model


def make_early_stopping():
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

def oversample_classes(x_train, y_train):
    """
    Should make label counts equal
    """
    classes, counts = np.unique(y_train, return_counts=True)
    max_count = counts.max()

    x_bal = {key: [] for key in x_train}
    y_bal = []

    for c in classes:
        idx = np.where(y_train == c)[0]

        sampled_idx = sklearn.utils.resample(
            idx,
            replace=True,
            n_samples=max_count,
            random_state=42
        )

        # Sample each feature array
        for key in x_train:
            x_bal[key].append(x_train[key][sampled_idx])

        y_bal.append(y_train[sampled_idx])

    # Merge class chunks together
    for key in x_bal:
        x_bal[key] = np.concatenate(x_bal[key])

    y_bal = np.concatenate(y_bal)

    # Shuffle everything together
    perm = np.random.permutation(len(y_bal))

    for key in x_bal:
        x_bal[key] = x_bal[key][perm]

    y_bal = y_bal[perm]

    return x_bal, y_bal
