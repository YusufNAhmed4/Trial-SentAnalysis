'''
SVC model
'''

# pylint: disable=invalid-name
import json
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC
import joblib
import modelhelpers


def train_svc_model():
    """
    Trains a Support Vector Classifier model.
    """

    with open("output2.jsonl", "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file if line.strip()]


    train, test = modelhelpers.train_test_temporal(data)
    train = modelhelpers.make_pairs(train)
    test = modelhelpers.make_pairs(test)

    justice_vocab, results_vocab = modelhelpers.make_vocabs(train)

    train = modelhelpers.prep_justice_result(
        train,
        justice_vocab,
        results_vocab
    )

    test = modelhelpers.prep_justice_result(
        test,
        justice_vocab,
        results_vocab
    )

    # Text features
    train_titles = [x.get("name") or "" for x, _ in train]
    train_excerpts = [x.get("excerpt") or "" for x, _ in train]

    test_titles = [x.get("name") or "" for x, _ in test]
    test_excerpts = [x.get("excerpt") or "" for x, _ in test]

    title_vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2
    )

    excerpt_vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 3),
        min_df=2,
        stop_words="english"
    )

    X_train_title = title_vectorizer.fit_transform(train_titles)
    X_train_excerpt = excerpt_vectorizer.fit_transform(train_excerpts)

    X_test_title = title_vectorizer.transform(test_titles)
    X_test_excerpt = excerpt_vectorizer.transform(test_excerpts)

    # Year feature
    year_train = np.array(
        [x["year"] for x, _ in train],
        dtype=np.float32
    ).reshape(-1, 1)

    year_test = np.array(
        [x["year"] for x, _ in test],
        dtype=np.float32
    ).reshape(-1, 1)

    year_mean = year_train.mean()
    year_std = year_train.std()

    if year_std == 0:
        year_std = 1.0

    year_train = (year_train - year_mean) / year_std
    year_test = (year_test - year_mean) / year_std

    # Justice multi-hot feature
    justice_train = csr_matrix(
        np.array([x["justices"] for x, _ in train], dtype=np.float32)
    )

    justice_test = csr_matrix(
        np.array([x["justices"] for x, _ in test], dtype=np.float32)
    )

    # Combine all features
    X_train = hstack([
        X_train_title,
        X_train_excerpt,
        csr_matrix(year_train),
        justice_train
    ])

    X_test = hstack([
        X_test_title,
        X_test_excerpt,
        csr_matrix(year_test),
        justice_test
    ])

    # Labels
    y_train = np.array([y for _, y in train])
    y_test = np.array([y for _, y in test])

    c = 1.0
    clf = LinearSVC(
        class_weight={
            0: 1.0,
            1: 1.0,
            2: 2.0
        },
        C=c,
        max_iter=10000
    )

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    print("Results vocab:")
    print(results_vocab)

    print("Predictions:")
    print(np.unique(preds, return_counts=True))

    print("Truth:")
    print(np.unique(y_test, return_counts=True))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    print("Classification report:")
    print(classification_report(y_test, preds, zero_division=0))

    feature_names = np.concatenate([
        title_vectorizer.get_feature_names_out(),
        excerpt_vectorizer.get_feature_names_out(),
        ["year"],
        ["justices"]
    ])

    for i, class_name in enumerate(results_vocab):
        top = np.argsort(clf.coef_[i])[-20:]

        print(class_name)
        print(feature_names[top])

    model_bundle = {
        "clf": clf,
        "title_vectorizer": title_vectorizer,
        "excerpt_vectorizer": excerpt_vectorizer,
        "justice_vocab": justice_vocab,
        "results_vocab": results_vocab,
        "year_mean": year_mean,
        "year_std": year_std,
    }

    joblib.dump(model_bundle, "scotus_svc_model.joblib")

    return clf, title_vectorizer, excerpt_vectorizer, results_vocab


if __name__ == "__main__":
    train_svc_model()
