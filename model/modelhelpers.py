'''
Helper functions for the model
'''

import random


def train_test(data, train_ratio=1, seed=1832) :
    """
    Splits long json into train-test (80-20)
    """
    data_copy = data.copy()

    random.seed(seed)
    random.shuffle(data_copy)

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

def scale_years(data) :
    """
    Takes years and normalizes them between 0-1
    """
    years = [int(x["year"]) for x, _ in data]
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
