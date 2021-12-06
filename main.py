import csv;
from copy import deepcopy
import struct

# Housekeeping
from typing import List, Dict

MAP_TO_THREE_DICT = {'0': '0',
                     '1': '0',
                     '2': '1',
                     '3': '2',
                     '4': '2'}


def read_and_store_tsv(fileName: str) -> List[List[str]]:
    tsv_rows = []
    tsv_file = open(fileName)
    read_tsv_file = csv.reader(tsv_file, delimiter="\t")
    for row in read_tsv_file:
        tsv_rows.append(row)
    tsv_rows.pop(0)
    return tsv_rows


def create_bag_of_words(rows: list, three_weight: bool) -> dict:
    bagOfWords = {}
    for row in rows:
        sentence = row[1]
        if three_weight:
            s_class = MAP_TO_THREE_DICT[row[2]]
        else:
            s_class = row[2]

        words = sentence.split(" ")
        for word in words:
            if word in bagOfWords:
                if s_class in bagOfWords[word]:
                    prevValue = bagOfWords[word][s_class]
                    bagOfWords[word][s_class] = (prevValue + 1)
                else:
                    bagOfWords[word].update({s_class: 1})
            else:
                dict = {word: {s_class: 1}}
                bagOfWords.update(dict)

    return bagOfWords


# Model Creating

def calculate_prior_probability(dataset_rows: list, three_weight: bool) -> Dict[int, float]:
    prior_probabilities = {}
    if three_weight:
        max_range = 3
    else:
        max_range = 5

    dataset_size = len(dataset_rows)
    for i in range(0, max_range):
        class_count = 0
        for row in dataset_rows:
            classification = row[2]
            if three_weight:
                if str(i) in MAP_TO_THREE_DICT[classification]:
                    class_count += 1
            else:
                if str(i) in classification:
                    class_count += 1
        prior_probabilities.update({i: (class_count / dataset_size)})
    return prior_probabilities


def laplace():
    pass


def calculate_likelihood(bag_of_words: dict, three_weight: bool) -> (dict,Dict[int, str]):
    local_bow = deepcopy(bag_of_words)
    if three_weight:
        max_range = 3
        # var below counts the occurrances of each class.
        class_counts = {"0": 0,
                        "1": 0,
                        "2": 0}
    else:
        max_range = 5
        class_counts = {"0": 0,
                        "1": 0,
                        "2": 0,
                        "3": 0,
                        "4": 0}
    # TODO: put this in another function
    for word in bag_of_words:
        associated_sentiments = bag_of_words[word]
        for sentiment_class in associated_sentiments:
            count = class_counts[sentiment_class] + associated_sentiments[sentiment_class]
            class_counts.update({sentiment_class: count})

    for word in local_bow:
        local_associated_sentiments = local_bow[word]
        for sentiment_class in local_associated_sentiments:
            temp_count = local_associated_sentiments[sentiment_class]
            class_total = class_counts[sentiment_class]
            # with laplace smoothing
            local_associated_sentiments[sentiment_class] = (temp_count + 1) / (class_total + len(local_bow))
            # without laplace smoothing
            # local_associated_sentiments[sentiment_class] = temp_count / class_total
    return local_bow, class_counts


def calculate_posterior_probability(prior_probabilities: dict, likelihoods: dict,
                                    class_counts: dict, rows: list, three_weight: bool) -> dict:
    classifications = {}
    for row in rows:
        sentence_id = row[0]
        sentence = row[1]
        sentence = sentence.split(" ")
        if three_weight:
            classes = {0: prior_probabilities[0],
                       1: prior_probabilities[1],
                       2: prior_probabilities[2]}
            max_range = 3
        else:
            classes = {
                0: prior_probabilities[0],
                1: prior_probabilities[1],
                2: prior_probabilities[2],
                3: prior_probabilities[3],
                4: prior_probabilities[4]
            }
            max_range = 5
        # calculation includes laplace smoothing if the word does not appear in the likelihoods or is there but has no
        # weight attached.
        for word in sentence:
            for i in range(0, max_range):
                # this condition down here is looking to be the problem
                str_i = str(i)
                if (word in likelihoods) and (str_i in likelihoods[word]):
                    # print("it got here")
                    # prev_val = classes[i]
                    classes[i] *= likelihoods[word][str_i]
                else:
                    classes[i] *= (1 / (class_counts[str(i)] + len(likelihoods)))

        classification = max(classes, key=classes.get)
        classifications.update({sentence_id: classification})
    return classifications


# evaluate can only be used on the dev set, where weights are given.
def calculate_accuracy(classifications: dict, dev_set: List[List[str]], three_weight: bool) -> None:
    correct = 0
    total = len(classifications)
    if len(classifications) != len(dev_set):
        print("something went wrong.")
    for item in dev_set:
        correct_class = item[2]
        doc_id = item[0]
        if three_weight:
            if str(classifications[doc_id]) == MAP_TO_THREE_DICT[correct_class]:
                correct += 1
        else:
            if str(classifications[doc_id]) == correct_class:
                correct += 1
    percentage = (correct / total * 100)
    print(str(percentage) + "% correct")

def calculate_confusion_matrix():
    pass


# This function will output the results of the posterior probability step using the results.
def output_classification():
    pass


# PREPROCESSING ################################################################
def stemming(sentence):
    stemmed_sentence = None
    return stemmed_sentence


def stop_list(sentence):
    stop_list_sentence = None
    return stop_list_sentence


if __name__ == '__main__':
    dataset_names = ("train.tsv", "dev.tsv")
    three: bool = False

    # Training
    rows = read_and_store_tsv(dataset_names[0])
    bow = create_bag_of_words(rows, three)
    prior_probability = calculate_prior_probability(rows, three)
    likelihood, class_counts = calculate_likelihood(bow, three)
    # Development
    dev_rows = read_and_store_tsv(dataset_names[1])
    posterior = calculate_posterior_probability(prior_probability, likelihood, class_counts, dev_rows, three)

    # Evaluate (Development Only)
    calculate_accuracy(posterior, dev_rows, three)

    print("")
