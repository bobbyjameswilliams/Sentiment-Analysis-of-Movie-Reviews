import csv
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import itertools
import re
from nltk.stem import LancasterStemmer, PorterStemmer

lancaster = LancasterStemmer()
porter = PorterStemmer()

# Housekeeping
from typing import List, Dict

from numpy import ndarray

MAP_TO_THREE_DICT = {'0': '0',
                     '1': '0',
                     '2': '1',
                     '3': '2',
                     '4': '2'}


def read_and_store_tsv(file_name: str) -> List[List[str]]:
    tsv_rows = []
    tsv_file = open(file_name)
    read_tsv_file = csv.reader(tsv_file, delimiter="\t")
    for row in read_tsv_file:
        tsv_rows.append(row)
    tsv_rows.pop(0)
    return tsv_rows


def stemming():
    pass


def stoplist():
    pass


def preprocessing(word: str):
    if punc:
        word = re.sub(r'[^\w\s]', '', word)
        word = word.lower()
        word = word.strip()
    if stem:
        word = porter.stem(word)
        pass
    return word


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
            pp_word = preprocessing(word)
            if pp_word != "":
                if pp_word in bagOfWords:
                    if s_class in bagOfWords[pp_word]:
                        prevValue = bagOfWords[pp_word][s_class]
                        bagOfWords[pp_word][s_class] = (prevValue + 1)
                    else:
                        bagOfWords[pp_word].update({s_class: 1})
                else:
                    dict = {pp_word: {s_class: 1}}
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


def calculate_likelihood(bag_of_words: dict, three_weight: bool) -> (dict, Dict[int, str]):
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
                                    class_counts: dict, rows: list, three_weight: bool, stop_list: ndarray) -> dict:
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
            pp_word = preprocessing(word)
            if (pp_word != ""):
                for i in range(0, max_range):
                    # this condition down here is looking to be the problem
                    str_i = str(i)
                    if (pp_word in likelihoods) and (str_i in likelihoods[pp_word]) and (pp_word not in stop_list):
                        # print("it got here")
                        # prev_val = classes[i]
                        classes[i] *= likelihoods[pp_word][str_i]
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


def calculate_confusion_matrix(predictions: dict, dev_set: List[List[str]], three_weight: bool) -> ndarray:
    if len(predictions) != len(dev_set):
        print("something went wrong.")

    if three_weight:
        confusion_matrix = np.zeros((3, 3))
        for item in dev_set:
            doc_id = item[0]
            correct_class = int(MAP_TO_THREE_DICT[item[2]])
            prediction = predictions[doc_id]
            confusion_matrix[correct_class][prediction] += 1
    else:
        confusion_matrix = np.zeros((5, 5))
        for item in dev_set:
            doc_id = item[0]
            correct_class = int(item[2])
            prediction = predictions[doc_id]
            confusion_matrix[correct_class][prediction] += 1
    return confusion_matrix


def plot_confusion_matrix(cm, target_names, title='Confusion matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    #
    # if normalize:
    #     cm = cm.astype('float') / cm.sum()
    #

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # if normalize:
        #     plt.text(j, i, "{:0.4f}".format(cm[i, j]),
        #              horizontalalignment="center",
        #              color="white" if cm[i, j] > thresh else "black")
        # else:
        plt.text(j, i, "{:,}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.grid(False)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def generate_data_for_plot(three_weight: bool):
    classes = []
    if three_weight:
        max = 3
        title = ("Three Class Classifier" + ", Stem=" + str(stem) + ", Punc=" + str(punc))
    else:
        max = 5
        title = ("Five Class Classifier" + ", Stem=" + str(stem) + ", Punc=" + str(punc))
    for i in range(0, max):
        classes.append(str(i))
    return title, classes


def calculate_evaluation_dictionaries(confusion_matrix: ndarray):
    precisions = {}
    recalls = {}
    f1s = {}
    size = confusion_matrix.shape[0]
    for i in range(0, size):
        correct = confusion_matrix[i][i]
        if correct > 0:
            precision = correct / sum(confusion_matrix[i, :])
            recall = correct / sum(confusion_matrix[:, i])
            f1 = (2 * precision * recall) / (precision + recall)

            precisions.update({i: precision})
            recalls.update({i: recall})
            f1s.update({i: f1})
        else:
            precisions.update({i: 0})
            recalls.update({i: 0})
            f1s.update({i: 0})

    # calculating macro score
    l_macro_f1 = sum(f1s.values()) / size

    return l_macro_f1, precisions, recalls, f1s


def create_stoplist(stop_bag_of_words: dict, k: int):
    sorted_dict = np.array(sorted(stop_bag_of_words.items(), key=lambda x: sum(x[1].values()), reverse=True))
    sorted_list = np.array(sorted_dict)
    stop_list = sorted_list[:k,0]
    return stop_list

# This function will output the results of the posterior probability step using the results.
def output_classification():
    pass


if __name__ == '__main__':
    dataset_names = ("train.tsv", "dev.tsv")
    # Preprocessing Booleans

    stem = True
    punc = True
    #stop list, 0 for no stop list
    stop_k = 4

    three: bool = False

    # Training
    rows = read_and_store_tsv(dataset_names[0])
    bow = create_bag_of_words(rows, three)
    prior_probability = calculate_prior_probability(rows, three)
    likelihood, class_counts = calculate_likelihood(bow, three)
    # Development
    stop_list = create_stoplist(bow,stop_k)
    dev_rows = read_and_store_tsv(dataset_names[1])
    posterior = calculate_posterior_probability(prior_probability, likelihood, class_counts, dev_rows, three, stop_list)

    # Evaluate (Development Only)
    calculate_accuracy(posterior, dev_rows, three)
    confusion_matrix = calculate_confusion_matrix(posterior, dev_rows, three)
    title, classes = generate_data_for_plot(three)
    plot_confusion_matrix(confusion_matrix, target_names=classes, title=title)
    macro_f1, precisions, recalls, f1s = calculate_evaluation_dictionaries(confusion_matrix)

    print(macro_f1)
    print("")
