import csv
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import itertools
import re
from nltk.stem import LancasterStemmer, PorterStemmer
from typing import List, Dict
from numpy import ndarray


lancaster = LancasterStemmer()
porter = PorterStemmer()

# Housekeeping

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
        #word = re.sub(r"[a-zA-Z0-9]+", lambda x: x.group(0).lower(), word)
        word = re.sub(r'[^\w\s]', '', word)
        word = word.lower()
        word = word.strip()
    if stem:
        word = porter.stem(word)
        pass
    return word


def create_bag_of_words(c_rows: list, three_weight: bool) -> dict:
    bag_of_words = {}
    for row in c_rows:
        sentence = row[1]
        if three_weight:
            s_class = MAP_TO_THREE_DICT[row[2]]
        else:
            s_class = row[2]

        words = sentence.split(" ")
        for word in words:
            pp_word = preprocessing(word)
            if pp_word != "":
                if pp_word in bag_of_words:
                    if s_class in bag_of_words[pp_word]:
                        prev_value = bag_of_words[pp_word][s_class]
                        bag_of_words[pp_word][s_class] = (prev_value + 1)
                    else:
                        bag_of_words[pp_word].update({s_class: 1})
                else:
                    dict = {pp_word: {s_class: 1}}
                    bag_of_words.update(dict)

    return bag_of_words


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
        c_counts = {
                    "0": 0,
                    "1": 0,
                    "2": 0}
    else:
        max_range = 5
        c_counts = {
                    "0": 0,
                    "1": 0,
                    "2": 0,
                    "3": 0,
                    "4": 0}
    # TODO: put this in another function
    for word in bag_of_words:
        associated_sentiments = bag_of_words[word]
        for sentiment_class in associated_sentiments:
            count = c_counts[sentiment_class] + associated_sentiments[sentiment_class]
            c_counts.update({sentiment_class: count})

    for word in local_bow:
        local_associated_sentiments = local_bow[word]
        for sentiment_class in local_associated_sentiments:
            temp_count = local_associated_sentiments[sentiment_class]
            class_total = c_counts[sentiment_class]
            # with laplace smoothing
            local_associated_sentiments[sentiment_class] = (temp_count + 1) / (class_total + len(local_bow))
            # without laplace smoothing
            # local_associated_sentiments[sentiment_class] = temp_count / class_total
    return local_bow, c_counts


def classify_sentence(sentence, likelihoods, prior_probabilities, three_weight, c_counts, s_list):
    sentence = sentence.split(" ")
    if three_weight:
        l_classes = {
            0: prior_probabilities[0],
            1: prior_probabilities[1],
            2: prior_probabilities[2]}
        max_range = 3
    else:
        l_classes = {
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
                if (pp_word in likelihoods) and (str_i in likelihoods[pp_word]) and (pp_word not in s_list):
                    # print("it got here")
                    # prev_val = classes[i]
                    l_classes[i] *= likelihoods[pp_word][str_i]
                else:
                    l_classes[i] *= (1 / (c_counts[str(i)] + len(likelihoods)))
    return l_classes



def calculate_posterior_probability(prior_probabilities: dict, likelihoods: dict,
                                    c_counts: dict, c_rows: list, three_weight: bool, s_list: ndarray) -> dict:
    classifications = {}
    for row in c_rows:
        sentence_id = row[0]
        sentence = row[1]
        l_classes = classify_sentence(sentence,likelihoods,prior_probabilities,three_weight, c_counts, s_list)

        classification = max(l_classes, key=l_classes.get)
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
        cm = np.zeros((3, 3))
        for item in dev_set:
            doc_id = item[0]
            correct_class = int(MAP_TO_THREE_DICT[item[2]])
            prediction = predictions[doc_id]
            cm[correct_class][prediction] += 1
    else:
        cm = np.zeros((5, 5))
        for item in dev_set:
            doc_id = item[0]
            correct_class = int(item[2])
            prediction = predictions[doc_id]
            cm[correct_class][prediction] += 1
    return cm


def plot_confusion_matrix(cm, target_names, t='Confusion matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    #
    # if normalize:
    #     cm = cm.astype('float') / cm.sum()
    #

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(t)
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
    classes_list = []
    if three_weight:
        max_int = 3
        t = ("Three Class Classifier" + ", Stem=" + str(stem) + ", Punc=" + str(punc))
    else:
        max_int = 5
        t = ("Five Class Classifier" + ", Stem=" + str(stem) + ", Punc=" + str(punc))
    for i in range(0, max_int):
        classes_list.append(str(i))
    return t, classes_list


def calculate_evaluation_dictionaries(cm: ndarray):
    precisions_dict = {}
    recalls_dict = {}
    f1s_dict = {}
    size = cm.shape[0]
    for i in range(0, size):
        correct = cm[i][i]
        if correct > 0:
            precision = correct / sum(cm[i, :])
            recall = correct / sum(cm[:, i])
            f1 = (2 * precision * recall) / (precision + recall)

            precisions_dict.update({i: precision})
            recalls_dict.update({i: recall})
            f1s_dict.update({i: f1})
        else:
            precisions_dict.update({i: 0})
            recalls_dict.update({i: 0})
            f1s_dict.update({i: 0})

    # calculating macro score
    l_macro_f1 = sum(f1s_dict.values()) / size

    return l_macro_f1, precisions_dict, recalls_dict, f1s_dict


#Creates a stoplist by removing the top most occurring words.
def create_zipf_stoplist(stop_bag_of_words: dict, k: int):
    sorted_dict = np.array(sorted(stop_bag_of_words.items(), key=lambda x: sum(x[1].values()), reverse=True))
    sorted_list = np.array(sorted_dict)
    s_list = sorted_list[:k,0]

    return s_list

# This function will output the results of the posterior probability step using the results.
def output_classification():
    pass


if __name__ == '__main__':
    dataset_names = ("train.tsv", "dev.tsv")
    # Preprocessing Booleans

    stem = True
    punc = True
    # stop list, 0 for no stop list
    zipf_stop_k = 5
    standard_stop_list = False

    three: bool = True

    # Training
    rows = read_and_store_tsv(dataset_names[0])
    bow = create_bag_of_words(rows, three)
    prior_probability = calculate_prior_probability(rows, three)
    likelihood, class_counts = calculate_likelihood(bow, three)
    # Development
    if standard_stop_list == True:
        pass
    else:
        stop_list = create_zipf_stoplist(bow, zipf_stop_k)

    dev_rows = read_and_store_tsv(dataset_names[1])
    posterior = calculate_posterior_probability(prior_probability, likelihood, class_counts, dev_rows, three, stop_list)

    # Evaluate (Development Only)
    calculate_accuracy(posterior, dev_rows, three)
    confusion_matrix = calculate_confusion_matrix(posterior, dev_rows, three)
    title, classes = generate_data_for_plot(three)
    plot_confusion_matrix(confusion_matrix, target_names=classes, t=title)
    macro_f1, precisions, recalls, f1s = calculate_evaluation_dictionaries(confusion_matrix)

    print(macro_f1)
    print("")
