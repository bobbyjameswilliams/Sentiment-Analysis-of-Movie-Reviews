import csv, itertools, re
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from nltk.stem import LancasterStemmer, PorterStemmer
from typing import List, Dict
from numpy import ndarray

lancaster = LancasterStemmer()
porter = PorterStemmer()

# Maps 5 class to 3 class
MAP_TO_THREE_DICT = {'0': '0',
                     '1': '0',
                     '2': '1',
                     '3': '2',
                     '4': '2'}

""" DATA PREPARATION AND HANDLING """


def read_and_store_tsv(file_name: str) -> List[List[str]]:
    """
    :param file_name: Opens file of that filename
    :return: returns rows separated by tabs
    """
    tsv_rows = []
    tsv_file = open(file_name)
    read_tsv_file = csv.reader(tsv_file, delimiter="\t")
    for row in read_tsv_file:
        tsv_rows.append(row)
    tsv_rows.pop(0)
    return tsv_rows


def create_bag_of_words(c_rows: list, three_weight: bool) -> dict:
    """
    Creates a bag of words for use in model creation
    :param c_rows: class rows
    :param three_weight: boolean for if to use three class weight
    :return: bag of words
    """
    bag_of_words = {}
    for row in c_rows:
        sentence = row[1]
        if three_weight:
            s_class = MAP_TO_THREE_DICT[row[2]]
        else:
            s_class = row[2]

        words = sentence.split(" ")
        for word in words:
            # applies preprocessing
            pp_word = preprocessing(word)
            if pp_word != "":
                if pp_word in bag_of_words:
                    if s_class in bag_of_words[pp_word]:
                        prev_value = bag_of_words[pp_word][s_class]
                        bag_of_words[pp_word][s_class] = (prev_value + 1)
                    else:
                        bag_of_words[pp_word].update({s_class: 1})
                else:
                    dictionary = {pp_word: {s_class: 1}}
                    bag_of_words.update(dictionary)

    return bag_of_words


def output_to_tsv(classifications: dict, three_weight: bool) -> None:
    """
    outputs the results to a tsv file
    :param classifications: doc ids and their classifications
    :param three_weight: true if 3 classes
    :return:
    """
    if dev:
        set_spec = "dev"
    else:
        set_spec = "test"

    if three_weight:
        class_spec = "3classes"
    else:
        class_spec = "5classes"

    file_path = "./{}_predictions_{}_Bobby_WILLIAMS.tsv".format(set_spec, class_spec)
    with open(file_path, 'wt') as out_file:
        out_file.write("SentenceId" + "\t" + "Sentiment" + "\n")
        for document in classifications:
            doc_id = document
            classification = classifications[document]
            out_file.write(str(doc_id) + "\t" + str(classification) + "\n")


""" PREPROCESSING AND FEATURE SELECTION """


def preprocessing(word: str) -> str:
    """
    Toggles the preprocessing applied to the words.
    :param word: word to be processed
    :return: peprocessed word
    """
    if punc:
        word = re.sub(r'[^\w\s]', '', word)
        word = word.strip()
    if lower:
        word = word.lower()
    if stem:
        word = porter.stem(word)
        pass
    return word


# Creates a stoplist by removing the top most occurring words.
def create_zipf_stoplist(stop_bag_of_words: dict, k: int):
    """
    creates a stoplist using Zipf's law, removing most common words
    :param stop_bag_of_words: bag of words
    :param k: number of most popular words to be removed
    :return: stop list
    """
    sorted_dict = np.array(sorted(stop_bag_of_words.items(), key=lambda x: sum(x[1].values()), reverse=True))
    sorted_list = np.array(sorted_dict)
    s_list = sorted_list[:k, 0]

    return s_list


""" MODEL CREATING """


def calculate_prior_probability(dataset_rows: list, three_weight: bool) -> Dict[int, float]:
    """
    calculates the prior probability and classifies
    :param dataset_rows: sentences to be classified
    :param three_weight: true if 3 classes
    :return: classifications dict with sentence ids and classifications
    """
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


# Likelihood
def prepare_c_counts(three_weight: bool) -> dict[str, int]:
    """
    returns a dict depending on how many classes are being used
    :param three_weight: true if 3 classes used
    :return: returns dict of the classes initialised as 0
    """
    if three_weight:
        # var below counts the occurrances of each class.
        c_counts = {
            "0": 0,
            "1": 0,
            "2": 0}
    else:
        c_counts = {
            "0": 0,
            "1": 0,
            "2": 0,
            "3": 0,
            "4": 0}
    return c_counts


def calculate_class_counts(c_counts: dict, bag_of_words: dict) -> dict:
    """
    Counts the classes occurrences in the training dataset
    :param c_counts: prepepared dict for class counts
    :param bag_of_words: bag of words from training set
    :return: dict of the classes and how many times they occur
    """
    for word in bag_of_words:
        associated_sentiments = bag_of_words[word]
        for sentiment_class in associated_sentiments:
            count = c_counts[sentiment_class] + associated_sentiments[sentiment_class]
            c_counts.update({sentiment_class: count})
    return c_counts


def calculate_likelihood(bag_of_words: dict, three_weight: bool) -> (dict, Dict[int, str]):
    """
    Calculates the likelihood and returns a dict of them
    :param bag_of_words: bag of words from training set
    :param three_weight: true if 3 weights used
    :return: dict of likelihoods
    """
    local_bow = deepcopy(bag_of_words)
    c_counts = calculate_class_counts(prepare_c_counts(three_weight), bag_of_words)

    for word in local_bow:
        local_associated_sentiments = local_bow[word]
        for sentiment_class in local_associated_sentiments:
            temp_count = local_associated_sentiments[sentiment_class]
            class_total = c_counts[sentiment_class]
            # Performs laplace smoothing if true. Laplace smoothing is applied in the likelihoods and again in classify
            # for 0 values.
            if laplace:
                local_associated_sentiments[sentiment_class] = (temp_count + 1) / (class_total + len(local_bow))
            else:
                local_associated_sentiments[sentiment_class] = temp_count / class_total
    return local_bow, c_counts


""" CLASSIFICATION """


def classify_sentence(sentence: str, likelihoods: dict, prior_probabilities: dict, three_weight: bool, c_counts: dict,
                      s_list: ndarray) -> int:
    """
    given a sentence and the related information, returns classification
    :param sentence: sentence for classification
    :param likelihoods: likelihoods
    :param prior_probabilities: probabilities
    :param three_weight: true if 3 classes used
    :param c_counts: dict of classes and their total occurrances
    :param s_list: stop list
    :return: integer classification
    """
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
        # applies preprocessing
        pp_word = preprocessing(word)
        if (pp_word != ""):
            for i in range(0, max_range):
                str_i = str(i)
                if (pp_word in likelihoods) and (str_i in likelihoods[pp_word]) and (pp_word not in s_list):
                    l_classes[i] *= likelihoods[pp_word][str_i]
                else:
                    # Performs laplace smoothing if laplace is true
                    if laplace:
                        l_classes[i] *= (1 / (c_counts[str(i)] + len(likelihoods)))
                    else:
                        l_classes[i] *= 0

    return max(l_classes, key=l_classes.get)


def calculate_posterior_probability(prior_probabilities: dict, likelihoods: dict,
                                    c_counts: dict, c_rows: list, three_weight: bool, s_list: ndarray) -> dict:
    """
    calculates the posterior probability and classifies the sentence, returning the classifications in a dict.
    :param prior_probabilities: prior probabilities
    :param likelihoods: likelihoods
    :param c_counts: class counts dict
    :param c_rows: class rows
    :param three_weight: true if 3 classes used
    :param s_list: stop list
    :return: a dict of classificatons sentenceid: classification
    """
    classifications = {}
    for row in c_rows:
        sentence_id = row[0]
        sentence = row[1]
        classification = classify_sentence(sentence, likelihoods, prior_probabilities, three_weight, c_counts, s_list)

        classifications.update({sentence_id: classification})
    return classifications


"""" EVALUATION FUNCTIONS """


# evaluate can only be used on the dev set, where weights are given.
def calculate_accuracy(classifications: dict, dev_set: List[List[str]], three_weight: bool) -> None:
    """
    calculates accuracy if dev set is used
    :param classifications: created using the model
    :param dev_set: dev set
    :param three_weight: true if 3 classes used
    :return: None. Outputs to console.
    """
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
    """
    creates confusion matrix based on the data.
    :param predictions: predictions made from your own classification step
    :param dev_set: dev set
    :param three_weight: true if 3 weights used
    :return: returns confusion matrix
    """
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


def plot_confusion_matrix(cm, target_names, t='Confusion matrix') -> None:
    """
    plots confusion matrix. Adapted from the lab class.
    :param cm: confusion matrix
    :param target_names: list of classes
    :param t: title
    :return: None
    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

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
        plt.text(j, i, "{:,}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.grid(False)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def generate_data_for_plot(three_weight: bool) -> (str, list[str]):
    """
    prepares data for the plot
    :param three_weight: true if 3 weights used
    :return: title and list of classes
    """
    classes_list = []
    if three_weight:
        max_int = 3
        t = ("Three Class Classifier" + ", Stem=" + str(stem) + ", Punc=" + str(punc) + ", Lower=" + str(lower))
    else:
        max_int = 5
        t = ("Five Class Classifier" + ", Stem=" + str(stem) + ", Punc=" + str(punc) + ", Lower=" + str(lower))
    for i in range(0, max_int):
        classes_list.append(str(i))
    return t, classes_list


def calculate_evaluation_dictionaries(cm: ndarray) -> (float, dict[int, int], dict[int, int], dict[int, int]):
    """
    creates dictionaries for evaluation to calculate F1
    :param cm: confusion matrix
    :return: l_macro_f1, precisions_dict, recalls_dict, f1s_dict
    """
    precisions_dict = {}
    recalls_dict = {}
    f1s_dict = {}
    size = cm.shape[0]
    for i in range(0, size):
        correct = cm[i][i]
        # cannot divide by 0
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


if __name__ == '__main__':
    """ CONFIG BOOLEANS """
    # set to true if using a dev set. false if just classifying
    dev = True
    # Set to true to output results to CSV
    output = False

    # Laplace Smoothing
    laplace = True

    # Preprocessing Booleans
    stem = True
    punc = False
    lower = False

    # Feature Selection Parameters
    # Stop list, 0 for no stop list
    zipf_stop_k = 2

    # True for 3 classes. False for 5
    three: bool = True
    """" END CONFIG BOOLEANS """

    # Preparation of dataset names
    if dev:
        dataset_names = ("train.tsv", "dev.tsv")
    else:
        dataset_names = ("train.tsv", "test.tsv")

    # Training
    rows = read_and_store_tsv(dataset_names[0])
    bow = create_bag_of_words(rows, three)
    prior_probability = calculate_prior_probability(rows, three)
    likelihood, class_counts = calculate_likelihood(bow, three)
    # Development

    # parametrised Zipfs law stoplist
    stop_list = create_zipf_stoplist(bow, zipf_stop_k)

    # rows that are to be classified
    classification_rows = read_and_store_tsv(dataset_names[1])
    final_classifications = calculate_posterior_probability(prior_probability, likelihood, class_counts,
                                                            classification_rows, three,
                                                            stop_list)
    if dev:
        # Evaluate (Development Only)
        calculate_accuracy(final_classifications, classification_rows, three)
        confusion_matrix = calculate_confusion_matrix(final_classifications, classification_rows, three)
        title, classes = generate_data_for_plot(three)
        plot_confusion_matrix(confusion_matrix, target_names=classes, t=title)
        macro_f1, precisions, recalls, f1s = calculate_evaluation_dictionaries(confusion_matrix)
        print(macro_f1)

    if output:
        output_to_tsv(final_classifications, three)
