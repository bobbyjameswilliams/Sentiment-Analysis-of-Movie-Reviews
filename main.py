import csv;
from copy import deepcopy
import struct

#Housekeeping

def read_and_store_tsv(fileName: str):
    tsv_rows = []
    tsv_file = open(fileName)
    read_tsv_file = csv.reader(tsv_file, delimiter="\t")
    for row in read_tsv_file:
        tsv_rows.append(row)
    tsv_rows.pop(0)
    return tsv_rows

def create_bag_of_words(rows: [str], three_weight):
    map_to_three_dict = {'0': '0',
                         '1': '0',
                         '2': '1',
                         '3': '2',
                         '4': '2'}
    bagOfWords = {}
    for row in rows:
        sentence = row[1]
        if three_weight:
                s_class = map_to_three_dict[row[2]]
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

#Model Creating

def calculate_likelihood(bag_of_words, three_weight : bool):
    local_bow =  deepcopy(bag_of_words)
    if three_weight:
        max_range = 3
        class_counts = {"0":0,
                        "1":0,
                        "2":0}
    else:
        max_range = 5
        class_counts = {"0":0,
                        "1":0,
                        "2":0,
                        "3":0,
                        "4":0}
    #TODO: put this in another function
    for word in bag_of_words:
        for wClass in bag_of_words[word]:
            count = class_counts[wClass] + bag_of_words[word][wClass]
            class_counts.update({wClass: count})

    for word in local_bow:
        for wClass  in local_bow[word]:
            temp_count =  local_bow[word][wClass]
            class_total = class_counts[wClass]
            #with laplace smoothing
            local_bow[word][wClass] = (temp_count + 1) / (class_total + len(local_bow))
    return local_bow, class_counts

#TODO fix for three
def calculate_prior_probability(dataset_rows, three_weight):
    
    return prior_probabilities

def calculate_posterior_probability(prior_probabilities, likelihoods, class_counts, rows, three_weight):
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
        #calculation includes laplace smoothing if the word does not appear in the likelihoods or is there but has no
        #weight attached.
        for word in sentence:
            for i in range(0, max_range):
                if (word in likelihoods) and (i in likelihoods[word]) :
                    #prev_val = classes[i]
                    classes[i] *= likelihoods[word][i]
                else:
                    classes[i] *= (1/ (class_counts[str(i)] + len(likelihoods)))

        classification = max(classes, key=classes.get)
        classifications.update({sentence_id: classification})
    return classifications



def evaluate():
    pass

def classify():
    pass

if __name__ == '__main__':

    dataset_names = ("train.tsv","dev.tsv")
    three: bool = True

    #Training
    rows = read_and_store_tsv(dataset_names[0])
    bow = create_bag_of_words(rows, three)
    prior_probability = calculate_prior_probability(rows, three )
    likelihood, class_counts = calculate_likelihood(bow, three)

    # dict_key_name = dataset_names[0][:(len(dataset_names[0]))-4]
    # combined_bag_of_words.update({dict_key_name : bow})
    # likelihoods.update({dict_key_name: likelihood})
    # dataset_rows.update({dict_key_name: rows})
    # prior_probabilities.update({dict_key_name: prior_probability})

    #Development
    rows = read_and_store_tsv(dataset_names[1])
    posterior = calculate_posterior_probability(prior_probability,likelihood,class_counts,rows,three)

    print("hehe")
    print("bruh")




