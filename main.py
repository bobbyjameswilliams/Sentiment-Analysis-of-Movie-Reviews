import csv;

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

def calculate_likelihood(bag_of_words):
    pass
def calculate_prior_probability(dataset_rows, three_weight):
    prior_probabilities = {}
    max_range = 0
    #calculate the total number of sentences
    #calculate the number of sentences for each class
    #calculate the probability
    #assign the probability in the dictionary
    if three_weight:
        max_range = 3
    else:
        max_range = 5

    dataset_size = len(dataset_rows)
    for i in range(0,max_range):
        class_count = 0
        for row in dataset_rows:
            if str(i) in row[2]:
                class_count += 1
        prior_probabilities.update({i:(class_count/dataset_size)})
    return prior_probabilities

if __name__ == '__main__':

    dataset_names = ("train.tsv","dev.tsv")
    dataset_rows = {}
    combined_bag_of_words = {}
    prior_probabilities = {}

    for name in dataset_names:
        rows = read_and_store_tsv(name)
        bow = create_bag_of_words(rows, False)
        prior_probability = calculate_prior_probability(rows, False )

        dict_key_name = name[:(len(name))-4]
        combined_bag_of_words.update({dict_key_name : bow})
        dataset_rows.update({dict_key_name: rows})
        prior_probabilities.update({dict_key_name: prior_probability})

    print("hehe")





