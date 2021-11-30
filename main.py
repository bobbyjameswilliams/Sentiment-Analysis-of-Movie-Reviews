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



if __name__ == '__main__':
    dataset_names = ("train.tsv","dev.tsv")
    combined_bag_of_words = {}
    for name in dataset_names:
        bow = create_bag_of_words(read_and_store_tsv(name), False)
        dict_key_name = name[:(len(name))-3]
        combined_bag_of_words.update({dict_key_name : bow })
    print("hehe")





