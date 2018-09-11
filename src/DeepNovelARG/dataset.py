import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import normalize


def obtain_dataset_wordvectors(dataset_file='', labels_file=''):
    dataset = []
    index = [int(i.strip().split('\t')[1]) for i in open(labels_file)]
    for ix, i in enumerate(open(dataset_file)):
        i = i.split()
        item = np.array([float(k) for k in i[index[ix]:]])
        dataset.append(item)
    # scaler = MinMaxScaler()
    # return scaler.fit_transform(np.array(dataset))
    return normalize(np.array(dataset), axis=-1, order=2)


def obtain_dataset_alignments(dataset_file='', features_file='', file_order=''):
    ''' From an alignment file generate a matrix of values,
        the order of the matrix features depends on the
        features_file, it has to be the same all the time

        file order: contains a list with the entries in the order
                    that they are used for the other sets. For instance,
                    the gene_1 in the fasta file, has to be the same 1
                    position in this file.
        features file: this file contains the list of genes that were
                    used as features, also known as the centroids.
    '''

    dataset = {}
    features = {i.strip().split()[0]: ix for ix,
                i in enumerate(open(features_file))}

    for i in open(dataset_file):
        i = i.split()
        try:
            assert(dataset[i[0]])
        except Exception as e:
            dataset[i[0]] = np.zeros(len(features))
        #
        dataset[i[0]][features[i[1]]] = float(i[-1])

    samples_oder = [i.strip().split('\t')[0] for i in open(file_order)]

    ordered_dataset = []
    for i in samples_oder:
        try:
            ordered_dataset.append(dataset[i])
        except Exception as e:
            ordered_dataset.append(np.zeros(len(features)))
    scaler = MinMaxScaler()

    return [scaler.fit_transform(np.array(ordered_dataset)), features]


def obtain_labels(labels_file=''):
    '''

    From the generated header files, subtract the labels for each ARG.
    Focus on groups and antibiotic categories

    '''
    categories = {}
    groups = {}
    category_index = 0
    group_index = 0
    index_start = []
    for i in open(labels_file):
        i = i.strip().split('\t')
        index_start.append(int(i[1]))
        arg_id, arg_classes, arg_name, arg_group = i[0].split("|")
        for arg_class in arg_classes.split(":"):
            try:
                assert(categories[arg_class])
            except Exception as e:
                categories[arg_class] = category_index
                category_index += 1
        try:
            assert(groups[arg_group])
        except Exception as e:
            groups[arg_group] = group_index
            group_index += 1

    total_categories = len(categories)
    total_groups = len(groups)
    group_labels = []
    category_labels = []

    categories = {i: ix for ix, i in enumerate(categories)}
    groups = {i: ix for ix, i in enumerate(groups)}

    for i in open(labels_file):
        i = i.strip().split('\t')
        #
        arg_id, arg_classes, arg_name, arg_group = i[0].split("|")
        #
        category_label = np.zeros(total_categories)
        group_label = np.zeros(total_groups)
        for arg_class in arg_classes.split(":"):
            category_label[categories[arg_class]] = 1
        #
        group_label[groups[arg_group]] = 1
        group_labels.append(group_label)
        category_labels.append(category_label)

    return [categories, groups, index_start, np.array(group_labels), np.array(category_labels)]
