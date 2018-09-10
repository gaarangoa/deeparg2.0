import numpy as np
# get the data


def obtain_dataset(dataset_file='', index=[]):
    dataset = []
    for ix, i in enumerate(open(dataset_file)):
        i = i.split()
        item = np.array([float(k) for k in i[index[ix]:]])
        dataset.append(item)

    return np.array(dataset)


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
