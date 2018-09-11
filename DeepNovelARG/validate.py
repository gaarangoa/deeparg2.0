from tensorflow import keras
from dataset import obtain_labels, obtain_dataset_wordvectors, obtain_dataset_alignments
import json

# Load labels (for testing only!)
classes, groups, index, group_labels, classes_labels = obtain_labels(
    labels_file='../model/dataset.no_centroids.ss.headers'
)

# load dataset
dataset_wordvectors = obtain_dataset_wordvectors(
    dataset_file='../model/dataset.no_centroids.ss.sv',
    labels_file='../model/dataset.no_centroids.ss.headers',
)

dataset_alignments, alignment_features = obtain_dataset_alignments(
    dataset_file='../model/dataset.no_centroids_vs_centroids.tsv',
    features_file='../model/centroids.ids',
    file_order='../model/dataset.no_centroids.ss.headers'
)

model = keras.models.load_model('../model/deeparg2.h5')
ynew = model.predict(
    {
        'wordvectors_input': dataset_wordvectors,
        'alignments_input': dataset_alignments
    },
)

metadata = json.load(open('../model/deearg2.parameters.json'))

for _ix in range(20):
    print('predicting', _ix)
    y_true = group_labels[_ix]
    y_pred = ynew[1][_ix]
    #
    for ix, i in enumerate(y_true):
        if y_pred[ix]*100 <= 5 and i == 0:
            continue
        print(
            i,
            int(y_pred[ix]*100),
            metadata['reverse_groups_dict'][str(ix)]
        )
