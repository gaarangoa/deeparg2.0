from tensorflow import keras
from DeepNovelARG.dataset import obtain_labels, obtain_dataset_wordvectors, obtain_dataset_alignments
import json
import click


@click.command()
@click.option('--indir', required=True, help='input directory with stage one')
@click.option('--modeldir', default='', help='Directory where the model is stored')
@click.option('--minp', default=0.1, help='minimum probability')
def predict(indir, modeldir, minp):
    """

    Input a fasta file and predict the sequences that are args.

    """

    # load dataset
    dataset_wordvectors = obtain_dataset_wordvectors(
        dataset_file=indir+'/dataset.no_centroids.ss.sv',
        labels_file=indir+'/dataset.no_centroids.ss.headers',
    )

    dataset_alignments, alignment_features = obtain_dataset_alignments(
        dataset_file=indir+'/dataset.no_centroids_vs_centroids.tsv',
        features_file=modeldir+'/centroids.ids',
        file_order=indir+'/dataset.no_centroids.ss.headers'
    )

    model = keras.models.load_model(modeldir+'/deeparg2.h5')
    ynew = model.predict(
        {
            'wordvectors_input': dataset_wordvectors,
            'alignments_input': dataset_alignments
        },
    )

    metadata = json.load(open(modeldir+'/deearg2.parameters.json'))

    file_order = [
        i.strip().split("\t")[0] for i in open(indir + '/dataset.no_centroids.ss.headers')
    ]

    fo = open(indir + '/predicted.classes.txt', 'w')
    fo.write("#Query\tProbability\tPrediction\n")
    for _ix in range(len(ynew[0])):
        y_pred = ynew[0][_ix]
        query = file_order[_ix]
        for ix, i in enumerate(y_pred):
            if round(i, 2) <= minp:
                continue
            fo.write("\t".join([
                query,
                str(round(i, 2)),
                metadata['reverse_classes_dict'][str(ix)]
            ]) + "\n")

    fo = open(indir + '/predicted.groups.txt', 'w')
    fo.write("#Query\tProbability\tPrediction\n")
    for _ix in range(len(ynew[1])):
        y_pred = ynew[1][_ix]
        query = file_order[_ix]
        for ix, i in enumerate(y_pred):
            if round(i, 2) <= minp:
                continue
            fo.write("\t".join([
                query,
                str(round(i, 2)),
                metadata['reverse_groups_dict'][str(ix)]
            ])+"\n")
