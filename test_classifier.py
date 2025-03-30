"""
This file here tests the different classifiers against the test dataset
"""
from train_classifier import SimpleClassifier, NeuralClassifier, BayesianClassifier
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Choose a model type')
    parser.add_argument(
        '--model',
        type=str,
        choices=['simple', 'neural', 'bayesian'],
        required=True,
        help='Choose a model type: simple, neural, or bayesian'
    )
    parser.add_argument('-k', '--kmer', type=int, default=2, help='Output dataset name')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('-c', '--checkpoint', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(f'You chose the {args.model} model')

    classifier_args = {
        'kmer_len': args.kmer,
        'mapping': './data/mapping.json',
        'pos_path': './data/k_fold_valid.fasta',
        'neg_path': './data/shuffle_k_fold_valid.fasta',
        'test_pos': './data/test_fasta.fasta',
        'test_neg': './data/shuffle_test_fasta.fasta',
        'output_csv': './unrounded_predictions.csv',
        'final_test': './input/public_test.fasta'
    }

    if args.model == 'simple':
        classifier_args['num_sample'] = 100
        classifier_args['original_file'] = './data/uniprot_sprot.fasta'
        classifier = SimpleClassifier(**classifier_args)
    elif args.model == 'neural':
        classifier_args['checkpoint_path'] = args.checkpoint
        classifier_args['epochs'] = 1

        classifier = NeuralClassifier(**classifier_args)
    else:
        classifier = BayesianClassifier(**classifier_args)

    if args.eval:
        classifier.test()
        classifier.eval_file()
    else:
        classifier.train()
