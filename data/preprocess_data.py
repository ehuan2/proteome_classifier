"""Preprocesses the FASTA file to get rid of any lines without our 20 chars"""

from Bio import SeqIO
from Bio.Seq import Seq
from typing import List
from tqdm import tqdm
import argparse
import logging
import json
import os
import random
import sys
import torch


def get_chars(input_file) -> set:
    """Parses the input file path to get the fasta sequence.

    Args:
        input_file (str): Path to input file.

    Returns:
        Seq: The fasta sequence
    """
    # parse the FASTA file
    with open(input_file, 'r') as handle:
        # Parse the FASTA file
        chars = set()
        
        for record in SeqIO.parse(handle, 'fasta'):
            # logging.debug(f'Record id: {record.id}, sequence: {record.seq}')
            # return the first sequence we get
            for char in record.seq:
                chars.add(char)
        
     # let's save this chars as a json file for positioning
    with open('mapping.json', 'w') as file:
        mapping = {
            char: i
            for i, char in
            enumerate(sorted(list(chars)))
        }
        json.dump(mapping, file, indent=4)
        logging.debug('Successfully dumped out mapping')
    
    return chars, mapping


def filter_improper(chars, training, output_path):
    # return the cache
    if os.path.exists(output_path):
        with open(output_path, 'r') as handle:
            records = list(SeqIO.parse(handle, 'fasta'))
        logging.debug(f'Read from cache: {len(records)}')
        return records

    with open(training, 'r') as handle:
        old_records = tqdm(list(SeqIO.parse(handle, 'fasta')))
        previous_size = len(old_records)
        new_records = list(filter(lambda seq: all(char in chars for char in seq), old_records))

        logging.debug(f'Previous size of dataset: {previous_size}, Current size: {len(list(new_records))}')

    SeqIO.write(new_records, output_path, "fasta")

    return new_records

def create_dataset(records, dataset_name):
    if os.path.exists(dataset_name):
        logging.warning(f'Dataset {dataset_name} already exists, remove or restart as needed')
        # now we need to count the number of sequences
        # with open(dataset_name, 'r') as handle:
        #     count = 0
        #     for _ in tqdm(SeqIO.parse(handle, 'fasta')):
        #         count += 1
        #     logging.debug(f'There are {count} sequences cached')
        #     return count
        records = None
        return

    min_length = 20
    max_length = 40

    with open(dataset_name, 'w') as output_file:
        i = 0
        for record in tqdm(records):
            peptide = record.seq
            while len(peptide) > max_length:
                split_point = random.randint(min_length, min(max_length, len(peptide)))
                fragment = peptide[:split_point]
                peptide = peptide[split_point:]

                output_file.write(f'>seq{i}\n')
                output_file.write(f'{fragment}\n')
                i += 1

            if len(peptide) >= min_length:
                output_file.write(f'>seq{i}\n')
                output_file.write(f'{peptide}\n')
                i += 1

    logging.debug(f'Wrote to {dataset_name}, with {i} sequences')

    # also free the memory once it's used
    records = None


def create_shuffle_dataset(dataset_name, shuffle_name):
    assert os.path.exists(dataset_name)
    if os.path.exists(shuffle_name):
        logging.warning(f'Dataset {shuffle_name} already exists, remove or restart as needed')
        return

    with open(shuffle_name, 'w') as output_file:
        i = 0
        with open(dataset_name, 'r') as handle:
            for record in tqdm(SeqIO.parse(handle, 'fasta')):
                seq_chars = list(str(record.seq))
                random.shuffle(seq_chars)
                shuffled_str = ''.join(seq_chars)
                output_file.write(f'>seq{i}\n')
                output_file.write(f'{shuffled_str}\n')
                i += 1

    logging.debug(f'Wrote to {shuffle_name}, with {i} sequences')

def map_kmer_to_index(kmer, mapping):
    """Given some k-mer, maps it to the 4^k index.

    Args:
        kmer (Seq): Given some k-mer, maps it to its proper 4^k index.
    """
    # reverse the kmer which will help with finding the index more easily
    kmer = kmer[::-1]
    
    index = 0
    for i in range(len(kmer)):
        index += mapping[kmer[i]] * (len(mapping.keys()) ** i)

    return index


def get_kmer_frequency_array(seq, k, mapping) -> List[int]:
    """Given some DNA sequence, returns the kmer frequency array.

    Args:
        seq (Seq): The given DNA sequence seq.
        k (int): The kmer length.

    Returns:
        List[float] | List[int]: A list of frequencies that are either partial weights
            or if there are no ambiguities, all ints
    """
    kmer_frequencies = [0 for _ in range(len(mapping.keys()) ** k)]

    # we should iterate n - k + 1 times, from [0, n - k] as our start range
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        # then, transform each kmer to a list of kmers that could represent it
        # after which, we will evenly split the weighting of the kmers' frequency
        # across all possible kmers
        kmer_index = map_kmer_to_index(kmer, mapping)
        kmer_frequencies[kmer_index] += 1

    return kmer_frequencies


def create_tensor_dataset(dataset_prefix, kmer, mapping):
    dataset_name = f'{dataset_prefix}.fasta'
    tensor_dir = f'tensors_{dataset_prefix}/'

    if not os.path.exists(tensor_dir):
        os.makedirs(tensor_dir)

    # import psutil
    # print(psutil.virtual_memory().available > 2**26, psutil.virtual_memory().available, 2**26)
    # based on the memory availability found above, and the fact that
    # the total number of tensors we want to save is 6813050 (number of training examples)
    # let's use 68130 as our chunk size with 101 chunks
    chunk_size = 68130

    with open(dataset_name, 'r') as handle:
        current_chunk = []
        chunk_count = 0

        for peptide in tqdm(SeqIO.parse(handle, 'fasta')):
            current_chunk.append(
                get_kmer_frequency_array(peptide, kmer, mapping)
            )

            if len(current_chunk) >= chunk_size:
                torch.save(
                    torch.tensor(current_chunk),
                    os.path.join(tensor_dir, f'chunk_{chunk_count}.pt')
                )
                chunk_count += 1
                current_chunk = []

        if len(current_chunk) > 1:
            torch.save(
                torch.tensor(current_chunk),
                os.path.join(tensor_dir, f'chunk_{chunk_count}.pt')
            )
            chunk_count += 1
            current_chunk = []

        logging.debug(f'Total number of chunks: {chunk_count}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', required=True, help='Input FASTA file path')
    parser.add_argument('-t', '--training', required=True, help='Input FASTA file path')
    parser.add_argument('-o', '--output_preprocess', required=True, help='Output directory, which if set will write to <input_filename>_k-mers.txt')
    parser.add_argument('-d', '--output_dataset', required=True, help='Output dataset name')
    parser.add_argument('-k', '--kmer', type=int, default=2, help='Output dataset name')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    # first we iterate through the input file to get all unique chars
    chars, mapping = get_chars(args.input_file)
    new_records = filter_improper(chars, args.training, args.output_preprocess)
    
    # we should save this in a two step approach, first as a FASTA file
    # second as a pt tensor instead!
    fasta_output = f'{args.output_dataset}.fasta'
    shuffle_output = f'shuffle_{args.output_dataset}'
    create_dataset(new_records, fasta_output)
    create_shuffle_dataset(fasta_output, f'{shuffle_output}.fasta')

    # finally create the output tensors
    # create_tensor_dataset(args.output_dataset, args.kmer, mapping)
    # create_tensor_dataset(shuffle_output, args.kmer, mapping)
