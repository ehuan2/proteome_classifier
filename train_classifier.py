from Bio import SeqIO
import json
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random

class Classifier():
    def __init__(self, kmer_len, mapping, pos_path, neg_path, test_pos, test_neg, final_test, output_csv):
        self.kmer_len = kmer_len
        # Read the JSON file and create a PyTorch tensor
        with open(mapping, 'r') as f:
            self.mapping = json.load(f)
        self.pos_path = pos_path
        self.neg_path = neg_path
        self.test_pos = test_pos
        self.test_neg = test_neg
        self.final_test = final_test
        self.output_csv = output_csv

        # also add the counts for the positive path (should match negative)
        # such that we can iterate through it faster
        self.train_count = 6813050
        # self.train_count = 0
        # with open(self.pos_path, 'r') as pos_path_file:
        #     for _ in tqdm(SeqIO.parse(pos_path_file, 'fasta')):
        #         self.train_count += 1
        # print(self.train_count)

    def _map_kmer_to_index(self, kmer):
        """Given some k-mer, maps it to the 4^k index.

        Args:
            kmer (Seq): Given some k-mer, maps it to its proper 4^k index.
        """
        # reverse the kmer which will help with finding the index more easily
        kmer = kmer[::-1]
        
        index = 0
        for i in range(len(kmer)):
            index += self.mapping[kmer[i]] * (len(self.mapping.keys()) ** i)

        return index

    def _get_kmer_frequency_array(self, seq):
        """Given some DNA sequence, returns the kmer frequency array.

        Args:
            seq (Seq): The given DNA sequence seq.
            k (int): The kmer length.

        Returns:
            List[float] | List[int]: A list of frequencies that are either partial weights
                or if there are no ambiguities, all ints
        """
        kmer_frequencies = [0 for _ in range(len(self.mapping.keys()) ** self.kmer_len)]

        # we should iterate n - k + 1 times, from [0, n - k] as our start range
        for i in range(len(seq) - self.kmer_len + 1):
            kmer = seq[i:i + self.kmer_len]
            # then, transform each kmer to a list of kmers that could represent it
            # after which, we will evenly split the weighting of the kmers' frequency
            # across all possible kmers
            kmer_index = self._map_kmer_to_index(kmer)
            kmer_frequencies[kmer_index] += 1

        return np.array(kmer_frequencies)

    def _seq_to_freq_array(self, seq):
        freq_array = self._get_kmer_frequency_array(seq)
        freq_array = freq_array / sum(freq_array)
        return torch.tensor(freq_array, dtype=torch.float)

    def classify_seq(self, seq):
        # given a sequence, calculate its frequency and then pass through the prediction
        # first however, we need to turn this sequence to a kmer frequency array
        # the first person to make a contiguous block of memory must've gotten arrays
        self._predict_tensor(self._seq_to_freq_array(seq))

    def train(self):
        pass

    def _predict_tensor(self, tensor):
        pass

    def test(self):
        with open(self.test_pos, 'r') as test_pos_file:
            for record in SeqIO.parse(test_pos_file, 'fasta'):
                # now do something with the positive probability
                print(f'Classifying {record.seq}')
                prob = self.classify_seq(record.seq)
                break

        with open(self.test_neg, 'r') as test_neg_file:
            for record in SeqIO.parse(test_neg_file, 'fasta'):
                print(f'Classifying {record.seq}')
                prob = self.classify_seq(record.seq)
                break


    def eval_file(self):
        with open(self.final_test, 'r') as final_test_file:
            with open(self.output_csv, 'w') as csv_file:
                for record in SeqIO.parse(final_test_file, 'fasta'):
                    csv_file.write(f'{record}, {self.classify_seq(record.seq)}')

    def _score(self, pos_tensor, neg_tensor):
        return torch.sum(
            torch.abs(
                torch.log(
                    (pos_tensor + sys.float_info.epsilon) /
                    (neg_tensor + sys.float_info.epsilon)
                )
            )
        )


class SimpleClassifier(Classifier):
    def __init__(self, kmer_len, mapping, pos_path, neg_path, test_pos, test_neg, final_test, output_csv, num_sample):
        super().__init__(kmer_len, mapping, pos_path, neg_path, test_pos, test_neg, final_test, output_csv)
        # how many peptides to sample from randomly
        self.num_sample = num_sample

    def _predict_tensor(self, tensor):
        # for the simple classifier, we build a score by randomly sampling
        # num_sample peptides from our negative dataset
        # where we take the lowest score as the chance of it being most random
        sample_peptides = random.sample(range(self.train_count), self.num_sample) if self.num_sample is not None else None

        score = None

        # the way the negative one goes is that we only want to score against q
        with open(self.neg_path, 'r') as neg_path_file:
            for i, record in tqdm(enumerate(SeqIO.parse(neg_path_file, 'fasta')), total=self.train_count):
                if sample_peptides is None or i in sample_peptides:
                    next_score = self._score(tensor, self._seq_to_freq_array(record.seq))
                    score = torch.min(score, next_score) if score is not None else next_score

                if sample_peptides is not None and max(sample_peptides) < i:
                    break
                    
        print(f'Final score: {score}')

    def train(self):
        # no training necessary for this...
        print(f'Classifier does not need to be trained...')

class NeuralClassifier(Classifier):
    class LinearClassifier(nn.Module):
        def __init__(self, n, input_dim):
            super(NeuralClassifier.LinearClassifier, self).__init__()
            self.layers = nn.ModuleList()
            
            # Initialize the first layer with input dimension
            # self.layers.append(nn.Linear(input_dim, input_dim))
            self.dropout = nn.Dropout(0.2)
            
            # Initialize the hidden layers
            for _ in range(n):
                output_dim = int(input_dim / 20)
                self.layers.append(nn.Linear(input_dim, output_dim))
                input_dim = output_dim
        
        def forward(self, x):
            # let's do a sigmoid on the very last layer
            x = self.dropout(x)
            for i, layer in enumerate(self.layers):
                if i == len(self.layers) - 1:
                    x = torch.sigmoid(layer(x))
                else:
                    x = torch.relu(layer(x))
            return x

    def __init__(self, kmer_len, mapping, pos_path, neg_path, test_pos, test_neg, final_test, output_csv, checkpoint_path, epochs):
        super().__init__(kmer_len, mapping, pos_path, neg_path, test_pos, test_neg, final_test, output_csv)
        self.classifier = NeuralClassifier.LinearClassifier(kmer_len, 20**kmer_len)
        self.batch_size = 512
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=0.01)
        # just do a simple huber loss, robust to outliers
        self.criterion = nn.HuberLoss(delta=1.0)
        self.checkpoint_path = checkpoint_path

        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path)
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(self.device)

    def _predict_tensor(self, tensor):
        print(tensor)
    
    def batch_eval(self, batch_tensor, labels):
        # evaluates the batch tensor and then does the gradient descent
        self.optimizer.zero_grad()

        # send the batched tensor to the device
        batch_tensor = batch_tensor.to(self.device)
        labels = labels.to(self.device)

        outputs = self.classifier(batch_tensor)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

    def train(self):
        # mix the positive and negative samples together
        self.classifier.train()
        current_batch = None
        with open(self.neg_path, 'r') as neg_path_file:
            with open(self.pos_path, 'r') as pos_path_file:
                for epoch in range(self.epochs):
                    for i, records in tqdm(
                            enumerate(zip(
                                SeqIO.parse(pos_path_file, 'fasta'),
                                SeqIO.parse(neg_path_file, 'fasta'),
                            )), total=self.train_count):

                        pos_record, neg_record = records

                        # we'll be doing both the positive and negative record
                        if current_batch is None:
                            current_batch = self._seq_to_freq_array(pos_record.seq)
                            current_batch = torch.stack((current_batch, self._seq_to_freq_array(neg_record.seq)))
                            assert torch.all(current_batch[0] == self._seq_to_freq_array(pos_record.seq))
                            assert torch.all(current_batch[1] == self._seq_to_freq_array(neg_record.seq))
                        else:
                            pos_record_freq = torch.unsqueeze(self._seq_to_freq_array(pos_record.seq), 0)
                            neg_record_freq = torch.unsqueeze(self._seq_to_freq_array(neg_record.seq), 0)
                            current_batch = torch.cat((current_batch, pos_record_freq), dim=0)
                            current_batch = torch.cat((current_batch, neg_record_freq), dim=0)

                            assert torch.all(current_batch[-2] == pos_record_freq)
                            assert torch.all(current_batch[-1] == neg_record_freq)
                            
                        if i % self.batch_size == self.batch_size - 1:
                            # todo: the batch eval
                            labels = torch.tensor([1, 0] * self.batch_size)
                            labels = labels.to(torch.float)
                            labels = labels.unsqueeze(1)
                            self.batch_eval(current_batch, labels)
                            current_batch = None
                            batch_num = i // self.batch_size
                            if batch_num % 50 == 49:
                                torch.save(self.classifier.state_dict(), f'k_{self.kmer_len}_model_epoch_{epoch + 1}_batch_{batch_num}.pth')
                    torch.save(self.classifier.state_dict(), f'k_{self.kmer_len}_model_epoch_{epoch + 1}.pth')


class BayesianClassifier(Classifier):
    def __init__(self, kmer_len, mapping, pos_path, neg_path, test_pos, test_neg, final_test, output_csv):
        super().__init__(kmer_len, mapping, pos_path, neg_path, test_pos, test_neg, final_test, output_csv)

    def _predict_tensor(self, tensor):
        print(tensor)

    def train(self):
        pass
