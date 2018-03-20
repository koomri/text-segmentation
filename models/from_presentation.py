from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import maybe_cuda, setup_logger, unsort
import numpy as np
from times_profiler import profiler


logger = setup_logger(__name__, 'train.log')
profilerLogger = setup_logger("profilerLogger", 'profiler.log', True)


def zero_state(module, batch_size):
    # * 2 is for the two directions
    return Variable(maybe_cuda(torch.zeros(module.num_layers * 2, batch_size, module.hidden))), \
           Variable(maybe_cuda(torch.zeros(module.num_layers * 2, batch_size, module.hidden)))


class SentenceEncodingRNN(nn.Module):
    def __init__(self, input_size=300, hidden=128, num_layers=2):
        super(SentenceEncodingRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden = hidden
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden,
                            num_layers=self.num_layers,
                            dropout=0,
                            bidirectional=True)

    def forward(self, x):
        batch_size = x.batch_sizes[0]
        s = zero_state(self, batch_size)
        _, (hidden, _) = self.lstm(x, s)  # (4, batch_size, 128)
        transposed = hidden.transpose(0, 1)  # (batch_size, 4, 128)
        reshaped = transposed.contiguous().view(batch_size, -1)

        return reshaped


class Model(nn.Module):
    def __init__(self, sentence_encoder, hidden=128, num_layers=2):
        super(Model, self).__init__()

        self.sentence_encoder = sentence_encoder

        self.sentence_lstm = nn.LSTM(input_size=sentence_encoder.hidden * sentence_encoder.num_layers * 2,
                                     hidden_size=hidden,
                                     num_layers=num_layers,
                                     batch_first=True,
                                     dropout=0,
                                     bidirectional=True)

        # We have two labels
        self.h2s = nn.Linear(hidden * 2, 2)

        self.num_layers = num_layers
        self.hidden = hidden

        self.criterion = nn.CrossEntropyLoss()


    def pad(self, s, max_length):
        s_length = s.size()[0]
        v = Variable(maybe_cuda(s.unsqueeze(0).unsqueeze(0)))
        padded = F.pad(v, (0, 0, 0, max_length - s_length))  # (1, 1, max_length, 300)
        shape = padded.size()
        return padded.view(shape[2], 1, shape[3])  # (max_length, 1, 300)


    def pad_document(self, d, max_document_length):
        d_length = d.size()[0]
        v = d.unsqueeze(0).unsqueeze(0)
        padded = F.pad(v, (0, 0,0, max_document_length - d_length ))  # (1, 1, max_length, 300)
        shape = padded.size()
        return padded.view(shape[2], 1, shape[3])  # (max_length, 1, 300)

    def forward(self, batch):
        profiler.init()  # 0

        batch_size = len(batch)

        sentences_per_doc = []
        all_batch_sentences = []
        for document in batch:
            all_batch_sentences.extend(document)
            sentences_per_doc.append(len(document))

        lengths = [s.size()[0] for s in all_batch_sentences]
        sort_order = np.argsort(lengths)[::-1]
        sorted_sentences = [all_batch_sentences[i] for i in sort_order]
        sorted_lengths = [s.size()[0] for s in sorted_sentences]

        max_length = max(lengths)
        logger.debug('Num sentences: %s, max sentence length: %s', 
                     sum(sentences_per_doc), max_length)

        padded_sentences = [self.pad(s, max_length) for s in sorted_sentences]
        big_tensor = torch.cat(padded_sentences, 1)  # (max_length, batch size, 300)
        packed_tensor = pack_padded_sequence(big_tensor, sorted_lengths)
        profiler.set()  # 1
        encoded_sentences = self.sentence_encoder(packed_tensor)
        profiler.set()  # 2
        unsort_order = Variable(maybe_cuda(torch.LongTensor(unsort(sort_order))))
        unsorted_encodings = encoded_sentences.index_select(0, unsort_order)

        index = 0
        encoded_documents = []
        for sentences_count in sentences_per_doc:
            end_index = index + sentences_count
            encoded_documents.append(unsorted_encodings[index : end_index, :])
            index = end_index

        doc_sizes = [doc.size()[0] for doc in encoded_documents]
        max_doc_size = np.max(doc_sizes)
        ordered_document_idx = np.argsort(doc_sizes)[::-1]
        ordered_doc_sizes = sorted(doc_sizes)[::-1]
        ordered_documents = [encoded_documents[idx] for idx in ordered_document_idx]
        padded_docs = [self.pad_document(d, max_doc_size) for d in ordered_documents]
        docs_tensor = torch.cat(padded_docs, 1)
        packed_docs = pack_padded_sequence(docs_tensor, ordered_doc_sizes)
        profiler.set()  # 3
        sentence_lstm_output, _ = self.sentence_lstm(packed_docs, zero_state(self, batch_size=batch_size))
        profiler.set()  # 4
        padded_x, _ = pad_packed_sequence(sentence_lstm_output)  # (max sentence len, batch, 256)

        doc_outputs = []
        for i, doc_len in enumerate(ordered_doc_sizes):
            doc_outputs.append(padded_x[0:doc_len - 1, i, :])  # -1 to remove last prediction

        unsorted_doc_outputs = [doc_outputs[i] for i in unsort(ordered_document_idx)]
        sentence_outputs = torch.cat(unsorted_doc_outputs, 0)

        x = self.h2s(sentence_outputs)

        profiler.finish(profilerLogger)  # 5
        return x


def create():
    sentence_encoder = SentenceEncodingRNN(input_size=300,
                                           hidden=256,
                                           num_layers=2)
    return Model(sentence_encoder, hidden=256, num_layers=2)
