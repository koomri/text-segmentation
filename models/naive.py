import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import maybe_cuda, unsort
import numpy as np



class Naive(nn.Module):
    def __init__(self, segment_average_size):
        super(Naive, self).__init__()

        self.segment_average_size = segment_average_size
        self.criterion = nn.CrossEntropyLoss()



    def create_random_output(self,size):

        cut_probability = float (1) / self.segment_average_size

        cuts = np.random.choice([0, 1], size=(size,), p=[1-cut_probability, cut_probability])
        ret = torch.zeros(size,2)

        for i in range(ret.size()[0]):
            ret[i,1] = cuts[i]
            ret[i,0] = 1 - cuts[i]
        return ret

    def forward(self, x):

        batch_segmentations = []
        for document in x:
            num_sentences = len(document)
            doc_segmentation = self.create_random_output(num_sentences - 1)
            batch_segmentations.append(doc_segmentation)
        batch_output = torch.cat(batch_segmentations,0)
        return Variable(batch_output)


def create():
    return Naive(13)
