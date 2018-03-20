import json
import logging
import sys
import numpy as np
import random
from pathlib2 import Path
from shutil import copy



config = {}


def read_config_file(path='config.json'):
    global config

    with open(path, 'r') as f:
        config.update(json.load(f))


def maybe_cuda(x, is_cuda=None):
    global config

    if is_cuda is None and 'cuda' in config:
        is_cuda = config['cuda']

    if is_cuda:
        return x.cuda()
    return x


def setup_logger(logger_name, filename, delete_old = False):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    stderr_handler = logging.StreamHandler(sys.stderr)
    file_handler   = logging.FileHandler(filename, mode='w') if delete_old else logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    stderr_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stderr_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
    logger.addHandler(file_handler)
    return logger


def unsort(sort_order):
    result = [-1] * len(sort_order)

    for i, index in enumerate(sort_order):
        result[index] = i

    return result

class f1(object):

    def __init__(self,ner_size):
        self.ner_size = ner_size
        self.tp = np.array([0] * (ner_size +1))
        self.fp = np.array([0] * (ner_size +1))
        self.fn = np.array([0] * (ner_size +1))

    def add(self,preds,targets,length):
        tp = self.tp
        fp = self.fp
        fn = self.fn
        ner_size = self.ner_size

        prediction = np.argmax(preds, 2)

        for i in range(len(targets)):
            for j in range(length[i]):
                if targets[i, j] == prediction[i, j]:
                    tp[targets[i, j]] += 1
                else:
                    fp[targets[i, j]] += 1
                    fn[prediction[i, j]] += 1

        unnamed_entity = ner_size - 1
        for i in range(ner_size):
            if i != unnamed_entity:
                tp[ner_size] += tp[i]
                fp[ner_size] += fp[i]
                fn[ner_size] += fn[i]


    def score(self):
        tp = self.tp
        fp = self.fp
        fn = self.fn
        ner_size = self.ner_size

        precision = []
        recall = []
        fscore = []
        for i in range(ner_size + 1):
            precision.append(tp[i] * 1.0 / (tp[i] + fp[i]))
            recall.append(tp[i] * 1.0 / (tp[i] + fn[i]))
            fscore.append(2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
        print(fscore)

        return fscore[ner_size]


class predictions_analysis(object):

    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0


    def add(self,predicions, targets):
        self.tp += ((predicions == targets) & (1 == predicions)).sum()
        self.tn += ((predicions == targets) & (0 == predicions)).sum()
        self.fp += ((predicions != targets) & (1 == predicions)).sum()
        self.fn += ((predicions != targets) & (0 == predicions)).sum()


    def calc_recall(self):
        if self.tp  == 0 and self.fn == 0:
            return -1

        return np.true_divide(self.tp, self.tp + self.fn)

    def calc_precision(self):
        if self.tp  == 0 and self.fp == 0:
            return -1

        return  np.true_divide(self.tp,self.tp + self.fp)




    def get_f1(self):
        if (self.tp + self.fp == 0):
            return 0.0
        if (self.tp + self.fn == 0):
            return 0.0
        precision = self.calc_precision()
        recall = self.calc_recall()
        if (not ((precision + recall) == 0)):
            f1 = 2*(precision*recall) / (precision + recall)
        else:
            f1 = 0.0

        return f1

    def get_accuracy(self):

        total = self.tp + self.tn + self.fp + self.fn
        if (total == 0) :
            return 0.0
        else:
            return np.true_divide(self.tp + self.tn, total)


    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fn = 0
        self.fp = 0


def get_random_files(count, input_folder, output_folder, specific_section = True):
    files = Path(input_folder).glob('*/*/*/*') if specific_section else Path(input_folder).glob('*/*/*/*/*')
    file_paths = []
    for f in files:
        file_paths.append(f)

    random_paths = random.sample(file_paths, count)

    for random_path in random_paths:
        output_path = Path(output_folder).joinpath(random_path.name)
        copy(str(random_path), str (output_path))