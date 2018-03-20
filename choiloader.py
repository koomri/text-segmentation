from __future__ import print_function
from pathlib2 import Path

import torch
from torch.utils.data import Dataset
import numpy as np
import random
from text_manipulation import split_sentences, word_model, extract_sentence_words
import utils
import math


logger = utils.setup_logger(__name__, 'train.log')


def get_choi_files(path):
    all_objects = Path(path).glob('**/*.ref')
    files = [str(p) for p in all_objects if p.is_file()]
    return files

def collate_fn(batch):
    batched_data = []
    batched_targets = []
    paths = []

    window_size = 1
    before_sentence_count = int(math.ceil(float(window_size - 1) /2))
    after_sentence_count = window_size - before_sentence_count - 1

    for data, targets, path in batch:
        try:
            max_index = len(data)
            tensored_data = []
            for curr_sentence_index in range(0, len(data)):
                from_index = max([0, curr_sentence_index - before_sentence_count])
                to_index = min([curr_sentence_index + after_sentence_count + 1, max_index])
                sentences_window = [word for sentence in data[from_index:to_index] for word in sentence]
                tensored_data.append(torch.FloatTensor(np.concatenate(sentences_window)))
            tensored_targets = torch.zeros(len(data)).long()
            tensored_targets[torch.LongTensor(targets)] = 1
            tensored_targets = tensored_targets[:-1]
            batched_data.append(tensored_data)
            batched_targets.append(tensored_targets)
            paths.append(path)
        except Exception as e:
            logger.info('Exception "%s" in file: "%s"', e, path)
            logger.debug('Exception!', exc_info=True)
            continue

    return batched_data, batched_targets, paths

def clean_paragraph(paragraph):
    cleaned_paragraph= paragraph.replace("'' ", " ").replace(" 's", "'s").replace("``", "").strip('\n')
    return cleaned_paragraph

def read_choi_file(path, word2vec, train, return_w2v_tensors = True,manifesto=False):
    seperator = '========' if manifesto else '=========='
    with Path(path).open('r') as f:
        raw_text = f.read()
    paragraphs = [clean_paragraph(p) for p in raw_text.strip().split(seperator)
                  if len(p) > 5 and p != "\n"]
    if train:
        random.shuffle(paragraphs)

    targets = []
    new_text = []
    lastparagraphsentenceidx = 0

    for paragraph in paragraphs:
        if manifesto:
            sentences = split_sentences(paragraph,0)
        else:
            sentences = [s for s in paragraph.split('\n') if len(s.split()) > 0]

        if sentences:
            sentences_count =0
            # This is the number of sentences in the paragraph and where we need to split.
            for sentence in sentences:
                words = extract_sentence_words(sentence)
                if (len(words) == 0):
                    continue
                sentences_count +=1
                if return_w2v_tensors:
                    new_text.append([word_model(w, word2vec) for w in words])
                else:
                    new_text.append(words)

            lastparagraphsentenceidx += sentences_count
            targets.append(lastparagraphsentenceidx - 1)

    return new_text, targets, path


# Returns a list of batch_size that contains a list of sentences, where each word is encoded using word2vec.
class ChoiDataset(Dataset):
    def __init__(self, root, word2vec, train=False, folder=False,manifesto=False, folders_paths = None):
        self.manifesto = manifesto
        if folders_paths is not None:
            self.textfiles = []
            for f in folders_paths:
                self.textfiles.extend(list(f.glob('*.ref')))
        elif (folder):
            self.textfiles = get_choi_files(root)
        else:
            self.textfiles = list(Path(root).glob('**/*.ref'))

        if len(self.textfiles) == 0:
            raise RuntimeError('Found 0 images in subfolders of: {}'.format(root))
        self.train = train
        self.root = root
        self.word2vec = word2vec

    def __getitem__(self, index):
        path = self.textfiles[index]

        return read_choi_file(path, self.word2vec, self.train,manifesto=self.manifesto)

    def __len__(self):
        return len(self.textfiles)
