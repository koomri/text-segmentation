from __future__ import division

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

from choiloader import ChoiDataset, collate_fn
from tqdm import tqdm
from argparse import ArgumentParser
from utils import maybe_cuda
import utils
import sys
from pathlib2 import Path
from wiki_loader import WikipediaDataSet
import accuracy

logger = utils.setup_logger(__name__, 'train.log')



def main(args):
    sys.path.append(str(Path(__file__).parent))

    utils.read_config_file(args.config)
    utils.config.update(args.__dict__)

    logger.debug('Running with config %s', utils.config)
    article_with_problems = 0

    dataset = WikipediaDataSet("dataset_path", word2vec=None,
                               folder=True, high_granularity=False)

    num_sentences = 0
    num_segments = 0
    num_documents = 0
    min_num_segment = 1000
    max_num_segment = 0
    min_num_sentences = 1000
    max_num_sentences = 0


    dl = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
    docs_num_segments_vec = np.zeros(len(dl))
    segments_num_sentences_vec = []
    print 'num of docs is ' + str(len(dl))

    with tqdm(desc='Testing', total=len(dl)) as pbar:

        for i, (data, targets, paths) in enumerate(dl):
            if (len(paths) == 0):
                article_with_problems += 1
                docs_num_segments_vec[i] = np.nan
                continue
            try:

                if ( ((i % 1000 ) == 0) & i > 0):
                    print i
                if len(targets) > 0:
                    targets_var = Variable(maybe_cuda(torch.cat(targets, 0), None), requires_grad=False)
                    target_seg = targets_var.data.cpu().numpy()
                    target_seg = np.concatenate([target_seg, np.array([1])])
                else:
                    target_seg = np.ones(1)
                num_sentences += (len(target_seg))
                doc_num_of_segment = (sum(target_seg))
                if (doc_num_of_segment < min_num_segment):
                    min_num_segment = doc_num_of_segment
                if (doc_num_of_segment > max_num_segment):
                    max_num_segment = doc_num_of_segment
                num_segments += doc_num_of_segment
                num_documents += 1
                docs_num_segments_vec[i] = doc_num_of_segment

                one_inds = np.where(target_seg == 1)[0]
                one_inds += 1
                one_inds = np.concatenate((np.zeros(1),one_inds))
                if (len(one_inds) == 1):
                    sentences_in_segments = [len(target_seg)]
                else:
                    sentences_in_segments = one_inds[1:] - one_inds[:-1]
                segments_num_sentences_vec = np.concatenate((segments_num_sentences_vec,sentences_in_segments))
                current_min = np.min(sentences_in_segments)
                current_max = np.max(sentences_in_segments)
                if (current_min < min_num_sentences):
                    min_num_sentences = current_min
                if (current_max > max_num_sentences):
                    max_num_sentences = current_max



            except Exception as e:
                logger.info('Exception "%s" in batch %s', e, i)
                logger.debug('Exception while handling batch with file paths: %s', paths, exc_info=True)
                raise



    print 'total sentences: {}.'.format(num_sentences)
    print 'total segments: {}.'.format(num_segments)
    print 'total documents: {}.'.format(num_documents)
    print 'average segment size is: {:.3}.'.format(np.true_divide(num_sentences,num_segments))
    print 'min #segment in document: {}.'.format(min_num_segment)
    print 'max #segment in document: {}.'.format(max_num_segment)
    print 'min #sentence in segment: {}.'.format(min_num_sentences)
    print 'max #sentence in segment: {}.'.format(max_num_sentences)


    print ''
    print 'new computing method'
    print ''
    print 'num of documents: {}.'.format(len(docs_num_segments_vec) - np.isnan(docs_num_segments_vec).sum())
    print 'total segments: {}.'.format(np.nansum(docs_num_segments_vec))
    print 'total sentences: {}.'.format(np.sum(segments_num_sentences_vec))
    print ''
    print 'min #segment in document: {}.'.format(np.nanmin(docs_num_segments_vec))
    print 'max #segment in document: {}.'.format(np.nanmax(docs_num_segments_vec))
    print 'mean segments in document: {:.3}.'.format(np.nanmean(docs_num_segments_vec))
    print 'std segments in document: {:.3}.'.format(np.nanstd(docs_num_segments_vec))
    print ''
    print 'min #sentence in segment: {}.'.format(np.min(segments_num_sentences_vec))
    print 'max #sentence in segment: {}.'.format(np.max(segments_num_sentences_vec))
    print 'average segment size is: {:.3}.'.format(np.mean(segments_num_sentences_vec))
    print 'std segment size is: {:.3}.'.format(np.std(segments_num_sentences_vec))

    print ''
    print 'article with problems {}'.format(article_with_problems)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', help='Path to config.json', default='config.json')
    main(parser.parse_args())
