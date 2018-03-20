import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

from choiloader import ChoiDataset, collate_fn
from tqdm import tqdm
from argparse import ArgumentParser
from utils import maybe_cuda
import gensim
import utils
from tensorboard_logger import configure
import os
import sys
from pathlib2 import Path
import accuracy
import numpy as np
from termcolor import colored

torch.multiprocessing.set_sharing_strategy('file_system')

preds_stats = utils.predictions_analysis()


def softmax(x):
    max_each_row = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - max_each_row)
    sums = np.sum(exps, axis=1, keepdims=True)
    return exps / sums


def import_model(model_name):
    module = __import__('models.' + model_name, fromlist=['models'])
    return module.create()


class Accuracies(object):
    def __init__(self):
        self.thresholds = np.arange(0, 1, 0.05)
        self.accuracies = {k: accuracy.Accuracy() for k in self.thresholds}

    def update(self, output_np, targets_np):
        current_idx = 0
        for k, t in enumerate(targets_np):
            document_sentence_count = len(t)
            to_idx = int(current_idx + document_sentence_count)

            for threshold in self.thresholds:
                output = ((output_np[current_idx: to_idx, :])[:, 1] > threshold)
                h = np.append(output, [1])
                tt = np.append(t, [1])

                self.accuracies[threshold].update(h, tt)

            current_idx = to_idx

    def calc_accuracy(self):
        min_pk = np.inf
        min_threshold = None
        min_epoch_windiff = None
        for threshold in self.thresholds:
            epoch_pk, epoch_windiff = self.accuracies[threshold].calc_accuracy()
            if epoch_pk < min_pk:
                min_pk = epoch_pk
                min_threshold = threshold
                min_epoch_windiff = epoch_windiff

        return min_pk, min_epoch_windiff, min_threshold

def validate(model, args, epoch, dataset, logger):
    model.eval()
    with tqdm(desc='Validatinging', total=len(dataset)) as pbar:
        acc = Accuracies()
        for i, (data, target, paths) in enumerate(dataset):
            if True:
                if i == args.stop_after:
                    break
                pbar.update()
                output = model(data)
                output_softmax = F.softmax(output, 1)
                targets_var = Variable(maybe_cuda(torch.cat(target, 0), args.cuda), requires_grad=False)

                output_seg = output.data.cpu().numpy().argmax(axis=1)
                target_seg = targets_var.data.cpu().numpy()
                preds_stats.add(output_seg, target_seg)

                acc.update(output_softmax.data.cpu().numpy(), target)

        epoch_pk, epoch_windiff, threshold = acc.calc_accuracy()

        logger.info('Validating Epoch: {}, accuracy: {:.4}, Pk: {:.4}, Windiff: {:.4}, F1: {:.4} . '.format(epoch + 1,
                                                                                                            preds_stats.get_accuracy(),
                                                                                                            epoch_pk,
                                                                                                            epoch_windiff,
                                                                                                            preds_stats.get_f1()))
        preds_stats.reset()

        return epoch_pk, threshold


def test(model, args, epoch, dataset, logger, test_threshold, test_acc):
    model.eval()
    with tqdm(desc='Testing', total=len(dataset)) as pbar:
        for i, (data, target, paths) in enumerate(dataset):
            if True:
                if i == args.stop_after:
                    break
                pbar.update()
                output = model(data)
                output_softmax = F.softmax(output, 1)
                targets_var = Variable(maybe_cuda(torch.cat(target, 0), args.cuda), requires_grad=False)
                output_seg = output.data.cpu().numpy().argmax(axis=1)
                target_seg = targets_var.data.cpu().numpy()
                preds_stats.add(output_seg, target_seg)

                current_idx = 0

                for k, t in enumerate(target):
                    document_sentence_count = len(t)
                    to_idx = int(current_idx + document_sentence_count)

                    output = ((output_softmax.data.cpu().numpy()[current_idx: to_idx, :])[:, 1] > test_threshold)
                    h = np.append(output, [1])
                    tt = np.append(t, [1])

                    test_acc.update(h, tt)

                    current_idx = to_idx

        test_pk, epoch_windiff = test_acc.calc_accuracy()

        logger.debug('Testing validation section: {}, accuracy: {:.4}, Pk: {:.4}, Windiff: {:.4}, F1: {:.4} . '.format(epoch + 1,
                                                                                                          preds_stats.get_accuracy(),
                                                                                                          test_pk,
                                                                                                          epoch_windiff,
                                                                                                          preds_stats.get_f1()))
        preds_stats.reset()

        return test_pk


def main(args):
    sys.path.append(str(Path(__file__).parent))

    logger = utils.setup_logger(__name__,  'cross_validate_choi.log')

    utils.read_config_file(args.config)
    utils.config.update(args.__dict__)
    logger.debug('Running with config %s', utils.config)

    configure(os.path.join('runs', args.expname))

    if not args.test:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(utils.config['word2vecfile'], binary=True)
    else:
        word2vec = None


    dataset_path = Path(args.flat_choi)

    with open(args.load_from, 'rb') as f:
        model = torch.load(f)
    model.eval()
    model = maybe_cuda(model)

    test_accuracy = accuracy.Accuracy()

    for j in range(5):
        validate_folder_numbers = range(5)
        validate_folder_numbers.remove(j)
        validate_folder_names = [dataset_path.joinpath(str(num)) for num in validate_folder_numbers]
        dev_dataset = ChoiDataset(dataset_path , word2vec, folder=True, folders_paths=validate_folder_names)
        test_dataset = ChoiDataset(dataset_path, word2vec, folder=True, folders_paths=[dataset_path.joinpath(str(j))])

        dev_dl = DataLoader(dev_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                            num_workers=args.num_workers)
        test_dl = DataLoader(test_dataset, batch_size=args.test_bs, collate_fn=collate_fn, shuffle=False,
                             num_workers=args.num_workers)

        _, threshold = validate(model, args, j, dev_dl, logger)
        test_pk = test(model, args, j, test_dl, logger, threshold, test_accuracy)
        logger.debug(colored('Cross validation section {} with p_k {} and threshold {}'.format(j, test_pk, threshold),'green'))

    cross_validation_pk, _ = test_accuracy.calc_accuracy()
    print ('Final cross validaiton Pk is: ' + str(cross_validation_pk))
    logger.debug(
        colored('Final cross validaiton Pk is: {}'.format(cross_validation_pk), 'green'))



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--test', help='Test mode? (e.g fake word2vec)', action='store_true')
    parser.add_argument('--bs', help='Batch size', type=int, default=8)
    parser.add_argument('--test_bs', help='Batch size', type=int, default=5)
    parser.add_argument('--load_from', help='Location of a .t7 model file to load. Training will continue')
    parser.add_argument('--expname', help='Experiment name to appear on tensorboard', default='exp1')
    parser.add_argument('--stop_after', help='Number of batches to stop after', default=None, type=int)
    parser.add_argument('--config', help='Path to config.json', default='config.json')
    parser.add_argument('--window_size', help='Window size to encode setence', type=int, default=1)
    parser.add_argument('--num_workers', help='How many workers to use for data loading', type=int, default=0)
    parser.add_argument('--flat_choi', help='Path to flat choi dataset')


    main(parser.parse_args())
