import wiki_loader
import gensim
import evaluate
import utils
from pathlib2 import Path
from argparse import ArgumentParser
import torch
import choiloader
import numpy as np

goldset_delimiter = "********"

def segment(path, model, word2vec, output_folder, wiki = False):
    file_id = str(path).split('/')[-1:][0]
    if wiki:
        splited_sentences, target, _ = wiki_loader.read_wiki_file(path, None, remove_preface_segment= True, return_w2v_tensors = False)
    else:
        splited_sentences, target, _ = choiloader.read_choi_file(path, word2vec, False, False)

    sentences = [' '.join(s) for s in splited_sentences]
    gold_set = np.zeros(len(splited_sentences)).astype(int)
    gold_set[np.asarray(target)] = 1



    cutoffs = evaluate.predict_cutoffs(sentences, model, word2vec)
    total = []
    segment = []
    for i, (sentence, cutoff) in enumerate(zip(sentences, cutoffs)):
        segment.append(sentence)
        if cutoff or gold_set[i] == 1:
            full_segment ='.'.join(segment) + '.'
            if cutoff:
                full_segment = full_segment + '\n' + wiki_loader.section_delimiter + '\n'
                if gold_set[i] == 1:
                    full_segment = full_segment + goldset_delimiter + '\n'
            else:
                full_segment = full_segment + '\n' +  goldset_delimiter + '\n'
            total.append(full_segment)
            segment = []



    # Model does not return prediction for last sentence
    segment.append(sentences[-1:][0])
    total.append('.'.join(segment))

    output_file_content = "".join(total)
    output_file_full_path = Path(output_folder).joinpath(Path(file_id))
    with output_file_full_path.open('w') as f:
        f.write(output_file_content)

def main(args):
    utils.read_config_file(args.config)
    utils.config.update(args.__dict__)

    with Path(args.file).open('r') as f:
        file_names = f.read().strip().split('\n')

    if not args.test:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(utils.config['word2vecfile'], binary=True)
    else:
        word2vec = None

    with open(args.model, 'rb') as f:
        model = torch.load(f)
        model.eval()


    for name in file_names:
        if name:
            segment(Path(name), model, word2vec, args.output, wiki=args.wiki)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test', help='Test mode? (e.g fake word2vec)', action='store_true')
    parser.add_argument('--model', help='Model to run - will import and run', required=True)
    parser.add_argument('--config', help='Path to config.json', default='../config.json')
    parser.add_argument('--file', help='file containing file names to segment by model', required=True)
    parser.add_argument('--output', help='output folder', required=True)
    parser.add_argument('--wiki', help='use wikipedia files', action='store_true')


    main(parser.parse_args())