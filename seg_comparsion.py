from argparse import ArgumentParser
import utils
from utils import maybe_cuda
import gensim
import torch
from torch.autograd import Variable
from test_accuracy_manifesto import ManifestoDataset
from wiki_loader import WikipediaDataSet
from choiloader import ChoiDataset, collate_fn, read_choi_file
from torch.utils.data import DataLoader
from test_accuracy import softmax
from wiki_loader import clean_section,split_sentences,section_delimiter,extract_sentence_words
import os
import sys


preds_stats = utils.predictions_analysis()
paragraphs_delimiter = "=="

def main(args):

    utils.read_config_file(args.config)


    if not args.test:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(utils.config['word2vecfile'], binary=True)
    else:
        word2vec = None

    with open(args.model, 'rb') as f:
        model = torch.load(f)
    model = maybe_cuda(model)
    model.eval()

    data_path = args.folder
    if (args.wiki):
        dataset = WikipediaDataSet(args.folder,word2vec,folder=True)
        delimeter = section_delimiter

    elif args.choi: #not in use
        dataset = ChoiDataset(args.folder, word2vec,is_cache_path=True)
        delimeter = paragraphs_delimiter
    else:
        print 'required dataset type'
        return

    dl = DataLoader(dataset,batch_size=1, collate_fn=collate_fn, shuffle=False)

    for i, (data, targets, paths) in enumerate(dl):
        doc_path = str(paths[0])
        output = model(data)
        targets_var = Variable(maybe_cuda(torch.cat(targets, 0), args.cuda), requires_grad=False)

        output_prob = softmax(output.data.cpu().numpy())
        output_seg = output_prob[:, 1] > 0.3
        target_seg = targets_var.data.cpu().numpy()
        preds_stats.add(output_seg, target_seg)

        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)

        result_file_path = os.path.join(args.output_folder,os.path.basename(doc_path))
        result_file = open(str(result_file_path ),"w")

        file = open(str(doc_path), "r")
        raw_content = file.read()
        file.close()
        sections = [clean_section(s) for s in raw_content.decode('utf-8').strip().split(delimeter) if len(s) > 0 and s != "\n"]

        sum_sentences = 0
        total_num_sentences = 0
        bad_sentences = 0

        for section in sections:
            sentences = split_sentences(section)
            if sentences:
                total_num_sentences += len(sentences)
                for i in range(0,len(sentences)):
                    sentence = sentences[i]
                    words = extract_sentence_words(sentence)
                    sentence = " ".join(words)

                    result_file.write(sentence.encode('utf-8'))

                    sys.stdout.flush()
                    result_file.write("\n".encode('utf-8'))
                    if (len(target_seg) == sum_sentences): ## last sentence
                        continue
                    if (target_seg[sum_sentences]):
                        result_file.write(delimeter.encode('utf-8'))
                        sys.stdout.flush()
                        result_file.write("\n".encode('utf-8'))
                    if (output_seg[sum_sentences]):
                        result_file.write("*******Our_Segmentation********".encode('utf-8'))
                        result_file.write("\n".encode('utf-8'))
                    sum_sentences += 1
        result_file.close()

        if ((total_num_sentences - bad_sentences) != (len(target_seg) + 1)): ## +1 last sentence segment doesn't counted
            print 'Pick another article'
            print 'len(targets) + 1= ' + str(len(target_seg) + 1)
            print 'total_num_sentences - bad_sentences= ' + str(total_num_sentences - bad_sentences)
        else :
            print 'finish comparsion'
            print 'result at ' + str(result_file_path )
            print ('F1: {:.4}.'.format(preds_stats.get_f1()))
            print ('Accuracy: {:.4}.'.format(preds_stats.get_accuracy()))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--test', help='Test mode? (e.g fake word2vec)', action='store_true')
    parser.add_argument('--model', help='Model to run - will import and run', required=True)
    parser.add_argument('--config', help='Path to config.json', default='config.json')
    parser.add_argument('--folder', help='folder with files to test on', required=True)
    parser.add_argument('--output_folder', help='folder for result', required=True)
    parser.add_argument('--wiki', help='if its wiki article', action='store_true')
    parser.add_argument('--manifesto', help='if its manifesto article', action='store_true')
    parser.add_argument('--choi', help='if its choi article', action='store_true')

    main(parser.parse_args())
