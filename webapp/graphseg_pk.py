import utils
from pathlib2 import Path
from argparse import ArgumentParser
from text_manipulation import extract_sentence_words
import accuracy
from convert_seperator import truth

# /home/adir/Projects/graphseg/data/output
# /home/adir/Projects/text-segmentation-2017/data/tested_articles/


graphseg_delimeter = "=========="




def get_files(path):
    all_objects = Path(path).glob('**/*')
    files = (str(p) for p in all_objects if p.is_file())
    return files


def main(args):
    utils.read_config_file(args.config)
    utils.config.update(args.__dict__)

    algo_delimeter = graphseg_delimeter


    files = get_files(args.folder)
    acc = accuracy.Accuracy()

    for file_path in files:
        file = open(str(file_path), "r")
        raw_content = file.read()
        file.close()
        sentences  = [s for s in raw_content.decode('utf-8').strip().split("\n") if len(s) > 0 and s != "\n"]
        sentences_length = []
        h = []
        t = []
        is_first_sentence = True
        for sentence in sentences:
            if sentence == truth :
                if not is_first_sentence:
                    t[-1] = 1
                continue
            if sentence == algo_delimeter:
                if not is_first_sentence:
                    h[-1] = 1
                continue
            words = extract_sentence_words(sentence)
            sentences_length.append(len(words))
            t.append(0)
            h.append(0)
            is_first_sentence = False
        t[-1] = 1 # end of last segment
        h[-1] = 1 # they already segment it correctly.

        acc.update(h, t)


    calculated_pk, calculated_windiff = acc.calc_accuracy()
    print 'Pk: {:.4}.'.format(calculated_pk)
    print 'Win_diff: {:.4}.'.format(calculated_windiff)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--config', help='Path to config.json', default='../config.json')
    parser.add_argument('--folder', help='folder containing files which segmented by Graphseg', required=True)
    #parser.add_argument('--graphseg', help='to calc graphseg pk', action='store_true')


    main(parser.parse_args())