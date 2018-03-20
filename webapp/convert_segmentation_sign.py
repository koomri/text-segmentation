import utils
from pathlib2 import Path
from argparse import ArgumentParser
import os

# /home/adir/Projects/graphseg/data/output
# /home/adir/Projects/text-segmentation-2017/data/tested_articles/



graphseg_delimeter = "=========="
wiki_delimiter = "========"



def get_files(path):
    all_objects = Path(path).glob('**/*')
    files = (str(p) for p in all_objects if p.is_file())
    return files


def main(args):
    utils.read_config_file(args.config)
    utils.config.update(args.__dict__)


    files = get_files(args.folder)

    for file_path in files:
        file = open(str(file_path), "r")
        raw_content = file.read()
        file.close()
        result_file_path = os.path.join(args.output, os.path.basename(file_path))
        result_file = open(str(result_file_path), "w")

        sentences  = [s for s in raw_content.decode('utf-8').strip().split("\n") if len(s) > 0 and s != "\n"]

        for sentence in sentences:
            if sentence == graphseg_delimeter:
                result_file.write(wiki_delimiter.encode('utf-8'))
            else:
                result_file.write(sentence.encode('utf-8'))
            result_file.write("\n".encode('utf-8'))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--config', help='Path to config.json', default='../config.json')
    parser.add_argument('--folder', help='folder containing manifesto files to work on', required=True)
    parser.add_argument('--output', help='folder for output files', required=True)

    main(parser.parse_args())