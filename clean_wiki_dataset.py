from pathlib2 import Path
import wiki_processor
from argparse import ArgumentParser

def remove_malicious_files(dataset_path):
    with open('malicious_wiki_files', 'r') as f:
        malicious_file_ids = f.read().splitlines()

    test_path = Path(dataset_path).joinpath(Path('test'))
    train_path = Path(dataset_path).joinpath(Path('train'))
    dev_path = Path(dataset_path).joinpath(Path('dev'))

    deleted_file_count = 0

    for id in malicious_file_ids:
        file_path_suffix = Path(wiki_processor.get_file_path(id)).joinpath(id)
        if test_path.joinpath(file_path_suffix).exists():
            test_path.joinpath(file_path_suffix).remove()
            deleted_file_count += 1

        elif train_path.joinpath(file_path_suffix).exists():
            train_path.joinpath(file_path_suffix).remove()
            deleted_file_count += 1

        elif dev_path.joinpath(file_path_suffix).exists():
            dev_path.joinpath(file_path_suffix).remove()
            deleted_file_count +=1

        else:
            raise Exception('meliciious file is not included in dataset: ' + str(id))

    print ('Deleted ' + str (deleted_file_count) + ' files. Malicious file count: ' + str(len(malicious_file_ids)))

def main(arg):
    remove_malicious_files(arg.path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', help='Path to dataset')

    main(parser.parse_args())



