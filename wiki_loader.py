from torch.utils.data import Dataset
from text_manipulation import word_model
from text_manipulation import extract_sentence_words
from pathlib2 import Path
import re
import wiki_utils
import os

import utils

logger = utils.setup_logger(__name__, 'train.log')

section_delimiter = "========"


def get_files(path):
    all_objects = Path(path).glob('**/*')
    files = [str(p) for p in all_objects if p.is_file()]
    return files


def get_cache_path(wiki_folder):
    cache_file_path = wiki_folder / 'paths_cache'
    return cache_file_path


def cache_wiki_filenames(wiki_folder):
    files = Path(wiki_folder).glob('*/*/*/*')
    cache_file_path = get_cache_path(wiki_folder)

    with cache_file_path.open('w') as f:
        for file in files:
            f.write(unicode(file) + u'\n')


def clean_section(section):
    cleaned_section = section.strip('\n')
    return cleaned_section


def get_scections_from_text(txt, high_granularity=True):
    sections_to_keep_pattern = wiki_utils.get_seperator_foramt() if high_granularity else wiki_utils.get_seperator_foramt(
        (1, 2))
    if not high_granularity:
        # if low granularity required we should flatten segments within segemnt level 2
        pattern_to_ommit = wiki_utils.get_seperator_foramt((3, 999))
        txt = re.sub(pattern_to_ommit, "", txt)

        #delete empty lines after re.sub()
        sentences = [s for s in txt.strip().split("\n") if len(s) > 0 and s != "\n"]
        txt = '\n'.join(sentences).strip('\n')


    all_sections = re.split(sections_to_keep_pattern, txt)
    non_empty_sections = [s for s in all_sections if len(s) > 0]

    return non_empty_sections


def get_sections(path, high_granularity=True):
    file = open(str(path), "r")
    raw_content = file.read()
    file.close()

    clean_txt = raw_content.decode('utf-8').strip()

    sections = [clean_section(s) for s in get_scections_from_text(clean_txt, high_granularity)]

    return sections


def read_wiki_file(path, word2vec, remove_preface_segment=True, ignore_list=False, remove_special_tokens=False,
                   return_as_sentences=False, high_granularity=True,only_letters = False):
    data = []
    targets = []
    all_sections = get_sections(path, high_granularity)
    required_sections = all_sections[1:] if remove_preface_segment and len(all_sections) > 0 else all_sections
    required_non_empty_sections = [section for section in required_sections if len(section) > 0 and section != "\n"]

    for section in required_non_empty_sections:
        sentences = section.split('\n')
        if sentences:
            for sentence in sentences:
                is_list_sentence = wiki_utils.get_list_token() + "." == sentence.encode('utf-8')
                if ignore_list and is_list_sentence:
                    continue
                if not return_as_sentences:
                    sentence_words = extract_sentence_words(sentence, remove_special_tokens=remove_special_tokens)
                    if 1 <= len(sentence_words):
                        data.append([word_model(word, word2vec) for word in sentence_words])
                    else:
                        #raise ValueError('Sentence in wikipedia file is empty')
                        logger.info('Sentence in wikipedia file is empty')
                else:  # for the annotation. keep sentence as is.
                    if (only_letters):
                        sentence = re.sub('[^a-zA-Z0-9 ]+', '', sentence)
                        data.append(sentence)
                    else:
                        data.append(sentence)
            if data:
                targets.append(len(data) - 1)

    return data, targets, path


class WikipediaDataSet(Dataset):
    def __init__(self, root, word2vec, train=True, manifesto=False, folder=False, high_granularity=False):

        if (manifesto):
            self.textfiles = list(Path(root).glob('*'))
        else:
            if (folder):
                self.textfiles = get_files(root)
            else:
                root_path = Path(root)
                cache_path = get_cache_path(root_path)
                if not cache_path.exists():
                    cache_wiki_filenames(root_path)
                self.textfiles = cache_path.read_text().splitlines()

        if len(self.textfiles) == 0:
            raise RuntimeError('Found 0 images in subfolders of: {}'.format(root))
        self.train = train
        self.root = root
        self.word2vec = word2vec
        self.high_granularity = high_granularity

    def __getitem__(self, index):
        path = self.textfiles[index]

        return read_wiki_file(Path(path), self.word2vec, ignore_list=True, remove_special_tokens=True,
                              high_granularity=self.high_granularity)

    def __len__(self):
        return len(self.textfiles)
