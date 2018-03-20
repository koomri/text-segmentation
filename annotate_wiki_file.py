from argparse import ArgumentParser
from wiki_loader import read_wiki_file
import pandas as pd
from pathlib2 import Path
import os


def get_files(path):
    all_objects = Path(path).glob('**/*')
    files = (str(p) for p in all_objects if p.is_file())
    return files

def generate_segmentation_template(path, output_path):
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    sentences, _, _ = read_wiki_file(path, None, remove_preface_segment= True, return_as_sentences=True, ignore_list=True, remove_special_tokens = False)
    df = pd.DataFrame({ 'Cut here': [0] * len(sentences),'Sentences': sentences})
    df = df[['Cut here','Sentences']]

    df.to_excel(writer, sheet_name='segment')
    writer.save()


def generate_test_article(path, output_path):
    sentences, _, _ = read_wiki_file(path, None, remove_preface_segment= True, return_as_sentences=True, ignore_list=True, remove_special_tokens = False,
                                     high_granularity=False)
    article_text = "\n".join(sentences)
    with open(output_path, "w") as f:
        f.write(article_text.encode('utf-8'))
    f.close()

def generate_folder(input_folder,output_folder):
    counter = 0
    input_files = get_files(input_folder)
    for file in input_files:
        id = os.path.basename(file)
        file_name = id + ".xlsx" if not args.toText else id
        output_file = os.path.join(output_folder, file_name)
        if (args.toText):
            generate_test_article(file, output_file)
        else:
            generate_segmentation_template(file,output_file)
        counter += 1
    print 'generates ' + str(counter) + ' files'



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--path', help='input folder path', default='/home/michael/Downloads/migo/68943', type=str)
    parser.add_argument('--output_path', help='output folder path', default='blah.xlsx', type=str)
    parser.add_argument('--toText', help='output to text files ?', action='store_true')
    args = parser.parse_args()

    generate_folder(args.path,args.output_path)

