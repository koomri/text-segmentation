from argparse import ArgumentParser
from wiki_loader import read_wiki_file
import pandas as pd
import accuracy
from annotate_wiki_file import get_files
import os
from glob import glob


graphseg_delimeter = "=========="


def generate_segmentation_template(path, output_path):
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    sentences, _, _ = read_wiki_file(path, None, False)

    sentences = [' '.join(s) + '.' for s in sentences]
    df = pd.DataFrame({'Sentences': sentences, 'Cut here': [0] * len(sentences)})
    df = df[['Sentences', 'Cut here']]

    df.to_excel(writer, sheet_name='segment')
    writer.save()


def target_place_to_list(targets):
    list_of_targets = []
    for i in range(targets[-1] + 1):
        if i in targets:
            list_of_targets.append(1)
        else:
            list_of_targets.append(0)

    list_of_targets[-1] = 1
    return list_of_targets


def get_graphseg_segments(file_path):
    file = open(str(file_path), "r")
    raw_content = file.read()
    file.close()
    sentences = [s for s in raw_content.decode('utf-8').strip().split("\n") if len(s) > 0 and s != "\n"]
    sentences_length = []
    h = []
    t = []

    for sentence in sentences:
        if sentence == graphseg_delimeter:
            if len(h) > 0:
                h[-1] = 1
        else:
            h.append(0)
        #words = extract_sentence_words(sentence)
        #sentences_length.append(len(words))
        #t.append(0)
        #h.append(0)

    #t[-1] = 1  # end of last segment
    h[-1] = 1  # they already segment it correctly.

    return h


def get_xlsx_segments(xlsx_path):
    df = pd.read_excel(xlsx_path)
    outputs = df['Cut here'].values
    outputs[-1] = 1
    return outputs


def get_gold_segments(path):
    sentences, targets, _ = read_wiki_file(path, None, remove_preface_segment= True, return_as_sentences=True, ignore_list=True, remove_special_tokens = False,high_granularity=False)

    return target_place_to_list(targets)


def get_sub_folders_for_graphseg(folder):
    d = folder
    folders = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]
    print folders
    return folders


def analyszie_folder(wiki_folder,xlsx_folder,isGraphseg, use_xlsx_sub_folders = False):

    acc = accuracy.Accuracy()

    input_files = get_files(wiki_folder)
    if use_xlsx_sub_folders:
        annotated_files_folders= []
        for f in os.listdir(xlsx_folder):
            sub_folder_path = xlsx_folder  + f
            if os.path.isdir(sub_folder_path):
                annotated_files_folders.append(sub_folder_path)
    else:
        annotated_files_folders = [xlsx_folder]




    for file in input_files:
        id = os.path.basename(file)
        file_name = id + ".xlsx" if not isGraphseg else id
        xlsx_file_paths = [os.path.join(xlsx_folder,file_name) for xlsx_folder in annotated_files_folders]
        print str(xlsx_file_paths)
        print str(file)

        for xlsx_file_path in xlsx_file_paths:
            if os.path.isfile(xlsx_file_path):
                if (isGraphseg):
                    tested_segments = get_graphseg_segments(xlsx_file_path)
                else:
                    tested_segments = get_xlsx_segments(xlsx_file_path )
            else:
                tested_segments = None

            gold_segments = get_gold_segments(file)
            if (tested_segments is not  None) and (len(tested_segments) != len(gold_segments)):
                print "(len(tested_segments) != len(gold_segments))"
                print "stop run"
                return 1000,1000
            if tested_segments is not None :
                acc.update(tested_segments,gold_segments)


    #Print results:
    calculated_pk, calculated_windiff = acc.calc_accuracy()
    print('Finished testing.')
    print ('Pk: {:.4}.'.format(calculated_pk))
    print ('')

    return calculated_pk,calculated_windiff


def result_to_file(pk_list,wd_list,path_list,result_file_path):
    writer = pd.ExcelWriter(result_file_path, engine='xlsxwriter')

    df = pd.DataFrame({ 'pk': pk_list,'wd': wd_list,'folders': path_list})
    df = df[['pk','wd','folders']]

    df.to_excel(writer, sheet_name='annotated_result')
    writer.save()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--path', help='wiki folder, truth', type=str)
    parser.add_argument('--xlsx_path', help='folder with xlsx files',  type=str)
    parser.add_argument('--graphseg', help='to calc graphseg pk', action='store_true')

    args = parser.parse_args()
    pk_list = []
    wd_list = []
    path_list = []

    if (args.graphseg):
        graphseg_folders = get_sub_folders_for_graphseg(args.xlsx_path)
        for folder in graphseg_folders:
            pk,wd =  analyszie_folder(args.path,folder,args.graphseg)
            pk_list.append(pk)
            wd_list.append(wd)
            path_list.append(folder)
    else:
        pk, wd = analyszie_folder(args.path, args.xlsx_path, args.graphseg, use_xlsx_sub_folders=True)
        pk_list.append(pk)
        wd_list.append(wd)
        path_list.append(args.xlsx_path)

    #writing result to file
    result_to_file(pk_list,wd_list,path_list,os.path.join(args.xlsx_path,"result_pk.xlsx") )
