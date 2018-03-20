import os
from argparse import ArgumentParser
import subprocess
import re
from pathlib2 import Path
from random import shuffle,seed,uniform
import math
from shutil import  move
import utils
import wiki_utils
import text_manipulation
import wiki_thresholds
import json


logger = utils.setup_logger(__name__, 'processor_log.log', True )
doc_split_delimiter = "</doc>"
id_parts = 7
# minimal number of sentences in document (used to filter non informal documents such as https://en.wikipedia.org/wiki?curid=32283

seed(1234)

wikipedia_namespaces = ['Category', 'File', 'Ru', 'Wikipedia', 'Talk', 'User', 'MediaWiki', 'Template', 'Help', 'Portal', 'Book', 'Draft',
                         'Education Program', 'TimedText', 'Module', 'Gadget', 'Gadget definition', 'Media', 'Special']

disambigutaiton_pattern = '(disambiguation)'


global num_sentneces_for_avg
global sum_sentneces_for_avg
num_sentneces_for_avg = 0
sum_sentneces_for_avg = 0


def count_str_occurrences(str,findStr):

    return len(str.split(findStr)) - 1

def get_file_path(id):
    chopped_id = []
    id_str = str(id)
    padding_count = id_parts - len(id_str)
    while padding_count > 0:
        id_str = "0" + id_str
        padding_count-= 1

    for i in range(0,3):
        chopped_id.append(id_str[:2])
        id_str = id_str[2:]

    path = ""
    for sub_path in chopped_id:
        path =os.path.join(path, sub_path)
    return path

def process_header(header):
    id_match = re.search(r'<doc id="(\d+)" url', header)
    id = id_match.groups()[0]


    title_match = re.search(r'title="(.*)">', header)
    title = title_match.groups()[0]

    not_valid = title.isdigit() or any(title.startswith(prefix + ':' or prefix + ' talk:' ) for prefix in wikipedia_namespaces) or  title.endswith(disambigutaiton_pattern)

    return id, not not_valid

def get_sections(content):
    lines = content.split('\n')
    section = ""
    # sections include headers
    sections = []
    sections.append(wiki_utils.get_segment_seperator(1,"preface."))
    for line in lines:
        if (wiki_utils.is_seperator_line(line)):
            if len(section) > 0:
                sections.append(section)
            section = ""
            sections.append(line)

        else:
            section += line
            section += '\n'

    if len(section) > 0:
        sections.append(section)

    return sections



def process_section(section, id):
    global num_sentneces_for_avg
    global sum_sentneces_for_avg
    sentences = text_manipulation.split_sentences(section, id)
    section_sentences = []
    num_lists = 0
    num_sentences = 0
    num_formulas = 0
    num_codes = 0
    last_sentence_was_list = False
    for sentence in sentences:
        is_list_sentence = wiki_utils.get_list_token() + "." == sentence.encode('utf-8')
        if '\n' in sentence:
            logger.info("DocId: " + str(id) + "   back slash in sentence: " + sentence)
        if (wiki_utils.get_list_token() in sentence) and (wiki_utils.get_list_token() + ".") != sentence.encode('utf-8'):
            # TODO: delete this if section, since it is not suupposed to happen any more - but still happen
            num_lists += 1
            last_sentence_was_list = True
            logger.info("DocId: " + str(id) +  "     Special case 1: " + sentence)
            continue
        elif is_list_sentence:
            if (last_sentence_was_list):
                continue
            last_sentence_was_list = True
            num_lists += 1
        else:
            last_sentence_was_list = False
            sentence_words = text_manipulation.extract_sentence_words(sentence)
            if len(sentence_words) < wiki_thresholds.min_words_in_sentence:
                # ignore this sentence
                continue
            sum_sentneces_for_avg += len(sentence_words)
            num_sentneces_for_avg += 1


        num_formulas += count_str_occurrences(sentence, wiki_utils.get_formula_token())
        num_codes += count_str_occurrences(sentence, wiki_utils.get_codesnipet_token())
        num_sentences += 1
        section_sentences.append(sentence)


    valid_section = True
    error_message = None
    if (num_sentences < wiki_thresholds.min_sentence_in_section):
        valid_section = False
        error_message = "sentences count in section is too low"

    if (num_sentences > 0):
        lists_perentage = float(num_lists) / float(num_sentences)
        if lists_perentage >= wiki_thresholds.max_list_in_section_percentage:
            valid_section = False
            error_message = "list percentage in section is too high: " + str(lists_perentage)

    section_text =  ''.join(section_sentences)
    if len(re.findall('[a-zA-Z]', section_text)) < wiki_thresholds.min_section_char_count:
        valid_section = False
        error_message = "char count in section is too low"

    if num_formulas >= wiki_thresholds.max_section_formulas_count:
        valid_section = False
        error_message = "number of formulas in section is too high: " + str(num_formulas)

    if num_codes >= wiki_thresholds.max_section_code_snipet_count:
        valid_section = False
        error_message = "number of code snippets in section is too high: " + str(num_codes)


    return valid_section, section_sentences, error_message

def is_valid_article(valid_section_count, section_count):
    if valid_section_count < wiki_thresholds.min_valid_section_count:
        return False, "Valid section count is too low: " + str(valid_section_count)

    valid_section_percentage = float(valid_section_count) / float (section_count)
    if valid_section_percentage < wiki_thresholds.min_valid_section_percentage:
        return False, "Valid section percentage is too low: " + str(valid_section_percentage)


    return True,""



def max_level_in_article(content):
    max_lavel = -1
    for line in content:
        if (wiki_utils.is_seperator_line(line)):
            current_level = wiki_utils.get_segment_level(line)
            if current_level > max_lavel:
                max_lavel = current_level
    return max_lavel


def delete_empty_segment_headers(content):
    num_of_deletions = 0
    max_level = max_level_in_article(content)
    for handle_level in range(max_level,0,-1):
        last_section_level = -1
        last_section_header = True
        for i in range(len(content) -1 , -1 , -1):
            section = content[i]
            if (wiki_utils.is_seperator_line(section)):
                section_level = wiki_utils.get_segment_level(section)
                if (section_level == handle_level):

                    # empty section if last seciont was also a header
                    is_empty =  last_section_header
                    if (is_empty &  (last_section_level <=  section_level)):
                        del content[i]
                        num_of_deletions += 1
                last_section_level = section_level
                last_section_header = True
            else:
                last_section_header = False

    return content, num_of_deletions


def vec_to_text(sections_with_headers):
    adjusted_content = ""
    for section in sections_with_headers:
        adjusted_content += section + '\n'
    return adjusted_content


def process_content(content, id):
    sections_with_headers = get_sections(content)
    adjueted_content_text = ""
    article_lines = []
    section_count = 0
    valid_section_count = 0
    for i in range(len(sections_with_headers)):
        section = sections_with_headers[i]
        if wiki_utils.is_seperator_line(section):
            article_lines.append(section)
        else:
            is_valid_section, section_sentences, message = process_section(section, id)
            section_count += 1
            if (is_valid_section):
                valid_section_count += 1
                article_lines.extend(section_sentences)
            else:
                logger.info('Invalid section in article id: ' + id +
                            '    Reason: ' + message + '    Content: ' + vec_to_text(section_sentences).strip('\n') )

    is_valid,reason = is_valid_article(valid_section_count, section_count )

    if is_valid:
        article_content,_ = delete_empty_segment_headers(article_lines)
        adjueted_content_text = vec_to_text(article_content)


    return is_valid, adjueted_content_text,reason


# old process content, for comparsion
# def process_content(content):
#
#     # keep only scetions with minimal number of characters
#     sections = [s.strip('\n') for s in content.strip().split(section_delimiter) if
#                 len(re.findall('[a-zA-Z]', s)) > min_section_length]
#
#     # article must have at least 3 sections, to avoid articles with only one section which is summaization. E.g:
#     # https://en.wikipedia.org/wiki?curid=821470
#     sections_count = len(sections)
#     if sections_count < min_article_sections_count or sections_count >= max_article_sections_count:
#         return content, False, 'Sections count is: ' + str(sections_count)
#
#     # remove first section since it usually the summary of the whole article
#     adjueted_content = ('\n' + section_delimiter + '\n').join(sections[1:])
#
#     return adjueted_content, True, ""



def process_article(article):
    non_empty_lines  = [l for l in article.strip().split("\n") if l != ""]
    header = non_empty_lines[0]
    id, is_valid_header = process_header(header)

    if not is_valid_header:
        logger.info('Invalid header in doc id: ' + str(id)+ '     header:   ' +  header)
        return "", id, False

    content = "\n".join(non_empty_lines[2:])
    is_valid_content, processed_content , debug = process_content(content, id)
    if not(is_valid_content):
        logger.info('Invalid article in doc id: ' + str(id) + '.  ' + debug +'\n\n')
    else:
        logger.info('Valid article , id: ' + str(id) +'\n\n')

    return processed_content, id, is_valid_content


def process_wiki_file(path, output_folder,train_ratio,test_ratio, forbidden_train_ids):
    train_size = 0
    dev_size = 0
    test_size = 0
    with open(path, "r") as file:
        raw_content = file.read()

    articles = [s for s in raw_content.decode('utf-8').strip().split(doc_split_delimiter) if len(s) > 0]
    created_articles_count = 0
    processed_articles_count = 0

    for article in articles:
        processed_article, id, is_valid = process_article(article)
        processed_articles_count+=1
        if not is_valid:
            continue;
        random_num = uniform(0, 1)
        if (random_num > train_ratio and  random_num <= train_ratio + test_ratio) or int(id) in forbidden_train_ids:
            partition = "test"
            test_size += 1
        elif (random_num >  train_ratio + test_ratio):
            partition = "dev"
            dev_size += 1
        else:
            partition = "train"
            train_size += 1
        output_sub_folder = os.path.join(output_folder,partition, get_file_path(id))
        if not os.path.exists(output_sub_folder):
            os.makedirs(output_sub_folder)
        output_file_path = os.path.join(output_sub_folder, str(id))
        with open(output_file_path, "w") as output_file:
            output_file.write(processed_article.encode('utf-8'), )
        created_articles_count+=1

    return created_articles_count, processed_articles_count, train_size,dev_size,test_size


def get_forbidden_train_ids():
    # Return ids of article which must be in test set (and not train/dev)
    with open('wikicities_article_names_to_ids') as f:
        wiki_cities = json.load(f)

    with open('wikielements_article_names_to_ids') as f:
        wiki_elements = json.load(f)

    forbidden_train_ids = []
    for k,v in wiki_cities.iteritems():
        forbidden_train_ids.append(int(v))
    for k,v in wiki_elements.iteritems():
        forbidden_train_ids.append(int(v))

    unique_ids = set(forbidden_train_ids)

    return unique_ids;



def get_wiki_files(path):
    all_objects = Path(path).glob('**/*')
    files = (str(p) for p in all_objects if p.is_file())
    return files


def process_wiki_folder(input_folder, output_folder,train_ratio,test_ratio):
    total_train_size = 0
    total_dev_size = 0
    total_test_size = 0
    folders =  [o for o in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, o))]
    total_created_articles = 0
    total_processed_articles = 0
    previous_debug = 0
    forbidden_train_ids = get_forbidden_train_ids()
    for folder in folders:
        full_folder_path = os.path.join(input_folder, folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        files = get_wiki_files(full_folder_path)
        for file in files:
            created_articles,  processed_articles, train_size, dev_size, test_size = process_wiki_file(file,  output_folder, float(train_ratio), float(test_ratio), forbidden_train_ids)
            total_train_size += train_size
            total_dev_size += dev_size
            total_test_size += test_size
            total_created_articles += created_articles
            total_processed_articles += processed_articles
            if (total_created_articles - previous_debug > 2500):
                previous_debug = total_created_articles
                print ('Created ' + str(total_created_articles) + ' wiki articles, out of ' + str(total_processed_articles) + ' processed articles')
    total_samples = total_train_size + total_dev_size + total_test_size
    print 'total_samples = ', str(total_samples)
    print "#train = ",total_train_size,"ratio: ","{:.2f}".format(total_train_size / float(total_samples))
    print "#dev = ", total_dev_size,"ratio: ","{:.2f}".format(total_dev_size/ float(total_samples))
    print "#test = ", total_test_size,"ratio: ","{:.2f}".format(total_test_size / float(total_samples))


def move_wiki_file(src,  folder, partition):
    # get relative path to inputFolder
    file = os.path.relpath(src, folder)

    # extract file path in train folder
    dstFile = os.path.join(folder, partition, file)
    dstdir = os.path.dirname(dstFile)
    if not os.path.exists(dstdir):
        os.makedirs(dstdir)
    move(src, dstFile)


def removeEmptyFolders(path, removeRoot=True):
    if not os.path.isdir(path):
        return

    # remove empty subfolders
    files = os.listdir(path)
    for f in files:
        fullpath = os.path.join(path, f)
        if os.path.isdir(fullpath):
            removeEmptyFolders(fullpath)

    # if folder empty, delete it
    files = os.listdir(path)
    if len(files) == 0 and removeRoot:
        #print "Removing empty folder:", path
        os.rmdir(path)



def trainTestDev(destFolder, train_size, test_size):
    train_size_ratio = float(train_size)
    test_size_ratio = float(test_size)
    dev_size_ratio = 1 - train_size_ratio - test_size_ratio

    print (destFolder,train_size,test_size)

    allFiles = []
    if not os.path.exists(destFolder):
        print ("Output folder does not exist")
        return
    folders =  [o for o in os.listdir(destFolder) if os.path.isdir(os.path.join(destFolder, o))]
    for folder in folders:
        full_folder_path = os.path.join(destFolder, folder)
        files = get_wiki_files(full_folder_path)
        allFiles.extend(files)


    shuffle(allFiles)

    trainSize = int(math.floor(len(allFiles) * train_size_ratio))
    devSize = int(math.floor(len(allFiles) * dev_size_ratio))
    for i in range(0,trainSize):
        move_wiki_file(allFiles[i], destFolder, partition="train")

    if devSize > 0:
        for i in range(trainSize, trainSize + devSize):
            move_wiki_file(allFiles[i], destFolder, partition="dev")

    for i in range(trainSize + devSize,len(allFiles)):
        move_wiki_file(allFiles[i], destFolder, partition="test")
    print ("#train = ",trainSize)
    print ("#dev = ", devSize)
    print ("#test = ", len(allFiles) - trainSize  -devSize)

    removeEmptyFolders(destFolder)


def main (args):
    global num_sentneces_for_avg
    global sum_sentneces_for_avg
    if not os.path.exists(args.temp):
        os.makedirs(args.temp)
    # execute extraction of wikipedia dump
    cmd = ['python', str(Path(__file__).parent / 'wiki_extractor.py'), '-s', '-o', args.temp, '--article_count', str(args.article_count),'--lists']
    print cmd

    if args.processes:
        cmd += ['--processes', args.processes]

    cmd += [args.input]

    if not args.no_extractor: 
        subprocess.call(cmd)
        print ("Finisehd extractor")



    if not os.path.exists(args.output):
        os.makedirs(args.output)
    # create file per each wiki value from the extracted dump
    process_wiki_folder(args.temp, args.output,args.train, args.test)

    print ("Number of processed sentences: " +  str(num_sentneces_for_avg))
    print "avg len sentence = " + str(sum_sentneces_for_avg / float(num_sentneces_for_avg))
    print ('done')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', help='Path to wikipedia dump', required=True)
    parser.add_argument('--processes', help='Num of processors to use in wiki_extractor')
    parser.add_argument('--no_extractor', help='Skip wiki-extractor', action='store_true')
    parser.add_argument('--temp', help='folder to save temporal files', required=True)
    parser.add_argument('--output', help='output folder', required=True)
    parser.add_argument('--train', help='train size ratio', required=True)
    parser.add_argument('--test', help='test size ratio', required=True)
    parser.add_argument("--article_count", help = 'max number of wikipedia articles to extract', default=1000000)
    main(parser.parse_args())

