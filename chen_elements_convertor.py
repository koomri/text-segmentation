import utils
from argparse import ArgumentParser
import os
import wiki_utils


def main(args):
    utils.read_config_file(args.config)
    utils.config.update(args.__dict__)

    file_path = args.input
    segments_path = args.segment
    output_folder_path = args.output


    file = open(str(segments_path), "r")
    segments_content = file.read()
    file.close()

    file = open(str(file_path ), "r")
    raw_content = file.read()
    file.close()

    sentences  = [s for s in raw_content.decode('utf-8').strip().split("\n") if len(s) > 0 and s != "\n"]
    segments = [s for s in segments_content.decode('utf-8').strip().split("\n") if len(s) > 0 and s != "\n"]

    result_file_path = None

    last_doc_id = 0
    last_topic = ""

    if (len(sentences) != len(segments)):
        print "len(sentences) != len(segments)"
        return

    for i in range(len(sentences)) :

        sentence = sentences[i]
        segment = segments[i].encode('utf-8').split("\r")[0]

        first_comma_index = segment.index(',')
        second_comma_index = segment[first_comma_index + 1 :].index(',')
        current_doc_id = segment[0:first_comma_index]
        current_topic = segment[first_comma_index + second_comma_index + 2:]

        if (current_doc_id != last_doc_id):
            last_doc_id = current_doc_id
            print 'new file index'
            print last_doc_id
            if (result_file_path != None):
                result_file.close()
            result_file_path = os.path.join(output_folder_path ,str(current_doc_id) + ".text")
            result_file = open(str(result_file_path), "w")
            last_topic = ""

        if (current_topic != last_topic):
            last_topic = current_topic
            level = 1 if (current_topic == "TOP-LEVEL SEGMENT") else 2
            result_file.write((wiki_utils.get_segment_seperator(level ,current_topic) +".").encode('utf-8'))
            result_file.write("\n".encode('utf-8'))

        actual_sentence = sentence
        result_file.write(actual_sentence.encode('utf-8'))
        if  ('\n' in sentence):
            print 'back slash in sentnece'
        #result_file.write(".".encode('utf-8'))
        result_file.write("\n".encode('utf-8'))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--config', help='Path to config.json', default='config.json')
    parser.add_argument('--input', help='Chen text file', required=True)
    parser.add_argument('--segment', help='regina segmentation file', required=True)
    parser.add_argument('--output', help='folder for converted files', required=True)

    main(parser.parse_args())