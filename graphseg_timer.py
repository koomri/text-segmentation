import os
import subprocess
from timeit import default_timer as timer
import utils
from argparse import ArgumentParser


def main(input, output, jar_path, threshold, min_segment):
    # java -jar graphseg.jar /home/seg-input /home/seg-output 0.25 3


    # for min_segment in range(1, 11):
    #     for tresh in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
    #                   0.95]:

    output_folder = os.path.join(output,
                                 'graphseg_output_{}_{}'.format(min_segment, threshold))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    logger = utils.setup_logger(__name__, os.path.join(output_folder, 'graphseg_timer.log'), delete_old=True)

    beginning_comd = ['java', '-jar', jar_path, input]
    params = [str(threshold), str(min_segment)]
    cmd = beginning_comd + [output_folder] + params
    print cmd
    start = timer()
    # os.system(cmd)
    subprocess.call(cmd)
    end = timer()
    print 'it tooks seconds:'
    print end - start
    logger.info('running on parmas: ' + str(params[0]) + " , " + str(params[1]))
    logger.info('it tooks seconds:')
    logger.info(end - start)
    logger.info('\n')

    print ('done')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', help='input folder path',
                        default='/home/adir/Projects/data/wikipedia/wiki4_no_seperators', type=str)
    parser.add_argument('--output', help='output folder path',
                        default='/home/adir/Projects/data/wikipedia/wiki4_output_graphseg/', type=str)
    parser.add_argument('--jar', help='graphseg jar path path',
                        default='/home/adir/Projects/graphseg/binary/graphseg.jar', type=str)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--min_segment', type=int, required=True)

    args = parser.parse_args()

    main(args.input, args.output, args.jar, args.threshold, args.min_segment)
