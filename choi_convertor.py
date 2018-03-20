import os
from pathlib2 import Path
from argparse import ArgumentParser
from shutil import  move



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



def convert_choi_to_bySegLength(path):
    folders =  [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]

    for folder in folders:
        full_folder_path = os.path.join(path, folder)
        seg_folders = [o for o in os.listdir(full_folder_path ) if os.path.isdir(os.path.join(full_folder_path , o))]
        for seg_folder in seg_folders:
            full_seg_folder_path = os.path.join(full_folder_path ,seg_folder )
            convertedPathList = full_seg_folder_path.split(os.sep)


            convertedPath = os.path.sep.join(convertedPathList[:-2] + [convertedPathList[-1]] + [convertedPathList[-2]])
            if not os.path.exists(convertedPath):
                os.makedirs(convertedPath)
            all_objects = Path(full_seg_folder_path).glob('**/*')
            files = (str(p) for p in all_objects if p.is_file())
            for file in files:
                target = os.path.join(convertedPath ,os.path.basename(file) )
                move(file,target)
            print "Removing empty folder: ", full_seg_folder_path
            removeEmptyFolders(full_seg_folder_path)



def main (args):


    convert_choi_to_bySegLength(args.input)

    print ('done')

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--input', help='Path to choi dataset', required=True)
    main(parser.parse_args())

