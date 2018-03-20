from pathlib2 import Path
import os

#root ='/home/adir/Projects/text-segmentation-2017/data/choi/'
root  = '/home/adir/Projects/text-segmentation-2017/data/choi/1/3-5'
output ='/home/adir/Projects/text-segmentation-2017/data/part_choi/'
delimeter = '=========='
truth = '********************************************'

textfiles = list(Path(root).glob('**/*.ref'))


counter = 0

for file in textfiles:
    counter += 1
    with file.open('r') as f:
        raw_text = f.read()
    new_text = raw_text.replace('==========',truth)
    f.close()
    new_file_path = os.path.join(output,str(counter) + "_" + os.path.basename(str(file)))
    with open(new_file_path, "w") as f:
        f.write(new_text)
    f.close()

print 'done'

