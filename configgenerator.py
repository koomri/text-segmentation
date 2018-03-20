import json

jsondata = {
    "word2vecfile": "/home/omri/datasets/word2vec/GoogleNews-vectors-negative300.bin",
    "choidataset": "/home/omri/code/text-segmentation-2017/data/choi",
    "wikidataset": "/home/omri/datasets/wikipedia/process_dump_r",
}

with open('config.json', 'w') as f:
    json.dump(jsondata, f)
