#!/bin/bash
for i in 0.2  0.25  0.3  0.35  0.4  0.45  0.5
do
            python graphseg_timer.py --input ~/Downloads/wiki_dev_100_np_seperators --output ~/Downloads/wiki_dev_100_np_seperators_output --jar graphseg.jar --threshold $i --min_segment $1
done