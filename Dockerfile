# Build the image using
# GPU: nvidia-docker build -t textseg .
# Normal: docker build -t textseg .

# How to run it
# (nvidia-)docker run -v $PWD:$PWD  -p 80:80 -it textseg
# `-p 80:80` : will help in running the flask app
# `-v $PWD:$PWD`: mounts the current folder inside the container with the same path so that your config files work


FROM library/python:2

# MAINTAINER Siddharth Yadav "siddharth16268@iiitd.ac.in"

RUN pip install http://download.pytorch.org/whl/cu80/torch-0.3.0-cp27-cp27mu-linux_x86_64.whl
RUN pip install cython numpy scipy gensim ipython jupyter tqdm pathlib2 segeval tensorboard_logger flask flask_wtf nltk pandas xlrd xlsxwriter termcolor

EXPOSE 80

CMD ["/bin/bash"]
