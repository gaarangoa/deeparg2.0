FROM tensorflow/tensorflow:1.13.1-gpu-py3

ADD ./src /dplus/src

WORKDIR /dplus/src/
RUN pip install . --upgrade

RUN apt-get install wget

RUN wget https://github.com/facebookresearch/fastText/archive/v0.2.0.zip &&\
    unzip v0.2.0.zip &&\
    cd fastText-0.2.0 && \
    make && \
    cp fasttext /usr/bin/ &&\
    chmod +x /usr/bin/fasttext



# Download models from server @ bench.cs.vt

CMD [ "deepARG+" ]

# docker build .  -f ../DockerFile --force-rm -t deepargplus:1.0
# docker run --runtime=nvidia -it -v $PWD:/data/  --rm deepargplus:latest /bin/bash