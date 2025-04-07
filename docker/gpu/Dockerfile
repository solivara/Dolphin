FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
LABEL authors="solivara"

WORKDIR /workspace

RUN sed -i "s@http://.*archive.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list && \
    sed -i "s@http://.*security.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3-pip ffmpeg git wget unzip build-essential python3-dev

RUN pip3 config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

RUN pip3 install --no-cache-dir dataoceanai-dolphin torchaudio

RUN wget -O nltk_data.zip https://ghfast.top/https://github.com/nltk/nltk_data/archive/gh-pages.zip && \
    unzip nltk_data.zip && \
    mv nltk_data-gh-pages nltk_data && \
    rm nltk_data.zip

ENV NLTK_DATA /workspace/nltk_data/packages

RUN git clone https://github.com/solivara/dolphin.git

WORKDIR /workspace/dolphin

RUN pip3 install --no-cache-dir -r /workspace/dolphin/requirements-api.txt

EXPOSE 50050

CMD ["python3", "/workspace/dolphin/api.py"]