FROM ubuntu:22.04

LABEL authors="solivara"

WORKDIR /workspace

RUN sed -i "s@http://.*archive.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list && \
    sed -i "s@http://.*security.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3-pip ffmpeg git wget unzip build-essential python3-dev

RUN pip3 config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

RUN pip3 install --no-cache-dir dataoceanai-dolphin torchaudio

RUN wget -O nltk_data.zip https://github.com/nltk/nltk_data/archive/gh-pages.zip && \
    unzip nltk_data.zip && \
    mv nltk_data-gh-pages nltk_data && \
    rm nltk_data.zip

RUN modelscope download --model DataoceanAI/dolphin-small --local_dir /workspace/models/DataoceanAI/dolphin-small

ENV NLTK_DATA /workspace/nltk_data/packages

ENV device "cpu"

ENV port 50050

RUN git clone https://github.com/solivara/dolphin.git

WORKDIR /workspace/dolphin

RUN pip3 install --no-cache-dir -r /workspace/dolphin/requirements-api.txt

EXPOSE 50050

CMD ["python3", "/workspace/dolphin/api.py", "--device", "$device", "--port", "$port"]