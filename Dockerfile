FROM nvidia/cuda:12.2.0-base-ubuntu22.04 as base


ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /root/

# Install necessary dependencies
RUN apt update && \
    apt install -y python3-pip \
    wget \
    libcudnn8 \
    libcudnn8-dev

RUN wget https://developer.download.nvidia.com/compute/cuda/redist/libcublas/linux-x86_64/libcublas-linux-x86_64-12.2.4.5-archive.tar.xz  \
 && tar -Jxvf libcublas-linux-x86_64-12.2.4.5-archive.tar.xz  \ 
 && cp libcublas-linux-x86_64-12.2.4.5-archive/lib/libcublas.so.12* /usr/local/cuda/lib64/  \
 && cp libcublas-linux-x86_64-12.2.4.5-archive/lib/libcublasLt.so.12* /usr/local/cuda/lib64/ \
 && wget https://developer.download.nvidia.com/compute/cuda/redist/libcublas/linux-x86_64/libcublas-linux-x86_64-11.11.3.6-archive.tar.xz \
 && tar -Jxvf libcublas-linux-x86_64-11.11.3.6-archive.tar.xz \
 && cp libcublas-linux-x86_64-11.11.3.6-archive/lib/libcublas.so.11* /usr/local/cuda/lib64/ \
 && cp libcublas-linux-x86_64-11.11.3.6-archive/lib/libcublasLt.so.11* /usr/local/cuda/lib64/

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64/

FROM base as whisper

COPY requirements.txt /root/
COPY prepare_base.py /root/
COPY prepare_medium.py /root/
COPY prepare_large.py /root/

RUN pip3 install -r requirements.txt


RUN python3 prepare_base.py && python3 prepare_medium.py && python3 prepare_large.py


