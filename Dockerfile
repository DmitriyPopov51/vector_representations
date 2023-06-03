from pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel


RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-get clean && apt-get update
RUN apt install -y wget git
# jupyter notebook settings
RUN mkdir src
WORKDIR src/
COPY . .

RUN mkdir models
RUN wget https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-base-4096.tar.gz 
RUN tar -C "models" -xzf longformer-base-4096.tar.gz
RUN wget https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-large-4096.tar.gz
RUN tar -C "models" -xzf longformer-large-4096.tar.gz
RUN mkdir notebooks
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install jupyter

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
