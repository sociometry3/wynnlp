# FROM sanicframework/sanic:3.8-latest

# WORKDIR /sanic

# COPY . .

# FROM tensorflow/tensorflow:1.15.0

# COPY . .
FROM python:3.7

WORKDIR /app

COPY . .

RUN pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple

# RUN pip install --upgrade pip

RUN pip install sanic==23.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install tensorflow==1.15.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install protobuf==3.19.6 -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install nltk==3.8.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install emoji==1.7

RUN pip install recognizers-text-date-time

# RUN pip --default-timeout=100 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# COPY ./assets /tmp

# RUN pip install --upgrade /tmp/tensorflow-1.15.0-cp37-cp37m-win_amd64.whl

EXPOSE 52002

CMD ["python", "server.py"]