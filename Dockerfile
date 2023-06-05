FROM python:3.10.4-slim

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

RUN python setup.py install

CMD [ "python", "main.py" ]
# docker build -t toxicityclassifier .
# docker run -it toxicityclassifier