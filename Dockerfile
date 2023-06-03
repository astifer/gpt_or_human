FROM python:3.8
COPY . /app
COPY requirements.txt /app/requirements.txt
WORKDIR /app
VOLUME /app/data
RUN pip install -r requirements.txt
RUN pip install catboost
RUN pip install simpletransformers
RUN python3 -c "import nltk; from nltk.tokenize import word_tokenize; import string; import catboost; from catboost import Pool, CatBoostClassifier; import os; import pandas as pd; from sentence_transformers import SentenceTransformer; from sklearn import svm; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2');"
RUN chmod +x /app/baseline.py
CMD ["python3","/app/baseline.py"]
