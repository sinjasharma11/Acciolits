#this python code pickles the countermatrix and knn model which is created as cosine similarity 

import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, render_template
import re


# we will import dataset, create count matrix and create similarity score matrix

data = pd.read_csv('final_data.csv')
    # create count matrix
cv = CountVectorizer()
count_matrix = cv.fit_transform(data['combined_features'])
    # create similarity score matrix
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(count_matrix)

#dumping the created models into pkl files which we will use in our recommendation system file

pkl.dump(model, open("model.pkl", "wb"))
pkl.dump(count_matrix, open("count_matrix.pkl", "wb"))


