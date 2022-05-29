import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, render_template
import re

# this function finds books related to the entered choice and return 
# list of 13 books and their cover images inclusive of the choice book

def recommend(choice):

    # the try-except block checks whether count matrix is created or not, if not
    # then it will load it from the pickle files
    try:
        model.get_params()
    except:
        #reading from pickle files we created earlier in model.py
        count_matrix = pickle.load(open('count_matrix.pkl', 'rb'))
        model= pickle.load(open('model.pkl', 'rb'))
        data = pd.read_csv('final_data.csv')

    # If book name exactly matches with the name of book in the data's title column
    # then this block will be executed and return the book list and cover image array.

    if choice in data['title'].values:
        choice_index = data[data['title'] == choice].index.values[0]
        distances, indices = model.kneighbors(
            count_matrix[choice_index], n_neighbors=13)
        book_list = []  # to save the cover images
        photos = []  # to save the cover images
        for i in indices.flatten():
            book_list.append(data[data.index == i]
                             ['original_title'].values[0].title())
            photos.append(data[data.index == i]['coverImg'].values[0])
        return book_list, photos

    # If no any book name exactly matches with the title column of the data then, in this 
    # block of code I am finding book name which highly matches with book name entered 
    # by the user and and return the book list and cover image array relating to the book.

    elif (data['title'].str.contains(choice).any() == True):

        # getting list of similar book names as choice.
        similar_names = list(str(s) for s in data['title'] if choice in str(s))
        # sorting to get the most matched book name.
        similar_names.sort()
        # taking the first book from the sorted similar book name.
        new_choice = similar_names[0]
        print(new_choice)
        # getting index of the choice from the dataset
        choice_index = data[data['title'] == new_choice].index.values[0]
        # getting distances and indices of 13 mostly related books with the choice.
        distances, indices = model.kneighbors(
            count_matrix[choice_index], n_neighbors=13)
        # creating book list and cover images list
        book_list = []
        photos = []
        for i in indices.flatten():
            book_list.append(data[data.index == i]
                             ['original_title'].values[0].title())
            photos.append(data[data.index == i]['coverImg'].values[0])
        return book_list, photos

    # If no name matches then this else statement will be executed.
    else:
        return "opps! book not found in our database", "Try again with another"


app = Flask(__name__)

#rendering to the home page

@app.route("/")
def home():
    return render_template('index.html')


@app.route("/Search")
def search_books():
    # getting input from user
    choice = request.args.get('book')
    # removing all the characters except alphabets and numbers.
    choice = re.sub("[^a-zA-Z1-9]", "", choice).lower()
    # getting the books and photos array returned from recommend function
    books, photos = recommend(choice)
    # if rocommendation is a string and not list then it is else part of the
    # recommend() function.
    if type(books) == type('string'):
        return render_template('listpage.html', book=books, s='opps')
    else:
        #passing both book=[] as well as photo=[]
        return render_template('listpage.html', book=books, pics=photos)


if __name__ == "__main__":
    app.run(debug=False)
