# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:15:34 2020

@author: User
"""

import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score


def clean_data(data):
    #convert labels with criticsm surprise and anger to attitude and Emotions
    #conver labels containing the word requests or action to a new general label requests
    for index, row in data.iterrows():
        if row['Final Label'] == 'criticism' or row['Final Label'] == 'surprise' or row['Final Label'] == 'anger':
            row['Final Label'] = 'Attitudes and Emotions'
            
        if 'request' in row['Final Label'] or row['Final Label'] == 'action':
            row['Final Label'] = 'request'
    
    #Extract questions with multiple questions by the number of '?' in the question
    for index, row in data.iterrows():
        parag = row['Question']
        question_list = parag.split('?')
        question_list.pop()
        question_list = [x + '?' for x in question_list]
        size = len(question_list)
        
        if size>1:  #if the question has more than 1 question mark
            for i in range(len(question_list)): #for each question in question list add the appropriate label 
                question_list[i] = {"Question":question_list[i], "Final Label":data.iloc[index+i]['Final Label']} 
            
            #append the new question to the dataframe
            for i in range(len(question_list)):
                data = data.append(question_list[i], ignore_index = True)
            
    #remove duplicate questions                   
    data_copy = data[:495]
    data_copy = data_copy.drop_duplicates(subset=['Question'], ignore_index = True, keep = False)
    data = data[495:]
    data = data_copy.append(data)  
    data.drop_duplicates(subset = ['Question'], inplace = True, ignore_index = True)  
    
    data['Final Label']=data['Final Label'].astype('category').cat.codes
    
    
    data.to_excel('output2.xlsx')
    return data

def preprocess(data):
    #lowercasing
    data['Question'] = data["Question"].str.lower()
    
    #stop word
    stop = stopwords.words('english')
    data['Question'] = data['Question'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
    #stemming
    stemmer = nltk.stem.PorterStemmer()
    data['Question'] = data['Question'].apply(lambda x: ' '.join([stemmer.stem(y) for y in x.split()])) # Stem every word.
    
    return data

def convert_to_BOW(data):
    document = []
    data['Question'].apply(lambda x: [document.append(x)])
    
    vectorizer = CountVectorizer()
    data_BOW = vectorizer.fit_transform(document)
    df = pd.DataFrame(data_BOW.toarray())
    data = pd.concat([data, df], axis=1)
    return data

def use_RandomForestClassifier(data, model):
    data = data.drop(columns = ['Question'])
    # using 10 trees
    
    column_names = list(data.columns)
    column_names.remove('Final Label')
    Y = data['Final Label']
    X = data[column_names]
    
    (x_train, x_test, y_train, y_test) = train_test_split(X, Y, train_size=0.7, random_state=1)
    modelRandomForest = RandomForestClassifier(random_state =1, n_estimators = 100)
    modelRandomForest.fit(x_train, y_train)
        
    yPredRandomForest = modelRandomForest.predict(x_test)
    print(f'the accuracy of the random forest classifier {model} is {accuracy_score(y_test, yPredRandomForest)}')
    print(f'the precision of the random forest classifier {model} is {precision_score(y_test, yPredRandomForest, average="micro")}')
    print(f'the accuracy of the random forest classifier {model} is {recall_score(y_test, yPredRandomForest, average="micro")}')

def generate_new_feature(data):
    '''
        The new feature added was the length of the sentence before it was preprocessed
    '''
    length = []
    for index, row in data.iterrows():
        length.append( len(data['Question'][index]))
    data['Length'] = length
    data.to_excel('output2.xlsx')
    return data

    
def classifier_with_new_feature_and_preprocessing(data):
    data = clean_data(data)
    data = generate_new_feature(data)
    data = preprocess(data)
    data = convert_to_BOW(data)
    model = 'with new features and with preprocessing'
    use_RandomForestClassifier(data, model)
    
def classifier_with_new_feature_without_preprocessing(data):
    data = clean_data(data)
    data = generate_new_feature(data)
    data = convert_to_BOW(data)
    model = 'with new features and without preprocessing'
    use_RandomForestClassifier(data, model)
    
def classifier_with_preprocessing(data):
    data = clean_data(data)
    data = preprocess(data)
    data = convert_to_BOW(data)
    model = 'without new features and with preprocessing'
    use_RandomForestClassifier(data, model)
    
def main():
    path = "lab2.xlsx"
    #read data
    data = pd.read_excel(path)
    data = data.drop(columns = ['# Comment', 'inline-comment-id'])
    data = data[data['Final Label'] != 'discarded']
    
    classifier_with_preprocessing(data)
    
    classifier_with_new_feature_without_preprocessing(data)
    
    classifier_with_new_feature_and_preprocessing(data)
    
    
    
    

if __name__ == "__main__":
    main()