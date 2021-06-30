#Utils File
import glob
import os
import csv
import re
import nltk
from nltk import ngrams
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from collections import defaultdict
import pandas as pd
import numpy as np

def get_paths(input_folder):
    '''
    This function provides all the names of the txt files in one folder
    Note that you must import glob and os to use this function
    :param input_folder: a string that specifies the path to the folder
    :returns: a list of all txt file paths
    '''
    list_of_filepaths = []                                          
    for filepath in glob.glob(input_folder + '/*.txt'):
        newpath = filepath.replace(os.sep, '/')           #convert the back slashes (windows) to forward slashes 
        list_of_filepaths.append(newpath)
        
    return list_of_filepaths

def load_text(txt_path):
    '''
    This function loads the content of a file for reading
    :param txt_path: a string that indicates the path to the file
    :returns: the content of the file as a string
    '''
    with open(txt_path, 'r', encoding= "utf8") as infile:           #preventing encoding errors
        content = infile.read()
        
    return content
    
def normalize_text(text):
    """
    This function takes a an input a text and 'cleans up' its content
    :param text: a text as a string
    """
    c1 = re.sub('[\n]', ' ', text)                         #remove newline chars
    c2 = re.sub('[\t]', '', c1)                            #remove title chars
    c3 = re.sub(r"[()\"#/@%<>{}—≤`→+=~$€*•]",'', c2)      #remove special chars
    
    return c3

def load_corpus(input_folder):
    """
    put all texts from corpus into a single string
    """
    list_of_texts = []
    
    filepaths = get_paths(input_folder)
    for filepath in filepaths:
        text = load_text(filepath)
        lower_text = text.lower()
        clean_text = normalize_text(lower_text)
        list_of_texts.append(clean_text)
    
    text = '. '.join(t for t in list_of_texts)
    
    return text
    
def load_stopwords(txt_file):
    """
    load list of keywords from txt file
    """
    content = load_text(txt_file)  #file in data folder
    c = re.sub('[\n]', ' ', content)
    list_of_stopwords = c.split(' ')
    stopwords = [] 
    for word in list_of_stopwords:
        stopwords.append(word)
    
    return stopwords
    
def corpus_for_training(input_folder):
    """
    put all texts from corpus into a single list
    """
    list_of_texts = []
    
    filepaths = get_paths(input_folder)
    for filepath in filepaths:
        text = load_text(filepath)
        clean_text = normalize_text(text)
        list_of_texts.append(clean_text)
        
    return list_of_texts
    
def co_occurrence(terms, window_size = 2):
    """
    co-occurance matrix of lemmatized words in terms
    """
    d = defaultdict(int)
    vocab = set()
    for text in terms:
        text = text.lower().split()
        for i in range(len(text)):
            token = text[i]
            token_lemma = lemmatizer.lemmatize(token)
            vocab.add(token_lemma)  # add to vocab
            next_token = text[i+1 : i+1+window_size]
            for t in next_token:
                key = tuple( sorted([t, token_lemma]))
                d[key] += 1
    
    # formulate the dictionary into dataframe
    vocab = sorted(vocab) # sort vocab
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab)
    for key, value in d.items():
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    
    return df
    
def frequency_counts(terms):
    """
    create a counter for the terms
    """
    freq_counts = dict()
    for t in terms:
        if t in freq_counts:
            freq_counts[t]+=1
        else:
            freq_counts[t] =1
    
    return freq_counts
    
def lemmatize_terms(terms):
    """
    lemmatize candidate terms
    """
    new_terms = []
    for t in terms:
        t_list = t.split(" ")
        if len(t_list) == 1:
            new_terms.append(lemmatizer.lemmatize(t_list[0]))
        if len(t_list) == 2:
            a = lemmatizer.lemmatize(t_list[0])
            b = lemmatizer.lemmatize(t_list[-1])
            chunk = a + " " + b
            new_terms.append(chunk)
        if len(t_list) == 3:
            a = lemmatizer.lemmatize(t_list[0])
            b = lemmatizer.lemmatize(t_list[1])
            c = lemmatizer.lemmatize(t_list[-1])
            chunk = a + " " + b + " " + c
            new_terms.append(chunk)
            
    return new_terms

    
