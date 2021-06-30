# Unithood Statistics
import nltk
from nltk import ngrams
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from collections import defaultdict
import nltk.collocations
import pandas as pd
import numpy as np
from utils import load_corpus
from utils import co_occurrence
from utils import frequency_counts
from utils import lemmatize_terms
from linguistics import get_chunks

def log_likelihood_collocations(text):
    """
    extract collocations based on log likelihood ratio 
    default frequency threshold = 3
    default log likelihood score > 10
    """
    ngrams = []
    tokens = nltk.word_tokenize(text)
    lemma_tokens = lemmatize_terms(tokens)
    chunks = get_chunks(text)
    lemma_chunks = lemmatize_terms(chunks)
    
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    
    bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(lemma_tokens)
    trigram_finder = nltk.collocations.TrigramCollocationFinder.from_words(lemma_tokens)
    bigram_finder.apply_ngram_filter(lambda w1, w2: " ".join([w1,w2]) not in lemma_chunks)
    trigram_finder.apply_ngram_filter(lambda w1, w2, w3: " ".join([w1,w2,w3]) not in lemma_chunks)
    bigram_finder.apply_freq_filter(3)
    trigram_finder.apply_freq_filter(3)
    
    bigram_log = bigram_measures.likelihood_ratio
    trigram_log = trigram_measures.likelihood_ratio
    scored_bigrams = bigram_finder.above_score(bigram_log, 10)
    scored_trigrams = trigram_finder.above_score(trigram_log, 10)
    
    for bigram in scored_bigrams:
        ngram = bigram[0] + " " + bigram[-1]
        ngrams.append(ngram) 
    for trigram in scored_trigrams:
        ngram = trigram[0] + " " + trigram[1] + " " + trigram[-1]
        ngrams.append(ngram)
        
    collocations = []
    for chunk in chunks:
        chunk_list = chunk.split(" ")
        if len(chunk_list) == 2:
            a = lemmatizer.lemmatize(chunk_list[0])
            b = lemmatizer.lemmatize(chunk_list[-1])
            chunk_lemma = a + " " + b
            if chunk_lemma in ngrams:
                collocations.append(chunk)
        if len(chunk_list) == 3:
            a = lemmatizer.lemmatize(chunk_list[0])
            b = lemmatizer.lemmatize(chunk_list[1])
            c = lemmatizer.lemmatize(chunk_list[-1])
            chunk_lemma = a + " " + b + " " + c
            if chunk_lemma in ngrams:
                collocations.append(chunk)
    
    return collocations
    
def t_score_collocations(text):
    """
    extract collocations based on t-score 
    default frequency threshold = 3
    default t-score score  > 10
    """
    ngrams = []
    tokens = nltk.word_tokenize(text)
    lemma_tokens = lemmatize_terms(tokens)
    chunks = get_chunks(text)
    lemma_chunks = lemmatize_terms(chunks)
    
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    
    bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(lemma_tokens)
    trigram_finder = nltk.collocations.TrigramCollocationFinder.from_words(lemma_tokens)
    bigram_finder.apply_ngram_filter(lambda w1, w2: " ".join([w1,w2]) not in lemma_chunks)
    trigram_finder.apply_ngram_filter(lambda w1, w2, w3: " ".join([w1,w2,w3]) not in lemma_chunks)
    bigram_finder.apply_freq_filter(3)
    trigram_finder.apply_freq_filter(3)
    
    bigram_t = bigram_measures.student_t
    trigram_t = trigram_measures.student_t
    scored_bigrams = bigram_finder.above_score(bigram_t, 10)
    scored_trigrams = trigram_finder.above_score(trigram_t, 10)
    
    for bigram in scored_bigrams:
        ngram = bigram[0] + " " + bigram[-1]
        ngrams.append(ngram) 
    for trigram in scored_trigrams:
        ngram = trigram[0] + " " + trigram[1] + " " + trigram[-1]
        ngrams.append(ngram)
        
    collocations = []
    for chunk in chunks:
        chunk_list = chunk.split(" ")
        if len(chunk_list) == 2:
            a = lemmatizer.lemmatize(chunk_list[0])
            b = lemmatizer.lemmatize(chunk_list[-1])
            chunk_lemma = a + " " + b
            if chunk_lemma in ngrams:
                collocations.append(chunk)
        if len(chunk_list) == 3:
            a = lemmatizer.lemmatize(chunk_list[0])
            b = lemmatizer.lemmatize(chunk_list[1])
            c = lemmatizer.lemmatize(chunk_list[-1])
            chunk_lemma = a + " " + b + " " + c
            if chunk_lemma in ngrams:
                collocations.append(chunk)
    
    return collocations
    
def pmi_collocations(text):
    """
    extract collocations based on pmi 
    default frequency threshold = 3
    default pmi score > 5
    """
    ngrams = []
    tokens = nltk.word_tokenize(text)
    lemma_tokens = lemmatize_terms(tokens)
    chunks = get_chunks(text)
    lemma_chunks = lemmatize_terms(chunks)
    
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    
    bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(lemma_tokens)
    trigram_finder = nltk.collocations.TrigramCollocationFinder.from_words(lemma_tokens)
    bigram_finder.apply_ngram_filter(lambda w1, w2: " ".join([w1,w2]) not in lemma_chunks)
    trigram_finder.apply_ngram_filter(lambda w1, w2, w3: " ".join([w1,w2,w3]) not in lemma_chunks)
    bigram_finder.apply_freq_filter(3)
    trigram_finder.apply_freq_filter(3)
    
    bigram_pmi = bigram_measures.pmi
    trigram_pmi = trigram_measures.pmi
    scored_bigrams = bigram_finder.above_score(bigram_pmi, 5)
    scored_trigrams = trigram_finder.above_score(trigram_pmi, 5)
    
    for bigram in scored_bigrams:
        ngram = bigram[0] + " " + bigram[-1]
        ngrams.append(ngram) 
    for trigram in scored_trigrams:
        ngram = trigram[0] + " " + trigram[1] + " " + trigram[-1]
        ngrams.append(ngram)
        
    collocations = []
    for chunk in chunks:
        chunk_list = chunk.split(" ")
        if len(chunk_list) == 2:
            a = lemmatizer.lemmatize(chunk_list[0])
            b = lemmatizer.lemmatize(chunk_list[-1])
            chunk_lemma = a + " " + b
            if chunk_lemma in ngrams:
                collocations.append(chunk)
        if len(chunk_list) == 3:
            a = lemmatizer.lemmatize(chunk_list[0])
            b = lemmatizer.lemmatize(chunk_list[1])
            c = lemmatizer.lemmatize(chunk_list[-1])
            chunk_lemma = a + " " + b + " " + c
            if chunk_lemma in ngrams:
                collocations.append(chunk)
    
    return collocations

def jaccard_collocations(text):
    """
    extract collocations based on jaccard coefficient 
	default frequency threshold = 3
    default jaccard score > 0.1
    """
    words = []
    sents = sent_tokenize(text)
    for sent in sents:
        tokens = nltk.word_tokenize(sent)
        for token in tokens:
            word_lemma = lemmatizer.lemmatize(token.lower())
            words.append(word_lemma)
    freq_word = frequency_counts(words)
    
    multi_terms = get_chunks(text)
    freq_multi = frequency_counts(multi_terms)
    candi_matrix = co_occurrence(multi_terms)
            
    jaccard_ratio = dict()
    for candi in multi_terms:
        candi_list = candi.split(" ")
        if len(candi_list) == 2:
            a = lemmatizer.lemmatize(candi_list[0])
            b = lemmatizer.lemmatize(candi_list[-1])
            if (freq_word[a] > 2) and (freq_word[b] > 2):
                jaccard_ratio[candi] = int(candi_matrix.at[a,b]) / (freq_word[a] + freq_word[b])
        if len(candi_list) == 3:
            a = lemmatizer.lemmatize(candi_list[0])
            b = lemmatizer.lemmatize(candi_list[1])
            c = lemmatizer.lemmatize(candi_list[-1])
            if (freq_word[a] > 2) and (freq_word[b] > 2) and (freq_word[c] > 2):
                jaccard_ratio[candi] = freq_multi[candi] / (freq_word[a] + freq_word[b] + freq_word[c])
        
    collocations = []
    for t,v in jaccard_ratio.items():
        if v > 0.1:
            collocations.append(t)
        
    return collocations