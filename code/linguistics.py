#Linguistic Filtering
import string
import nltk
from nltk import ngrams
from nltk.tokenize import sent_tokenize

def ngramise(sequence):
    '''
    Iterate over ngrams
    '''
    for bigram in nltk.ngrams(sequence, 2):
        yield bigram
    for trigram in nltk.ngrams(sequence, 3):
        yield trigram

def get_chunks(text):
    '''
    extract canidate chunks
    '''
    list_of_ngrams = []
    candidate_ngrams = []
    filtered_ngrams = []
    
    sents = sent_tokenize(text)
    for sent in sents:
        tokens = nltk.word_tokenize(sent)
        pos_tags = nltk.pos_tag(tokens)
        for ngram in ngramise(pos_tags):
            words, tags = zip(*ngram)
            if len(tags) == 2 and \
            (tags[0] in ['NN', 'NNS', 'JJ', 'VB', 'VBG', 'VBN']) and \
            (tags[1] in ['NN', 'NNS']):
                    candidate_ngrams.append(words)
            if len(tags) == 3 and \
            (tags[0] in ['NN', 'NNS', 'JJ', 'VB', 'VBG', 'VBN']) and \
            (tags[1] in ['NN', 'NNS', 'JJ', 'VB', 'VBG', 'VBN', 'IN']) and \
            (tags[2] in ['NN', 'NNS']):
                    candidate_ngrams.append(words)
            
    for ngram in candidate_ngrams:
        if len(ngram) == 2:
            a = ngram[0].lower()
            b = ngram[-1].lower()
            term = a + " " + b
            list_of_ngrams.append(term)
        if len(ngram) == 3:
            a = ngram[0].lower()
            b = ngram[1].lower()
            c = ngram[-1].lower()
            term = a + " " + b + " " + c
            list_of_ngrams.append(term)
        
    for item in list_of_ngrams:
        if any(c.isalpha() for c in item) and \
        (item[0] not in string.punctuation) and \
        (item[-1] not in string.punctuation):
            filtered_ngrams.append(item)
                             
    return filtered_ngrams
    
def get_words(text):
    """
    extract candidate unigrams
    """
    list_of_words = []
    filtered_words = []
    
    sents = sent_tokenize(text)
    for sent in sents:
        tokens = nltk.word_tokenize(sent)
        pos_tags = nltk.pos_tag(tokens)
        for word, tag in pos_tags:
            if tag in ['NN', 'NNS', 'JJ', 'VB', 'VBG', 'VBN']:
                list_of_words.append(word.lower())
                
    for item in list_of_words:
        if any(c.isalpha() for c in item) and \
        (item.startswith('www.') == False) and \
        (item[0] not in string.punctuation) and \
        (item[-1] not in string.punctuation):
            filtered_words.append(item)
    
    return filtered_words