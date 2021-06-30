# Relative Frequency Ratio
import nltk
from nltk import ngrams
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from utils import load_corpus
from utils import frequency_counts
from utils import lemmatize_terms
from linguistics import get_words
from linguistics import ngramise
from unithood_statistics import log_likelihood_collocations
from unithood_statistics import t_score_collocations
from unithood_statistics import pmi_collocations
from unithood_statistics import jaccard_collocations

def get_terms_from_corpus(text, measure = 'log'):
    """
    extract candidate terms from corpus and put them in a single list
    
    text: the domain specific corpus as a string
    measure: the statistical measure for the extraction of collocations (multi word candidate terms)
        - measure = jaccard (jaccard coefficient)
        - measure = pmi     (pointwise mutual information)
        - measure = log     (log likelihood ratio)
        - measure = tscore  (t-test)
    """
    terms = []
    single_word_candidates = get_words(text)
    for s in single_word_candidates:
        terms.append(s)
    
    if measure == 'jaccard':
        multi_word_candidates = jaccard_collocations(text)
        for m in multi_word_candidates:
            terms.append(m)
    if measure == 'pmi':
        multi_word_candidates = pmi_collocations(text)
        for m in multi_word_candidates:
            terms.append(m)
    if measure == 'log':
        multi_word_candidates = log_likelihood_collocations(text)
        for m in multi_word_candidates:
            terms.append(m)
    if measure == 'tscore':
        multi_word_candidates = t_score_collocations(text)
        for m in multi_word_candidates:
            terms.append(m)
        
    return terms

def terms_in_general_corpus(terms, text):
    """
    create a list with common candidate terms in general coprus and in domain corpus
    
    terms: the extracted candidate terms from the domain corpus
    text:  the general corpus as a string
    """
    terms_in_domain_corpus = list(set(terms))
    terms_in_general_corpus = []
    list_of_chunks = []
    
    sents = sent_tokenize(text)
    for sent in sents:
        tokens = nltk.word_tokenize(sent)
        for token in tokens:
            if token in terms_in_domain_corpus:
                terms_in_general_corpus.append(token)
        for ngram in ngramise(tokens):
            chunk = ' '.join([i.lower() for i in ngram])
            list_of_chunks.append(chunk)
    
    for chunk in list_of_chunks:
        if chunk in terms_in_domain_corpus:
            terms_in_general_corpus.append(chunk)
    
    return terms_in_general_corpus
	
def relative_frequency_ratio(domain_folder, general_folder, measure = 'log'):
    """
    selects terms based on relative frequency ratio
    default relative frequency ratio = 0.1
    
    domain_folder: the folder with the domain specific corpus
    general_folder: the folder with the general corpus (brown corpus)
    measure: the statistical measure for the extraction of collocations (multi word candidate terms)
        - measure = jaccard (jaccard coefficient)
        - measure = pmi     (pointwise mutual information)
        - measure = log     (log likelihood ratio)
        - measure = tscore  (t-test)
    """
    domain_corpus = load_corpus(domain_folder)                                      # specific domain corpus
    general_corpus = load_corpus(general_folder)                                    # general corpus
    
    terms_in_domain = get_terms_from_corpus(domain_corpus, measure)                 # candidate terms in specific domain corpus
    terms_in_general = terms_in_general_corpus(terms_in_domain, general_corpus)     # candidate terms in general corpus
    
    lemmas_in_domain = lemmatize_terms(terms_in_domain)                             # lemmatized candidate terms in specific domain corpus
    lemmas_in_general = lemmatize_terms(terms_in_general)                           # lemmatized candidate terms in general corpus
    
    freq_domain = frequency_counts(lemmas_in_domain)                                # Term:Freq in specific domain corpus
    freq_general = frequency_counts(lemmas_in_general)                              # Term:Freq in general corpus
    
    num_token_domain = len(nltk.word_tokenize(domain_corpus))                       # Num of tokens in specific domain corpus
    num_token_general = len(nltk.word_tokenize(general_corpus))                     # Num of tokens in general corpus
    
    freq_dict = dict()
    
    for t,v in freq_domain.items():
        if t in freq_general.keys():
            freq_dict[t] = (freq_general[t] * num_token_domain) / (v * num_token_general)
        else:
            freq_dict[t] = (0 * num_token_domain) / (v * num_token_general)
            
    terms = []
    for term in terms_in_domain:
        term_list = term.split(" ")
        if len(term_list) == 1:
            term_lemma = lemmatizer.lemmatize(term_list[0])
            for t,v in freq_dict.items():
                if (t == term_lemma) and (v < 0.1):
                    terms.append(term)
        if len(term_list) == 2:
            a = lemmatizer.lemmatize(term_list[0])
            b = lemmatizer.lemmatize(term_list[-1])
            term_lemma = a + " " + b
            for t,v in freq_dict.items():
                if (t == term_lemma) and (v < 0.1):
                    terms.append(term)
        if len(term_list) == 3:
            a = lemmatizer.lemmatize(term_list[0])
            b = lemmatizer.lemmatize(term_list[1])
            c = lemmatizer.lemmatize(term_list[-1])
            term_lemma = a + " " + b + " " + c
            for t,v in freq_dict.items():
                if (t == term_lemma) and (v < 0.1):
                    terms.append(term)
        
    return terms