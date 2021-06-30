#Import
from utils import load_corpus
from linguistics import get_chunks
from linguistics import get_words
import csv
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
model_path = 'distilbert-base-uncased-2021-06-21_01-30-13'
model = SentenceTransformer(model_path)
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

def get_candidates(text):
    """
    extract candidate terms from any text based on linguitistic filtering
	text: a text as a string
    """
    words = get_words(text)
    chunks = get_chunks(text)
    
    candidates = []
    for w in words:
        candidates.append(w)
    for c in chunks:
        candidates.append(c)
        
    return candidates
	
def get_cosine_similarity_of_candidates(text):
    """
    extract terms by computing the cosine similarity score of document embeddings and candidate embeddings
	cut off scores (by default):
		- unigrams: 0.009
		- bigrams:  0.3
		- trigrams: 0.45	
	text: a text as a string
    """
    terms = []
    candidates = get_candidates(text)
    
    if len(candidates) > 0:
        document_embeddings = model.encode([text])
        candidate_embeddings = model.encode(candidate)
        top_n = 500
        distances = cosine_similarity(document_embeddings, candidate_embeddings)
        keywords = [(candidates[index], round(float(distances[0][index]), 4)) for index in distances.argsort()[0][-top_n:]][::-1]
        for key in keywords:
            term = key[0]
            score = key[-1]
            term_list = term.split(" ")
            if len(term_list) == 1 and score > 0.09:
                terms.append(term)
            if len(term_list) == 2 and score > 0.3:
                terms.append(term)
            if len(term_list) == 3 and score > 0.45:
                terms.append(term)
        else:
            terms = []
    
    return terms
	
def chunkify(lst,n):
    return [lst[i::n] for i in range(n)]
	
def extract_terms(input_folder, outfile):
    """
	extract terms from corpus by using a BERT-based Sentence Transformer model
	input_folder: the path to the domain corpus folder
	outfile: the path to the csv file for the output
    """
    output = open(outfile, 'w', newline='', encoding = 'utf-8', errors = 'ignore')
    writer = csv.writer(output)
    
    terms = []
    corpus = load_corpus(input_folder)
    sents = sent_tokenize(corpus)
    length = len(sents)
    chunks = round(length / 4)
    pieces_of_text = chunkify(sents, chunks)
    
    for piece in pieces_of_text:
        text = ' '.join(p for p in piece)
        local_terms = get_cosine_similarity_of_candidates(text)
        if len(local_terms) > 0:
            for t in local_terms:
                terms.append(t)
    
    sorted_terms = sorted(list(set(terms)))
    
    for term in sorted_terms:
        writer.writerow([term])
		
#Input folder
htfl_folder = '../data/ACTER-master/en/htfl/texts/annotated'
#Outfile
htfl_terms = '../data/htfl_bert.csv'
#Create Output
extract_terms(htfl_folder, htfl_terms)
