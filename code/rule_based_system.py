# Rule Based System
import csv
from relative_ratio import relative_frequency_ratio

def find_terms(domain_folder, general_folder, outfile, measure = 'log'):
    """
    extract candidate terms from corpus and put them in a csv file
    default relative ratio < 0.1 (term extracted is at least 10 times more frequent in domain corpus)
	
	domain_folder: the folder with the domain specific corpus
    general_folder: the folder with the general corpus (brown corpus)
    measure: the statistical measure for the extraction of collocations (multi word candidate terms)
        - measure = jaccard (jaccard coefficient: default score > 0.1, frequency threshold = 3)
        - measure = pmi     (pointwise mutual information: default score > 5, frequency threshold = 3)
        - measure = log     (log likelihood ratio: default score > 10, frequency threshold = 3)
        - measure = tscore  (t-score: default score > 10, frequency threshold = 3)
	outfile: output in a csv file
    """
    output = open(outfile, 'w', newline='', encoding = 'utf-8', errors = 'ignore')
    writer = csv.writer(output)
    terms = relative_frequency_ratio(domain_folder, general_folder, measure)
    sorted_terms = sorted(list(set(terms)))
    
    for term in sorted_terms:
        writer.writerow([term])
        
#General Folder: placed on data folder
general_folder = '../data/brown_corpus'
#Domain Folder: placed on data folder
corp_folder = '../data/ACTER-master/en/corp/texts/annotated'
equi_folder = '../data/ACTER-master/en/equi/texts/annotated'
wind_folder = '../data/ACTER-master/en/wind/texts/annotated'
htfl_folder = '../data/ACTER-master/en/htfl/texts/annotated'

#Create Output (Log Likelihood + Relative Freq): placed on data folder
corp_terms = '../data/corp_log_rf.csv'
equi_terms = '../data/equi_log_rf.csv'
wind_terms = '../data/wind_log_rf.csv'
htfl_terms = '../data/htfl_log_rf.csv'
find_terms(corp_folder, general_folder, corp_terms, measure = 'log')
find_terms(equi_folder, general_folder, equi_terms, measure = 'log')
find_terms(wind_folder, general_folder, wind_terms, measure = 'log')       
find_terms(htfl_folder, general_folder, htfl_terms, measure = 'log')
