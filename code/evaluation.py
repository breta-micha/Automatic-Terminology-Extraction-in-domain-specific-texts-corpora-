import sys
import pandas as pd
import csv

def convert_csv_to_df(inputfile):
    """
    convert csv file to pandas dataframe
    """
    df = pd.read_csv(inputfile, sep= '\t', header=None, encoding = 'utf-8')
    return df

def extract_all_terms(inputfile):
    '''
    extract all annotated terms from csv file
    '''
    terms = []
    dataframe = convert_csv_to_df(inputfile)
    for term in dataframe[0]:
        terms.append(term)
    
    return terms
	
def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))
	
def calculate_precision_recall_fscore(gold_file, machine_file):
    '''
    Calculate precision recall and fscore for each class and return them in a dictionary
    '''
    gold_terms = extract_all_terms(gold_file)
    machine_terms = extract_all_terms(machine_file)
    true_positives = len(intersection(gold_terms, machine_terms))
    
    Recall = true_positives / len(gold_terms)
    Precision = true_positives / len(machine_terms)
    F1score = (2 * Precision * Recall) / (Precision + Recall)
    
    return f"Recall: {Recall} , Precision: {Precision} , F1score: {F1score}."
	
#Annotations
gold_corp_file = '../data/ACTER-master/en/corp/annotations/corp_en_terms.ann'
gold_equi_file = '../data/ACTER-master/en/equi/annotations/equi_en_terms.ann'
gold_wind_file = '../data/ACTER-master/en/wind/annotations/wind_en_terms.ann'
gold_htfl_file = '../data/ACTER-master/en/htfl/annotations/htfl_en_terms.ann'

#Rule-based System -- Output
machine_corp_file = '../data/corp_log_rf.csv'
machine_equi_file = '../data/equi_log_rf.csv'
machine_wind_file = '../data/wind_log_rf.csv'
machine_htfl_file = '../data/htfl_log_rf.csv'

#BERT-based System -- Output
bert_htfl_file = '../data/htfl_bert.csv'

#Evaluations
corp_evaluation = calculate_precision_recall_fscore(gold_corp_file, machine_corp_file)   #Rule-based system output
equi_evaluation = calculate_precision_recall_fscore(gold_equi_file, machine_equi_file)   #Rule-based system output
wind_evaluation = calculate_precision_recall_fscore(gold_wind_file, machine_wind_file)   #Rule-based system output
htfl_evaluation = calculate_precision_recall_fscore(gold_htfl_file, machine_htfl_file)   #Rule-based system output
htfl_bert_evaluation = calculate_precision_recall_fscore(gold_htfl_file, bert_htfl_file) #BERT-based system output

print("------------------- 1. RULE BASED SYSTEM -----------------------")
print()
print("------------ Evaluation on Corp Corpus (dev set) ---------------")
print(corp_evaluation)
print()
print("------------ Evaluation on Equi Corpus (dev set) ---------------")
print(equi_evaluation)
print()
print("------------ Evaluation on Wind Corpus (dev set) ---------------")
print(wind_evaluation)
print()
print("------------ Evaluation on Htfl Corpus (test set) --------------")
print(htfl_evaluation)
print()
print("-------------------  2. BERT BASED SYSTEM ----------------------")
print()
print("------------ Evaluation on Htfl Corpus (test set) --------------")
print(htfl_bert_evaluation)
