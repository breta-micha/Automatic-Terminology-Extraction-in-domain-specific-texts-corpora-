# Automatic Terminology Extraction in domain specific texts in English
A comparison between a rule-based system (statistical approach) and a machine learning system (BERT based approach)

for the excecution of this program the following packages are needed:
   - glob
   - os
   - csv
   - re
   - nltk
   - pandas
   - numpy
   - sklearn
   - Sentence Tranformers (https://www.sbert.net/) 

on data folder place the following folders:
   - ACTER dataset (link https://github.com/AylaRT/ACTER)
   - Brown corpus (link http://korpus.uib.no/icame/manuals/BROWN/INDEX.HTM)

1. from command line run the following command (output of the rule-based system):
   python rule_based_system.py
   
2. from command line run the following command (output of the BERT-based system):
   python BERT_based_system.py
   
3. from command line run the following command (evaluation on the output of the two systems):
   python evaluation.py
