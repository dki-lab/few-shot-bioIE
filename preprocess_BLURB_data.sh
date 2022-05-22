#!/bin/sh

# For Chemprot corpus, please register and download from
# https://biocreative.bioinformatics.udel.edu/resources/corpora/chemprot-corpus-biocreative-vi/
if [ -s raw_data/ChemProt_Corpus.zip ]
then 
    unzip raw_data/ChemProt_Corpus.zip -d raw_data/
    unzip raw_data/ChemProt_Corpus/chemprot_training.zip -d raw_data/ChemProt_Corpus/chemprot_training
    unzip raw_data/ChemProt_Corpus/chemprot_development.zip -d raw_data/ChemProt_Corpus/chemprot_development
    unzip raw_data/ChemProt_Corpus/chemprot_test_gs.zip -d raw_data/ChemProt_Corpus/chemprot_test_gs
fi

python preprocessor.py 
