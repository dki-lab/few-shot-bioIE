#!/bin/sh

# Downloading NER data.
mkdir raw_data
mkdir data
git clone https://github.com/cambridgeltl/MTL-Bioinformatics-2016.git raw_data/MTL-Bioinformatics-2016
mkdir raw_data/JNLPBA
wget -O raw_data/JNLPBA/Genia4ERtraining.tar.gz http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Train/Genia4ERtraining.tar.gz
wget -O raw_data/JNLPBA/Genia4ERtest.tar.gz http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Evaluation/Genia4ERtest.tar.gz
tar zxvf raw_data/JNLPBA/Genia4ERtraining.tar.gz -C raw_data/JNLPBA/
tar zxvf raw_data/JNLPBA/Genia4ERtest.tar.gz -C raw_data/JNLPBA/

# Downloading DDI corpus.
git clone https://github.com/zhangyijia1979/hierarchical-RNNs-model-for-DDI-extraction.git raw_data/DDI
tar -zxvf raw_data/DDI/DDIextraction2013/DDIextraction_2013.tar.gz -C raw_data/DDI/

#Downloading GAD corpus
wget -O raw_data/REdata.zip https://drive.google.com/u/0/uc?id=1-jDKGcXREb2X9xTFnuiJ36PvsqoyHWcw
unzip raw_data/REdata.zip -d raw_data/