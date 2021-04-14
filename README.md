# Reading Task Classification

## Data

From ZuCo 1.0 and 2.0: NR and TSR tasks

On UZH server:  
ZuCo 1.0: methlab/NLP/Ce-ETH/FirstLevel_concat_unfold_correctlyMergedSacc_   
ZuCo 2.0: methlab/NLP/Ce_ETH/2019/FirstLevelV2_concat_unfold_correctlyMergedSacc

On spaceML:  
noraho@spaceml3:/mnt/ds3lab-scratch/noraho/datasets/zuco/zuco1_preprocessed_sep2020  
noraho@spaceml3:/mnt/ds3lab-scratch/noraho/datasets/zuco/zuco2_preprocessed_sep2020


## Classification with sentence-level features

classify_nr_trs.py 

###  Features

#### Eye-Tracking
- Omission rate
- Number of fixations (relative to the number of words in a sentence)
- Reading speed (seconds spent on a sentence relative to the number of words)