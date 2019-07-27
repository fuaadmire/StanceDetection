#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 19:31:48 2019

@author: hardiksahi
"""
import pandas as pd
from utils.score import report_score
#import csv

# Read original stance
original_stance_path = '/Users/hardiksahi/LastTerm/TextAnalytics/Project/Stance/data/competition_test_stances.csv'
predicted_stance_path = '/Users/hardiksahi/LastTerm/TextAnalytics/Project/Stance/output/stance.csv'

original_stance_list = pd.read_csv(original_stance_path)['Stance'].tolist()
predicted_stance_list = pd.read_csv(predicted_stance_path)['Stance'].tolist()

# =============================================================================
# predicted_stance_list = []
# with open(predicted_stance_path,"r") as f:
#     reader = csv.reader(f,delimiter="\n")
#     for line in reader:
#         predicted_stance_list.append(line[0])
# =============================================================================


print("Len ooriginal_stance_list: ", len(original_stance_list))
print()

#

score = report_score(original_stance_list, predicted_stance_list)
print("Score on competition dataset::", score)
