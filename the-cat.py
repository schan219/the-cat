"""
chance of student finishing the MOOC
Training data: Climate
Testing data: China
SKLearn Preprocessing Normalize: Avg_Dt
"""

import pandas as pd
import os

PERSON_COURSE_CLEANED = "../UBCx__Climate101x__3T2015_cleaned/person_course_cleaned.tsv"
PERSON_COURSE_DAY_CLEANED = "../UBCx__Climate101x__3T2015_cleaned/person_course_day_cleaned.tsv"

topdir = os.getcwd()
user_path = os.path.join(topdir, PERSON_COURSE_CLEANED)
person_course_day_cleaned_path = os.path.join(topdir, PERSON_COURSE_DAY_CLEANED)

user_df = pd.read_table(user_path)
person_course_day_cleaned_df = pd.read_table(person_course_day_cleaned_path)

result = user_df.join(person_course_day_cleaned_df, lsuffix='user_id', rsuffix='user_id')

def get_grade_label(grade):
    '''
    Return 1 if course was completed (non-NA value), 0 otherwise.
    '''
    if (pd.notnull(grade)):
        return 1
    else:
        return 0
    
result['grade'] = result['grade'].map(get_grade_label)

'''
Dropping all the non-numerical values (for now)
'''
result = result.apply(pd.to_numeric, errors='coerce')
result = result.dropna(axis=1, how='all')

'''
For the numerical values, replace all the NA's with 0's
(because if we only keep the rows that don't have NAs we only get 5 datapoints :'()
'''
result = result.fillna(0)

result.to_csv('left-join.tsv', sep="\t")

'''
Normalize the remaining numerical data: set it to have mean of 0 and standard deviation of 1.
'''
result_norm = result.apply(lambda x: (x - np.mean(x)) / np.std(x))
result_norm.to_csv('result-norm.tsv', sep="\t")

#print(result[1,1])

