"""
chance of student finishing the MOOC
Training data: Climate
Testing data: China
SKLearn Preprocessing Normalize: Avg_Dt
"""

import numpy as np
import pandas as pd
import os

PERSON_COURSE_CLEANED = "../UBCx__Climate101x__3T2015_cleaned/person_course_cleaned.tsv"
PERSON_COURSE_DAY_CLEANED = "../UBCx__Climate101x__3T2015_cleaned/person_course_day_cleaned.tsv"
PERSON_COURSE_CLEANED_CHINA = "../UBCx__China300_1x__3T2015_cleaned/person_course_cleaned.tsv"
PERSON_COURSE_DAY_CLEANED_CHINA = "../UBCx__China300_1x__3T2015_cleaned/person_course_day_cleaned.tsv"


def get_data(data_path1, data_path2, filename):
    topdir = os.getcwd()
    data_path1 = os.path.join(topdir, data_path1)
    data_path2 = os.path.join(topdir, data_path2)

    data1_df = pd.read_table(data_path1)
    data2_df = pd.read_table(data_path2)

    result = pd.merge(data1_df, data2_df, on='user_id', how='inner')

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

    '''
    Since we don't want to normalize the grades, and want to use it as the labels instead, we store them separately
    '''
    grades = result['grade']
    result = result.drop(['grade', 'user_id'], axis=1)
    if 'nproblems_answered' in result:
        result = result.drop(['nproblems_answered'], axis=1)

    '''
    Normalize the remaining numerical data: set it to have mean of 0 and standard deviation of 1.
    '''
    result_norm = result.apply(lambda x: (x - np.mean(x)) / np.std(x))

    frames = [result_norm, grades]
    final_result = pd.concat(frames, axis=1)
    final_result.to_csv(filename, sep="\t")

get_data(PERSON_COURSE_CLEANED, PERSON_COURSE_DAY_CLEANED, 'final-result.tsv')
get_data(PERSON_COURSE_CLEANED_CHINA, PERSON_COURSE_DAY_CLEANED_CHINA, 'final-result-china.tsv')
