"""
chance of student finishing the MOOC
Training data: Climate
Testing data: China
SKLearn Preprocessing Normalize: Avg_Dt
"""

import numpy as np
import pandas as pd
import os
import pydotplus 
from sklearn import tree

PERSON_COURSE_CLEANED = "../UBCx__Climate101x__3T2015_cleaned/person_course_cleaned.tsv"
PERSON_COURSE_DAY_CLEANED = "../UBCx__Climate101x__3T2015_cleaned/person_course_day_cleaned.tsv"
PERSON_COURSE_CLEANED_CHINA = "../UBCx__China300_1x__3T2015_cleaned/person_course_cleaned.tsv"
PERSON_COURSE_DAY_CLEANED_CHINA = "../UBCx__China300_1x__3T2015_cleaned/person_course_day_cleaned.tsv"


def get_data(data_path1, data_path2, filename):
    topdir = os.getcwd()
    person_course_cleaned_path = os.path.join(topdir, data_path1)
    person_course_day_cleaned_path = os.path.join(topdir, data_path2)

    person_course_cleaned_df = pd.read_table(person_course_cleaned_path)
    person_course_day_cleaned_df = pd.read_table(person_course_day_cleaned_path)

    #result = person_course_cleaned_df.join(person_course_day_cleaned_df, lsuffix='user_id', rsuffix='user_id')
    result_1 = pd.merge(person_course_cleaned_df, person_course_day_cleaned_df, on='user_id', how = 'inner')
    def get_grade_label(grade):
        '''
        Return 1 if course was completed (non-NA value), 0 otherwise.
        '''
        if (pd.notnull(grade)):
            return 1
        else:
            return 0

    result_1['grade'] = result_1['grade'].map(get_grade_label)

    '''
    Dropping all the non-numerical values (for now)
    '''
    result = result_1.apply(pd.to_numeric, errors='coerce')
    
    '''
    Save all the categorical columns
    '''
    categorical_columns = result.columns[pd.isnull(result).all()].tolist()
    result = result.dropna(axis=1, how='all')

    '''
    For the numerical values, replace all the NA's with 0's
    (because if we only keep the rows that don't have NAs we only get 5 datapoints :'()
    '''
    result = result.fillna(0)
    # result.to_csv('left-join.tsv', sep="\t")

    '''
    Since we don't want to normalize the grades, and want to use it as the labels instead, we store them separately
    '''
    grades = result['grade']
    result = result.drop(['grade', 'user_id'], axis=1)

    '''
    Normalize the remaining numerical data: set it to have mean of 0 and standard deviation of 1.
    '''
    result_norm = result.apply(lambda x: (x - np.mean(x)) / np.std(x))
    
    '''
    Now convert the categorical features into binary features
    '''
    cat_data = result_1[categorical_columns]
    cat_data.applymap(str)
    cat_data.fillna("")


    cat_data = pd.get_dummies(cat_data.drop(['continent'], axis=1))
    final_frame = cat_data.join(result_norm) 

    
    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf = clf.fit(final_frame, grades)
    dot_data = tree.export_graphviz(clf, out_file=None,feature_names=list(final_frame),  
                             class_names=['Pass','Fail'],  
                             filled=True, rounded=True) 
    graph = pydotplus.graph_from_dot_data(dot_data) 
    graph.write_pdf("gradetree.pdf")
    
    frames = [result_norm, grades]
    final_result = pd.concat(frames, axis=1)
    final_result.to_csv(filename, sep="\t")

get_data(PERSON_COURSE_CLEANED, PERSON_COURSE_DAY_CLEANED, 'final-result.tsv')
get_data(PERSON_COURSE_CLEANED_CHINA, PERSON_COURSE_DAY_CLEANED_CHINA, 'final-result-china.tsv')
