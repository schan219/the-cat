"""
chance of student finishing the MOOC
Training data: Climate
Testing data: China
"""

import pandas as pd
import os
import csv

PERSON_COURSE_CLEANED = "../UBCx__Climate101x__3T2015_cleaned/person_course_cleaned.tsv"
PERSON_COURSE_DAY_CLEANED = "../UBCx__Climate101x__3T2015_cleaned/person_course_day_cleaned.tsv"

topdir = os.getcwd()
user_path = os.path.join(topdir, PERSON_COURSE_CLEANED)
person_course_day_cleaned_path = os.path.join(topdir, PERSON_COURSE_DAY_CLEANED)

user_df = pd.read_table(user_path)
person_course_day_cleaned_df = pd.read_table(person_course_day_cleaned_path)

result = user_df.join(person_course_day_cleaned_df, lsuffix='user_id', rsuffix='user_id')

with open('left-join.csv', 'w') as csvfile:
    a = csv.writer(csvfile)
    a.writerows(result)
    csvfile.close()
