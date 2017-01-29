
# coding: utf-8

# In[43]:

import numpy as np
import pandas as pd
import pylab as plt 
import os
from sklearn import tree


# In[3]:

FORUM_POST_PATH = "forum_posts_anonmyized.tsv"
USER_ID_PATH = "person_course_cleaned.tsv"
EXIT_SURVEY_PATH = "climate_exit_survey.csv"
PRE_POST_PATH = "climate_pre_post_test.csv"


# In[55]:

topdir = os.getcwd()
forum_path = os.path.join(topdir, FORUM_POST_PATH)
user_path = os.path.join(topdir, USER_ID_PATH)
exit_survey_path = os.path.join(topdir, EXIT_SURVEY_PATH)
pre_post_path = os.path.join(topdir, PRE_POST_PATH)
forum_df = pd.read_table(forum_path)
user_df = pd.read_table(user_path)
exit_df = pd.read_csv(exit_survey_path)
pre_df = pd.read_csv(pre_post_path)


# In[31]:

survey_df = pd.merge(exit_df, pre_df, how='inner', on='user_id')
survey_df = survey_df[pd.notnull(survey_df["Were your goals for taking the course met?"])]
survey_df = survey_df[pd.notnull(survey_df["learning_gain"])]
answers = [x for x in pd.unique(survey_df["Were your goals for taking the course met?"])]
answers_key = {}
for i in range(len(answers)):
    answers_key[str(answers[i])] = i
for idx, data in survey_df.iterrows():
    survey_df.loc[idx, "Satisfaction"] = answers_key[str(data["Were your goals for taking the course met?"])]
#print(exit_df["user_id"].values)
#print(pre_df["user_id"].values)
survey_df = survey_df[pd.notnull(survey_df["learning_gain"])]
get_ipython().magic('matplotlib inline')
#survey_df.head()
#print(survey_df["Were your goals for taking the course met?"].values)
plt.scatter(survey_df["Satisfaction"].values, survey_df["learning_gain"].values)
plt.text(0.5, -1.3, "Satisfaction")
plt.text(-0.8, 0.6, "gain")


# In[40]:

flearn_df = pd.merge(pre_df, forum_df, on = 'user_id', how = 'inner')
flearn_df.head()
num_posts = []
l_gain = []
for name, group in flearn_df.groupby('user_id'):
    num_posts.append(group.size)
    l_gain.append(group["learning_gain"].values[0])
plt.scatter(num_posts, l_gain)


# In[42]:

ulearn_df = pd.merge(user_df, forum_df, on = 'user_id', how = 'inner')
ulearn_df.head()
num_posts = []
l_gain = []
for name, group in ulearn_df.groupby('user_id'):
    if (group.size <= 500):
        num_posts.append(group.size)
        l_gain.append(group["grade"].values[0])
plt.scatter(num_posts, l_gain)


# In[54]:

clf = tree.DecisionTreeClassifier()
grades = []
user_df = user_df[pd.notnull(user_df['grade'])]
#user_df = user_df.drop(['start_time', 'last_event'], 1)
def get_grade(num):
    if (num <= 50):
        return 'F'
    elif (num <=68):
        return 'C'
    elif (num <=80):
        return 'B'
    else:
        return 'A'     
    
for idx, row in user_df.iterrows():
    grades.append(get_grade(float(row['grade'])))
    gender = user_df.loc[idx, 'gender']
    if (gender == 'm'):
        user_df.loc[idx, 'gender'] = 1
    elif (gender == 'f'):
        user_df.loc[idx, 'gender'] = 0
    else:
        user_df.loc[idx, 'gender'] = 2
    
    
clf.fit(user_df, grades)

