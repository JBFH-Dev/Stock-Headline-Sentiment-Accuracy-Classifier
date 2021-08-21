#               README
#   - This code requires the use of the following Python packages:
#       * Pandas
#       * Numpy
#       * Matplotlib
#       * Sklearn
#       * Stanza
#       * NLTK

#
#   - These packages must be the versions available as of 24/05/2021 or higher
#
#   - To install/update these packages type:
#       -$ pip3 install <PACKAGE NAME> -U
#
#   - CSV files are also required which can be found in the included folder 'NLPData'
#
#   RUNNING THE CODE
#   - Ensure that the file structure is as follows (Other csv files are included for convenience but not necessary):
#      | HeadlinePerfModel.py
#        NLPData/
#          | semantic_headlines.csv
#          | api_response_window_91.csv

#
#   - Navigate to the parent directory of HeadlinePerfModel.py
#
#   - Type the following into the terminal
#       -$ python3 HeadlinePerfModel.py
#
# ------ REFERENCED SOURCES OF CODE ------------
#   MY 2020 MACHINE LEARNING COURSEWORK FOR THE SOFTWARE METHODOLOGIES MODULE
#   https://stackoverflow.com/questions/54730276/how-to-randomly-split-a-dataframe-into-several-smaller-dataframes
#   https://github.com/javedsha/text-classification/blob/master/Text%2BClassification%2Busing%2Bpython%2C%2Bscikit%2Band%2Bnltk.py
#   #https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python

# -------------------------------------------------------------------------------------------

# ----- BELOW CODE USED FOR REQUESTING API DATA, NOT NEEDED TO RUN BUT KEPT TO SHOW MY WORKING

'''


import requests
df_list = []
for stk in tqdm(api_stocks):
  print(stk)
  total_records = 0
  offset = 0
  params = {
    'access_key': '5e51eceeb1873e7f0a5602556af03366',
    'symbols': stk,
    'limit': '1000',
    'offset': offset
  }
  api_result = requests.get('https://api.marketstack.com/v1/eod', params)
  if not api_result.status_code == 200:
    print('stock was invalid')
    print('error code: ', api_result.status_code)
    continue
  try:
    api_response = api_result.json()
  except JSONDecodeError:
    print('Parsing failed, here\'s why:')
    print(api_result.text)
    continue
  total_records = int(api_response['pagination']['total'])
  print(total_records)
  df_list.append(pd.json_normalize(api_response['data']))
  while total_records - (1000*(offset+1)) > 0:
    print(offset)
    offset += 1
    params = {
    'access_key': '5e51eceeb1873e7f0a5602556af03366',
    'symbols': stk,
    'limit': '1000',
    'offset': offset
    }
    api_result = requests.get('https://api.marketstack.com/v1/eod', params)
    try:
      api_response = api_result.json()
    except JSONDecodeError:
      print('Parsing failed, here\'s why:')
      print(api_result.text)
      continue
    df_list.append(pd.json_normalize(api_response['data']))
api_data = pd.concat(df_list)



'''


# FIND DATA PREP CODE AT
#https://colab.research.google.com/drive/135VCyf121kD8-v7YkDpnjHLIVjKAQ9Zo?usp=sharing

# FIND FULL EXPERIMENT OF MODELS AT
# https://colab.research.google.com/drive/14pJo-uB-9uIh7eVfJHZUBYmFtBoHK_Wx?usp=sharing





# IMPORTED MODULES
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import time
import pandas as pd

pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt

# ------ LOAD HEADLINE DATA -------
headlines = pd.read_csv('NLPData/semantic_headlines.csv').drop(columns=['Unnamed: 0']).reset_index()
headlines['date'] = pd.to_datetime(headlines['date'])
headlines.sort_values('stock', inplace=True)

# ------- DROPPING NEUTRAL SENTIMENT HEADLINES -------
headlines.drop(headlines[headlines.sentiment == 1].index, inplace=True)

# ------ STOCK PRICE DATA PREP ----
api_data = pd.read_csv('NLPData/api_response_window_91.csv').drop(columns=['Unnamed: 0'])
api_data['date'] = pd.to_datetime(api_data['date'])
api_data.drop(
    columns=['open', 'high', 'low', 'close', 'adj_high', 'adj_low', 'adj_open', 'adj_volume', 'volume', 'adj_close',
             'split_factor', 'exchange'], inplace=True)
api_data.columns = ['stock', 'date', 'daily_change']
api_data.sort_values('stock', inplace=True)

# ------ MERGING THE TWO TABLES BY DATE AND STOCK ---------
accuracy = pd.merge(headlines, api_data, on=['stock', 'date'], how='inner')
accuracy = accuracy.astype({"daily_change": int, "sentiment": int})
accuracy.drop_duplicates(inplace=True)


# ------ ACCURACY OF HEADLINE SENTIMENT TO STOCK PERFORMANCE ------
def acc(sentiment, daily_change):
    return int(sentiment == daily_change)

# ------ APPLY ACCURACY FUNCTION TO CREATE NEW COLUMN -------
accuracy['accuracy'] = accuracy.apply(lambda x: acc(x['sentiment'], x['daily_change']), axis=1)
accuracy = accuracy.astype({"accuracy": int})
accuracy.dropna(inplace=True)

# ------- SPLITTING INTO TEST AND TRAIN SETS OF EQUAL PERCENTAGE TRUE AND FALSE -------
pos_set = accuracy.loc[accuracy['accuracy'] == 1]
neg_set = accuracy.loc[accuracy['accuracy'] == 0]

# ------- SPLITTING CODE FOUND ONLINE ------------
def split_set(input_set):
    # https://stackoverflow.com/questions/54730276/how-to-randomly-split-a-dataframe-into-several-smaller-dataframes
    shuffled = input_set.sample(frac=1)
    result = np.array_split(shuffled, 4)
    return result

pos_split = split_set(pos_set)
neg_split = split_set(neg_set)
test_set = pd.concat([pos_split[0], neg_split[0]])
train_set = pd.concat([pos_split[1], neg_split[1], pos_split[2], neg_split[2], pos_split[3], neg_split[3]])

train_set_data = train_set.title.tolist()
test_set_data = test_set.title.tolist()
test_set_labels = test_set.accuracy.tolist()
train_set_labels = train_set.accuracy.tolist()

# ------------ TIME TO MAKE AND TEST MODELS!!! ----------------

# https://github.com/javedsha/text-classification/blob/master/Text%2BClassification%2Busing%2Bpython%2C%2Bscikit%2Band%2Bnltk.py


nltk.download('popular', quiet=True)

stemmer = SnowballStemmer("english", ignore_stopwords=True)

# --------- CREATE NEW COUNT VECTORIZER THAT UTILISES THE SNOWBALL STEMMER FROM NLTK
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

# --------- MARK STOP WORDS SO THAT 'IF' 'AND' ETC DO NOT GET USED IN FEATURE CALCULATION -------
stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

# ------- GRID SEARCH (WILL NOT RUN IN CODE BECAUSE IT TAKES A LONG TIME
#                               BUT IS LEFT IN SO YOU CAN SEE MY METHOD) --------

'''
from sklearn.model_selection import GridSearchCV



# ----- CUSTOM PARAMETERS FOR TUNING RANDOM FOREST --------

parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf-svm__n_estimators': [10, 100, 200], 'clf-svm__criterion': ['gini','entropy'],
                  'clf-svm__bootstrap': [True, False]}

gs_clf_svm = GridSearchCV(text_rfc_stemmed, parameters_svm, n_jobs=-1, verbose=1)
gs_clf_svm = gs_clf_svm.fit(train_set_data, train_set_labels)

tuned_model = gs_clf_svm.best_estimator_
print(gs_clf_svm.best_params_) #{'clf-svm__bootstrap': True, 'clf-svm__criterion': 'gini', 'clf-svm__n_estimators': 100, 'tfidf__use_idf': True, 'vect__ngram_range': (1, 2)}
print(gs_clf_svm.best_score_) #0.618206745270085

'''

# ------ BUILD MODEL INTO PIPELINE WITH THE PARAMETERS FOUND TO WORK BEST FROM THE GRID SEARCH -------
text_rfc_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer(use_idf=True)),
                             ('clf-svm', RandomForestClassifier(n_jobs=-1, random_state=117, class_weight='balanced',
                                                                bootstrap=True, criterion='gini', n_estimators=100))])

# ------ GATHER ALL FEATURES AND LABELS FOR USE IN CROSS VALIDATION
all_x = accuracy.title.tolist()
all_y = accuracy.accuracy.tolist()

# ------ RUN CROSS VALIDATION ON MODEL SEVERAL TIMES WITH VARIOUS METRICS OF EVALUATION ------
# THIS FRAMEWORK OF EVALUATION IS BASED ON THAT USED IN MY 2020 SOFTWARE METHODOLOGIES, MACHINE LEARNING
# COURSEWORK
print('Validating')
score = cross_val_score(text_rfc_stemmed, all_x, all_y, n_jobs=-1)
print('Score done')
rmse = cross_val_score(text_rfc_stemmed, all_x, all_y, scoring='neg_mean_squared_error', n_jobs=-1)
print('Rmse done')
cv_accuracy = cross_val_score(text_rfc_stemmed, all_x, all_y, scoring='accuracy', n_jobs=-1)
print('Acc done')
print("Score: ", score.mean(), "\tRMSE: ", (rmse.mean() * -1.0), "\tAccuracy: ", cv_accuracy.mean())

# ----- PLOTS CONFUSION MATRIX FOR CALCULATION OF PRECISION, RECALL ETC *SEE REPORT* --------
print('Plotting confusion matrix')
cvp = cross_val_predict(text_rfc_stemmed, all_x, all_y)
print("Confusion matrix:\n", confusion_matrix(all_y, cvp))
print()

# ----- PLOTS ROC CURVE FOR EVALUATION OF CLASSIFIER PERFORMANCE *SEE REPORT* -----------
# https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
print('Plotting ROC curve')
ys = cross_val_predict(text_rfc_stemmed, all_x, all_y, method='predict_proba')
ys = 1 - ys
ys = ys[:, 0]
fpr, tpr, thresholds = roc_curve(all_y, ys)
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
fname = 'HEADLINE_CLASSIFIER' + '_' + str(time.time()) + '.png'
plt.savefig(fname)
plt.clf()
print("-- ROC graph saved to: ", fname)
print()

