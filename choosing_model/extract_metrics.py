# importing libraries
import time
from tqdm import tqdm

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

# reading data
df = pd.read_csv('hotel_bookings.csv')

# checking for null values 
null = pd.DataFrame({'Null Values' : df.isna().sum(), 'Percentage Null Values' : (df.isna().sum()) / (df.shape[0]) * (100)})

# filling null values with zero
df.fillna(0, inplace = True)

# adults, babies and children cant be zero at same time, so dropping the rows having all these zero at same time
filter = (df.children == 0) & (df.adults == 0) & (df.babies == 0)
df = df[~filter]


# ----------------------------------- Data Pre Processing ---------------------------------------------- #

correlation = df.corr()['is_canceled'].abs().sort_values(ascending = False)

# dropping columns that are not useful

useless_col = ['days_in_waiting_list', 'arrival_date_year', 'arrival_date_year', 'assigned_room_type', 'booking_changes',
               'reservation_status', 'country', 'days_in_waiting_list']
df.drop(useless_col, axis = 1, inplace = True)


# creating numerical and categorical dataframes

cat_cols = [col for col in df.columns if df[col].dtype == 'O']

cat_df = df[cat_cols]

cat_df['reservation_status_date'] = pd.to_datetime(cat_df['reservation_status_date'])

cat_df['year'] = cat_df['reservation_status_date'].dt.year
cat_df['month'] = cat_df['reservation_status_date'].dt.month
cat_df['day'] = cat_df['reservation_status_date'].dt.day

cat_df.drop(['reservation_status_date','arrival_date_month'] , axis = 1, inplace = True)


# encoding categorical variables

cat_df['hotel'] = cat_df['hotel'].map({'Resort Hotel' : 0, 'City Hotel' : 1})

cat_df['meal'] = cat_df['meal'].map({'BB' : 0, 'FB': 1, 'HB': 2, 'SC': 3, 'Undefined': 4})

cat_df['market_segment'] = cat_df['market_segment'].map({'Direct': 0, 'Corporate': 1, 'Online TA': 2, 'Offline TA/TO': 3,
                                                           'Complementary': 4, 'Groups': 5, 'Undefined': 6, 'Aviation': 7})

cat_df['distribution_channel'] = cat_df['distribution_channel'].map({'Direct': 0, 'Corporate': 1, 'TA/TO': 2, 'Undefined': 3,
                                                                       'GDS': 4})

cat_df['reserved_room_type'] = cat_df['reserved_room_type'].map({'C': 0, 'A': 1, 'D': 2, 'E': 3, 'G': 4, 'F': 5, 'H': 6,
                                                                   'L': 7, 'B': 8})

cat_df['deposit_type'] = cat_df['deposit_type'].map({'No Deposit': 0, 'Refundable': 1, 'Non Refund': 3})

cat_df['customer_type'] = cat_df['customer_type'].map({'Transient': 0, 'Contract': 1, 'Transient-Party': 2, 'Group': 3})

cat_df['year'] = cat_df['year'].map({2015: 0, 2014: 1, 2016: 2, 2017: 3})


num_df = df.drop(columns = cat_cols, axis = 1)
num_df.drop('is_canceled', axis = 1, inplace = True)

# normalizing numerical variables
num_df['lead_time'] = np.log(num_df['lead_time'] + 1)
num_df['arrival_date_week_number'] = np.log(num_df['arrival_date_week_number'] + 1)
num_df['arrival_date_day_of_month'] = np.log(num_df['arrival_date_day_of_month'] + 1)
num_df['agent'] = np.log(num_df['agent'] + 1)
num_df['company'] = np.log(num_df['company'] + 1)
num_df['adr'] = np.log(num_df['adr'] + 1)
num_df['adr'] = num_df['adr'].fillna(value = num_df['adr'].mean())


X = pd.concat([cat_df, num_df], axis = 1)
y = df['is_canceled']

# # ----------------------------------- Models and Cross-Validation ---------------------------------------------- #


metrics_names = ["accuracy", "recall", "precision", "f1", "training_time", "predict_time"]
metrics = {}
for metric in metrics_names:
    metrics[metric] = pd.DataFrame({"logistic_regression": [], "knn": [], "decision_tree": [], "random_forest": [], "ada_boost": [], "gradient_boost": [], "xgboost": [], "cat_boost": [], "extra_trees": [], "lgbm": [], "voting": []})

for i in tqdm(range(50)):
    # splitting data into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=i)

    # ----------------------------------- Logistic Regression ---------------------------------------------- #

    start = time.time()
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    training_time_lr = time.time() - start

    start = time.time()
    y_pred_lr = lr.predict(X_test)
    predict_time_lr = time.time() - start

    acc_lr = accuracy_score(y_test, y_pred_lr)
    recall_lr = recall_score(y_test, y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)   

    # ----------------------------------- KNN ---------------------------------------------- #

    start = time.time()
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    training_time_knn = time.time() - start

    start = time.time()
    y_pred_knn = knn.predict(X_test)
    predict_time_knn = time.time() - start

    acc_knn = accuracy_score(y_test, y_pred_knn)
    recall_knn = recall_score(y_test, y_pred_knn)
    prec_knn = precision_score(y_test, y_pred_knn)
    f1_knn = f1_score(y_test, y_pred_knn)

    # ----------------------------------- Decision Tree Classifier ---------------------------------------------- #

    start = time.time()
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    training_time_dtc = time.time() - start

    start = time.time()
    y_pred_dtc = dtc.predict(X_test)
    predict_time_dtc = time.time() - start

    acc_dtc = accuracy_score(y_test, y_pred_dtc)
    recall_dtc = recall_score(y_test, y_pred_dtc)
    prec_dtc = precision_score(y_test, y_pred_dtc)
    f1_dtc = f1_score(y_test, y_pred_dtc)

    # ----------------------------------- Random Forest Classifier ---------------------------------------------- #

    start = time.time()
    rd_clf = RandomForestClassifier()
    rd_clf.fit(X_train, y_train)
    training_time_rfc = time.time() - start

    start = time.time()
    y_pred_rd_clf = rd_clf.predict(X_test)
    predict_time_rfc = time.time() - start

    acc_rfc = accuracy_score(y_test, y_pred_rd_clf)
    recall_rfc = recall_score(y_test, y_pred_rd_clf)
    prec_rfc = precision_score(y_test, y_pred_rd_clf)
    f1_rfc = f1_score(y_test, y_pred_rd_clf)

    # ----------------------------------- Ada Boost Classifier ---------------------------------------------- #

    start = time.time()
    ada = AdaBoostClassifier(base_estimator = dtc)
    ada.fit(X_train, y_train)
    training_time_abc = time.time() - start

    start = time.time()
    y_pred_ada = ada.predict(X_test)
    predict_time_abc = time.time() - start

    acc_abc = accuracy_score(y_test, y_pred_ada)
    recall_abc = recall_score(y_test, y_pred_ada)
    prec_abc = precision_score(y_test, y_pred_ada)
    f1_abc = f1_score(y_test, y_pred_ada)

    # ----------------------------------- Gradient Boosting Classifier ---------------------------------------------- #

    start = time.time()
    gb = GradientBoostingClassifier()
    gb.fit(X_train, y_train)
    training_time_gbc = time.time() - start

    start = time.time()
    y_pred_gb = gb.predict(X_test)
    predict_time_gbc = time.time() - start

    acc_gbc = accuracy_score(y_test, y_pred_gb)
    recall_gbc = recall_score(y_test, y_pred_gb)
    prec_gbc = precision_score(y_test, y_pred_gb)
    f1_gbc = f1_score(y_test, y_pred_gb)

    # ----------------------------------- XgBoost ---------------------------------------------- #

    start = time.time()
    xgb = XGBClassifier(booster = 'gbtree', learning_rate = 0.1, max_depth = 5, n_estimators = 180)
    xgb.fit(X_train, y_train)
    training_time_xg = time.time() - start

    start = time.time()
    y_pred_xgb = xgb.predict(X_test)
    predict_time_xg = time.time() - start

    acc_xg = accuracy_score(y_test, y_pred_xgb)
    recall_xg = recall_score(y_test, y_pred_xgb)
    prec_xg = precision_score(y_test, y_pred_xgb)
    f1_xg = f1_score(y_test, y_pred_xgb)

    # ----------------------------------- Cat Boost Classifier ---------------------------------------------- #

    start = time.time()
    cat = CatBoostClassifier(iterations=100, verbose=False)
    cat.fit(X_train, y_train)
    training_time_cbc = time.time() - start

    start = time.time()
    y_pred_cat = cat.predict(X_test)
    predict_time_cbc = time.time() - start

    acc_cbc = accuracy_score(y_test, y_pred_cat)
    recall_cbc = recall_score(y_test, y_pred_cat)
    prec_cbc = precision_score(y_test, y_pred_cat)
    f1_cbc = f1_score(y_test, y_pred_cat)

    # ----------------------------------- Extra Trees Classifier ---------------------------------------------- #

    start = time.time()
    etc = ExtraTreesClassifier()
    etc.fit(X_train, y_train)
    training_time_etc = time.time() - start

    start = time.time()
    y_pred_etc = etc.predict(X_test)
    predict_time_etc = time.time() - start

    acc_etc = accuracy_score(y_test, y_pred_etc)
    recall_etc = recall_score(y_test, y_pred_etc)
    prec_etc = precision_score(y_test, y_pred_etc)
    f1_etc = f1_score(y_test, y_pred_etc)

    # ----------------------------------- LGBM Classifier ---------------------------------------------- #

    start = time.time()
    lgbm = LGBMClassifier(learning_rate = 1)
    lgbm.fit(X_train, y_train)
    training_time_lgbm = time.time() - start

    start = time.time()
    y_pred_lgbm = lgbm.predict(X_test)
    predict_time_lgbm = time.time() - start

    acc_lgbm = accuracy_score(y_test, y_pred_lgbm)
    recall_lgbm = recall_score(y_test, y_pred_lgbm)
    prec_lgbm = precision_score(y_test, y_pred_lgbm)
    f1_lgbm = f1_score(y_test, y_pred_lgbm)

    # ----------------------------------- Voting Classifier ---------------------------------------------- #

    start = time.time()
    classifiers = [('Gradient Boosting Classifier', gb), ('Cat Boost Classifier', cat), ('XGboost', xgb),  ('Decision Tree', dtc),
                   ('Extra Tree', etc), ('Light Gradient', lgbm), ('Random Forest', rd_clf), ('Ada Boost', ada), ('Logistic', lr),
                   ('Knn', knn)]
    vc = VotingClassifier(estimators = classifiers)
    vc.fit(X_train, y_train)
    training_time_vc = time.time() - start

    start = time.time()
    y_pred_vc = vc.predict(X_test)
    predict_time_vc = time.time() - start

    acc_vc = accuracy_score(y_test, y_pred_vc)
    recall_vc = recall_score(y_test, y_pred_vc)
    prec_vc = precision_score(y_test, y_pred_vc)
    f1_vc = f1_score(y_test, y_pred_vc)

    metrics["accuracy"] = metrics["accuracy"].append({"logistic_regression": acc_lr, "knn": acc_knn, "decision_tree": acc_dtc, "random_forest": acc_rfc, "ada_boost": acc_abc, "gradient_boost": acc_gbc, "xgboost": acc_xg, "cat_boost": acc_cbc, "extra_trees": acc_etc, "lgbm": acc_lgbm, "voting": acc_vc}, ignore_index=True)
    metrics["recall"] = metrics["recall"].append({"logistic_regression": recall_lr, "knn": recall_knn, "decision_tree": recall_dtc, "random_forest": recall_rfc, "ada_boost": recall_abc, "gradient_boost": recall_gbc, "xgboost": recall_xg, "cat_boost": recall_cbc, "extra_trees": recall_etc, "lgbm": recall_lgbm, "voting": acc_vc}, ignore_index=True)
    metrics["precision"] = metrics["precision"].append({"logistic_regression": prec_lr, "knn": prec_knn, "decision_tree": prec_dtc, "random_forest": prec_rfc, "ada_boost": prec_abc, "gradient_boost": prec_gbc, "xgboost": prec_xg, "cat_boost": prec_cbc, "extra_trees": prec_etc, "lgbm": prec_lgbm, "voting": prec_vc}, ignore_index=True)
    metrics["f1"] = metrics["f1"].append({"logistic_regression": f1_lr, "knn": f1_knn, "decision_tree": f1_dtc, "random_forest": f1_rfc, "ada_boost": f1_abc, "gradient_boost": f1_gbc, "xgboost": f1_xg, "cat_boost": f1_cbc, "extra_trees": f1_etc, "lgbm": f1_lgbm, "voting": f1_vc}, ignore_index=True)
    metrics["training_time"] = metrics["training_time"].append({"logistic_regression": training_time_lr, "knn": training_time_knn, "decision_tree": training_time_dtc, "random_forest": training_time_rfc, "ada_boost": training_time_abc, "gradient_boost": training_time_gbc, "xgboost": training_time_xg, "cat_boost": training_time_cbc, "extra_trees": training_time_etc, "lgbm": training_time_lgbm, "voting": training_time_vc}, ignore_index=True)
    metrics["predict_time"] = metrics["predict_time"].append({"logistic_regression": predict_time_lr, "knn": predict_time_knn, "decision_tree": predict_time_dtc, "random_forest": predict_time_rfc, "ada_boost": predict_time_abc, "gradient_boost": predict_time_gbc, "xgboost": predict_time_xg, "cat_boost": predict_time_cbc, "extra_trees": predict_time_etc, "lgbm": predict_time_lgbm, "voting": predict_time_vc}, ignore_index=True)

    # saving a new csv every iteration so that if theres any problem in the run we don't lose any data
    for ind, metric in enumerate(metrics):
        metrics[metric].to_csv(metric + '.csv', index=False)


# ANN is commented and separated cause as it takes a lot longer to train than the other models
# I trained it separately

# # ----------------------------------- ANN ---------------------------------------------- #

# metrics_names = ["accuracy", "recall", "precision", "f1", "training_time", "predict_time"]
# metrics = {}
# for metric in metrics_names:
#     metrics[metric] = pd.DataFrame({"ann": []})

# for i in tqdm(range(50)):
#     start = time.time()
#     from tensorflow.keras.utils import to_categorical

#     X = pd.concat([cat_df, num_df], axis = 1)
#     y = to_categorical(df['is_canceled'])

#     # splitting data into training set and test set

#     X_train_ann, X_test_ann, y_train_ann, y_test_ann = train_test_split(X, y, test_size = 0.30, random_state=i)


#     import keras
#     from keras.layers import Dense
#     from keras.models import Sequential

#     model  = Sequential()
#     model.add(Dense(100, activation = 'relu', input_shape = (26, )))
#     model.add(Dense(100, activation = 'relu'))
#     model.add(Dense(2, activation = 'sigmoid'))
#     model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#     start = time.time()
#     model_history = model.fit(X_train_ann, y_train_ann, validation_data = (X_test_ann, y_test_ann),
#                               epochs = 100)
#     training_time_ann = time.time() - start

#     # acc_ann = model.evaluate(X_test_ann, y_test_ann)[1]
#     y_test_ann = np.argmax(y_test_ann, axis=1) # getting argmax to reverse the one-hot encoding
#     start = time.time()
#     y_pred_ann = np.argmax(np.rint(model.predict(X_test_ann)), axis=1) # with rounding to int to get actual classifications and getting argmax to reverse the one-hot encoding
#     predict_time_ann = time.time() - start
#     acc_ann = accuracy_score(y_test_ann, y_pred_ann)
#     recall_ann = recall_score(y_test_ann, y_pred_ann)
#     prec_ann = precision_score(y_test_ann, y_pred_ann)
#     f1_ann = f1_score(y_test_ann, y_pred_ann)

#     metrics["accuracy"] = metrics["accuracy"].append({"ann": acc_ann}, ignore_index=True)
#     metrics["recall"] = metrics["recall"].append({"ann": recall_ann}, ignore_index=True)
#     metrics["precision"] = metrics["precision"].append({"ann": prec_ann}, ignore_index=True)
#     metrics["f1"] = metrics["f1"].append({"ann": f1_ann}, ignore_index=True)
#     metrics["training_time"] = metrics["training_time"].append({"ann": training_time_ann}, ignore_index=True)
#     metrics["predict_time"] = metrics["predict_time"].append({"ann": predict_time_ann}, ignore_index=True)

#     # saving a new csv every iteration so that if theres any problem in the run we don't lose any data
#     for ind, metric in enumerate(metrics):
#         metrics[metric].to_csv(metric + '_ann.csv', index=False)