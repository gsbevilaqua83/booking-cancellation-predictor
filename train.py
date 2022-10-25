import os
import warnings
import sys
from tqdm import tqdm
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier # Chosen model for this project

from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def preprocess():
    '''
        Preprocesses the data to be used on the model training.
        Handles null values, removes unnecessary columns and encodes categorical data.

        Returns:
            X (pandas.DataFrame): training data for the model
            y (pandas.Series) : ground truth of the training data
    '''

    # reading data
    df = pd.read_csv(os.path.join('data', 'hotel_bookings.csv'))

    # checking for null values 
    null = pd.DataFrame({'Null Values' : df.isna().sum(), 'Percentage Null Values' : (df.isna().sum()) / (df.shape[0]) * (100)})

    # filling null values with zero
    df.fillna(0, inplace = True)

    # adults, babies and children cant be zero at same time, so dropping the rows having all these zero at same time
    filter = (df.children == 0) & (df.adults == 0) & (df.babies == 0)
    df = df[~filter]

    corr = df.corr()
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

    return X, y

def eval_metrics(actual, pred):
    '''
        Evaluates accuracy, recall, precision, f1 score and roc for the
        given ground truth over what was predicted by the model

        Parameters:
            actual (pandas.Series): ground truth of the data
            pred (pandas.Series): predictions of the model

        Returns:
            acc (float): accuracy of the model
            recall (float): recall of the model
            prec (float): precision of the model
            f1 (float): f1 score of the model
            roc_auc (float): roc of the model
    '''
    acc = accuracy_score(actual, pred)
    recall = recall_score(actual, pred)
    prec = precision_score(actual, pred)
    f1 = f1_score(actual, pred)
    roc_auc = roc_auc_score(actual, pred)
    return acc, recall, prec, f1, roc_auc


if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # training parameters
    cli=argparse.ArgumentParser()
    cli.add_argument(
        "--iterations",  # name on the cli - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        type=int,
        default=[100],  # default if nothing is provided
    )
    cli.add_argument(
        "--depth",
        nargs="*",
        type=int,  # any type/callable can be used here
        default=[None],
    )
    cli.add_argument(
        "--random_strength",
        nargs="*",
        type=float,  # any type/callable can be used here
        default=[None],
    )
    cli.add_argument(
        "--data_split_seed",
        nargs=1,
        type=int,  # any type/callable can be used here
        default=42,
    )
    # parse the command line
    args = cli.parse_args()

    # gathering and processing data
    X, y = preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=args.data_split_seed[0])

    for iter in tqdm(args.iterations):
        for depth in args.depth:
            for rs in args.random_strength:
                with mlflow.start_run() as run:
                    cat = CatBoostClassifier(iterations=iter, random_strength=rs, depth=depth, verbose=False)
                    cat.fit(X_train, y_train)

                    y_pred = cat.predict(X_test)

                    (acc, recall, prec, f1, roc_auc) = eval_metrics(y_test, y_pred)

                    mlflow.log_param("iterations", iter)
                    mlflow.log_param("depth", depth)
                    mlflow.log_param("random_strength", rs)
                    mlflow.log_param("data_split_seed", args.data_split_seed[0])
                    mlflow.log_metric("acc", acc)
                    mlflow.log_metric("recall", recall)
                    mlflow.log_metric("prec", prec)
                    mlflow.log_metric("f1", f1)
                    mlflow.log_metric("roc_auc", roc_auc)

                    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                    mlflow.sklearn.log_model(cat, "model")

                    print("Run ID: ", run.info.run_id)
