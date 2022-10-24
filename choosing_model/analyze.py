import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# This script generates graphs and data to compare the models.
# It generates graphs of the distribution of the metrics over 50 runs
# of training/predicting for each model and rankings of the averages of those runs

sns.set_palette("Paired")

metrics_names = ["accuracy", "recall", "precision", "f1", "training_time", "predict_time"]
metrics = {}
for metric in metrics_names:
    metrics[metric] = pd.DataFrame({"logistic_regression": [], "knn": [], "decision_tree": [], "random_forest": [], "ada_boost": [], "gradient_boost": [], "xgboost": [], "cat_boost": [], "extra_trees": [], "lgbm": [], "voting": [], "ann": []})


metrics["accuracy"] = pd.read_csv(os.path.join('data', 'accuracy.csv'))[:50]
metrics["recall"] = pd.read_csv(os.path.join('data', 'recall.csv'))[:50]
metrics["precision"] = pd.read_csv(os.path.join('data', 'precision.csv'))[:50]
metrics["f1"] = pd.read_csv(os.path.join('data', 'f1.csv'))[:50]
metrics["training_time"] = pd.read_csv(os.path.join('data', 'training_time.csv'))[:50]
metrics["predict_time"] = pd.read_csv(os.path.join('data', 'predict_time.csv'))[:50]

metrics["accuracy"]['ann'] = pd.read_csv(os.path.join('data', 'accuracy_ann.csv'))
metrics["recall"]['ann'] = pd.read_csv(os.path.join('data', 'recall_ann.csv'))
metrics["precision"]['ann'] = pd.read_csv(os.path.join('data', 'precision_ann.csv'))
metrics["f1"]['ann'] = pd.read_csv(os.path.join('data', 'f1_ann.csv'))
metrics["training_time"]['ann'] = pd.read_csv(os.path.join('data', 'training_time_ann.csv'))
metrics["predict_time"]['ann'] = pd.read_csv(os.path.join('data', 'predict_time_ann.csv'))


# ------------- Ranks ------------------ #

averages = {}
for metric in metrics_names:
	averages[metric] = pd.DataFrame(data={'Model': ["logistic_regression", "knn", "decision_tree", "random_forest", "ada_boost", "gradient_boost", "xgboost", "cat_boost", "extra_trees", "lgbm", "voting", "ann"],
	                        	  		  'Average': [
	                        	  		  	metrics[metric]['logistic_regression'].mean(),
	                        	  		  	metrics[metric]['knn'].mean(),
	                        	  		  	metrics[metric]['decision_tree'].mean(),
	                        	  		  	metrics[metric]['random_forest'].mean(),
	                        	  		  	metrics[metric]['ada_boost'].mean(),
	                        	  		  	metrics[metric]['gradient_boost'].mean(),
	                        	  		  	metrics[metric]['xgboost'].mean(),
	                        	  		  	metrics[metric]['cat_boost'].mean(),
	                        	  		  	metrics[metric]['extra_trees'].mean(),
	                        	  		  	metrics[metric]['lgbm'].mean(),
	                        	  		  	metrics[metric]['voting'].mean(),
	                        	  		  	metrics[metric]['ann'].mean()
	                        	  		  ]})


for metric in metrics_names:
	if 'time' in metric:
		data = averages[metric].sort_values(by=['Average'])
		x = data['Model']
		y = data['Average']
		sns.barplot(x=x, y=y).set(title=metric.capitalize())
		plt.ylabel("Seconds", fontsize=18)
		plt.xlabel("Model", fontsize=18)
	else:
		data = averages[metric].sort_values(by=['Average'], ascending=False)
		x = data['Model']
		y = data['Average']
		sns.barplot(x=x, y=y).set(title=metric.capitalize())
		plt.ylabel("Score", fontsize=18)
		plt.xlabel("Model", fontsize=18)
	plt.show()



# ----------------- Distributions ------------------#

for ind, metric in enumerate(metrics):
    sns.kdeplot(metrics[metric]['logistic_regression'], label = "Logistic Regression")
    sns.kdeplot(metrics[metric]['knn'], label = "KNN")
    sns.kdeplot(metrics[metric]['decision_tree'], label = "Decision Tree")
    sns.kdeplot(metrics[metric]['random_forest'], label = "Random Forest")
    sns.kdeplot(metrics[metric]['ada_boost'], label = "Ada Boost")
    sns.kdeplot(metrics[metric]['gradient_boost'], label = "Gradient Boost")
    sns.kdeplot(metrics[metric]['xgboost'], label = "XgBoost")
    sns.kdeplot(metrics[metric]['cat_boost'], label = "Cat Boost")
    sns.kdeplot(metrics[metric]['extra_trees'], label = "Extra Trees")
    sns.kdeplot(metrics[metric]['lgbm'], label = "LGBM")
    sns.kdeplot(metrics[metric]['voting'], label = "Voting Classifier")
    sns.kdeplot(metrics[metric]['ann'], label = "ANN")
    plt.xlabel(metric.capitalize(), fontsize=18)
    plt.ylabel('Density', fontsize=18)
    plt.legend()
    plt.show()