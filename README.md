# booking-cancellation-predictor

This project extends the modeling and analisys within the following notebook by Nitesh Yadav: https://www.kaggle.com/code/niteshyadav3103/hotel-booking-prediction-99-5-acc

## Evaluating and Choosing a Model:

### Model Options:
```
Logistic Regression
Knn
Decision Tree Classifier
Random Forest Classifier
Ada Boost Classifier
Gradient Boosting Classifier
XgBoost
Cat Boost Classifier
Extra Trees Classifier
LGBM Classifier
ANN
Voting Classifier
```

To tackle the problem of choosing the best model I've considered not only the accuracy but also the following metrics: recall, precision, f1-score, training time and prediction time.
I've run the training and validation of all models for 50 runs each on distinct random splits of the data and plotted both the distributions of those runs as well as rankings for the average values for all metrics.
The resulting plots can be found below:

#### Accuracy
![accuracy_averages](https://user-images.githubusercontent.com/68133293/197448785-d28b0d25-faa3-4362-afbb-948b2b659b48.png)
![accuracy_distributions](https://user-images.githubusercontent.com/68133293/197448792-3137033f-c7c2-438f-80b1-bd85755b9ad5.png)

#### Recall
![recall_averages](https://user-images.githubusercontent.com/68133293/197448925-b6974c4e-bee2-474d-9b0e-c2b63e4acb27.png)
![recall_distributions](https://user-images.githubusercontent.com/68133293/197448945-392dad53-087a-4f9b-9fbc-c2f6945a4281.png)

#### Precision
![precision_averages](https://user-images.githubusercontent.com/68133293/197448971-14810b14-d833-41ae-8c8e-aeaf7cc7f054.png)
![precision_distributions](https://user-images.githubusercontent.com/68133293/197448986-8b1a0aa1-88f1-4daf-9faa-3dbba77ccf62.png)

#### F1-score
![f1_averages](https://user-images.githubusercontent.com/68133293/197449014-5a1ca90a-5149-4753-a773-8f148afd340c.png)
![f1_distributions](https://user-images.githubusercontent.com/68133293/197449023-b0009a6c-56ba-4150-8d1e-82e6cf69af4a.png)

#### Training Time
![training_time_averages](https://user-images.githubusercontent.com/68133293/197449050-e65f2aa7-0b98-419e-919b-3d28a5beb1d1.png)
![training_time_distributions](https://user-images.githubusercontent.com/68133293/197449060-cd828a65-d43c-4ff3-8981-ca8b67957601.png)

#### Prediction Time
![predict_time_averages](https://user-images.githubusercontent.com/68133293/197449076-0e5db9f3-a4f9-497c-8698-1449c48f5dcb.png)
![predict_time_distributions](https://user-images.githubusercontent.com/68133293/197449078-efde0a66-7584-4be5-965c-58dcf5b5ba09.png)

We can see from the accuracy, recall, precision and f1 plots that the Cat Boost Classifier is likely the most robust model. It only slightly falls behind XgBoost on precision while also having very low training and predicting times.
Therefore the Cat Boost Classifier was the model chosen for this project.

## Setup:

First clone the repository:
```
git clone https://github.com/gsbevilaqua83/booking-cancellation-predictor.git
```

Change to the root directory:
```
cd booking-cancellation-predictor
```

Run docker compose to start the API container and the mlflow container:
```
docker compose up
```

The mlflow server will be running at port 5000 and the API server be running at port 5001.
Both can be accessed through an internet browser at:
```
http://localhost:5000
```
and
```
http://localhost:5001
```


## Training the Model:

At the start of the containers we'll have no models to run. So first thing to do is to train a model. We'll do this by accessing the API container with docker and running the train script inside of it.

Check the id of the *booking-cancellation-predictor-api* container:
```
docker ps
```

Access the API container:
```
docker exec -it [container_id] /bin/bash
```

Then run the training script:
```
python train.py
```

By default the model will be trained with 100 iterations and the depth and random_strength parameters of CatBoostClassifier will be set to None. You can pass a list of values for each of those parameters and the train script will run for all the combinations of them:
```
python train.py --iterations 100 200 300 --depth 6 7 8 9 10 --random_strength 0 0.01 0.1 1
```
In the end this command will do *3 x 5 x 4 = 60* training runs for the model and will generate 60 experiments on MLFlow.

You can also set a seed for train/test data split with --data_split_seed:
```
python train.py --iterations 100 200 300 --depth 6 7 8 9 10 --random_strength 0 0.01 0.1 1 --data_split_seed 42
```
*42* is the default value for the seed.

The model will automatically be logged to MLFlow with the 4 parameters metioned above and the following metrics: accuracy, recall, precision, f1-score and roc auc score.

The API will only use models set to production. More specifically it will use the latest version of the *booking_cancellation-cat_boost_classifier* model the has been set to "Production" on MLFlow. The *booking_cancellation-cat_boost_classifier* model is automatically created at the initialization of the API and you now only need to regiter one of the training runs to it and set that version to "Production".

Access the MLFlow ui by going to *http://localhost:5000* and select the experiment you wish to use on the API:
![mlflow1](https://user-images.githubusercontent.com/68133293/197670181-b301a0f9-fd85-41ed-be39-a4920a5cd580.png)

Click the blue button to register the model:
![mlflow2](https://user-images.githubusercontent.com/68133293/197670588-869c08bd-7fca-4cba-b563-aeffa84c42f3.png)

Then select the *booking_cancellation-cat_boost_classifier* model:
![mlflow3](https://user-images.githubusercontent.com/68133293/197670716-a51ef61b-57ce-4021-89b4-a34b5c046cec.png)

Now go to the Models section of MLFlow and click the latest version area:
![mlflow4](https://user-images.githubusercontent.com/68133293/197670811-3b7100b7-9f24-4544-ba9d-4746987c8e59.png)

And then set the model to "Production":
![mlflow5](https://user-images.githubusercontent.com/68133293/197670864-a47ebc3e-f061-4ec8-afab-d718cc36392e.png)

And that will be the model used by the API.

By using the MLFlow Registry we can have a better model management and assure that only models that are ready will be used by the API

## Getting Predictions:

#### Option 1: CURL

Obs: I'm using curl on the following examples, however it should work with any tool that can make POST requests.

The url of the API is *http://127.0.0.1:5001* and the curl command needs to have the header parameter indicating that the data is json.

The proper command format can be seen below:
```
curl -X POST -H "Content-Type: application/json" -d '[json]' "http://127.0.0.1:5001"
```

A full example can be seen below:
```
curl -X POST -H "Content-Type: application/json" -d '{"hotel":1,"meal":0,"market_segment":2,"distribution_channel":2,"reserved_room_type":2,"deposit_type":0,"customer_type":0,"year":3,"month":7,"day":6,"lead_time":4.31748811353631,"arrival_date_week_number":3.332204510175204,"arrival_date_day_of_month":1.6094379124341003,"stays_in_weekend_nights":0,"stays_in_week_nights":2,"adults":2,"children":0.0,"babies":0,"is_repeated_guest":0,"previous_cancellations":0,"previous_bookings_not_canceled":0,"agent":2.302585092994046,"company":0.0,"adr":5.017279836814924,"required_car_parking_spaces":0,"total_of_special_requests":0}' "http://127.0.0.1:5001"
```

On Windows you may need to scape the double quotes on the json:
```
curl -X POST -H "Content-Type: application/json" -d "{\"hotel\":1,\"meal\":0,\"market_segment\":2,\"distribution_channel\":2,\"reserved_room_type\":2,\"deposit_type\":0,\"customer_type\":0,\"year\":3,\"month\":7,\"day\":6,\"lead_time\":4.31748811353631,\"arrival_date_week_number\":3.332204510175204,\"arrival_date_day_of_month\":1.6094379124341003,\"stays_in_weekend_nights\":0,\"stays_in_week_nights\":2,\"adults\":2,\"children\":0.0,\"babies\":0,\"is_repeated_guest\":0,\"previous_cancellations\":0,\"previous_bookings_not_canceled\":0,\"agent\":2.302585092994046,\"company\":0.0,\"adr\":5.017279836814924,\"required_car_parking_spaces\":0,\"total_of_special_requests\":0}" "http://127.0.0.1:5001"
```

The response of the api will also be a json with a unique key: "predictions" and the value is an array with the predictions:
```
{"predictions":[0]}
```

You can request to server multiple predictions by sending more than one row of data inside a list like below:
```
curl -X POST -H "Content-Type: application/json" -d '[{"hotel":1,"meal":0,"market_segment":2,"distribution_channel":2,"reserved_room_type":2,"deposit_type":0,"customer_type":0,"year":3,"month":7,"day":6,"lead_time":4.31748811353631,"arrival_date_week_number":3.332204510175204,"arrival_date_day_of_month":1.6094379124341003,"stays_in_weekend_nights":0,"stays_in_week_nights":2,"adults":2,"children":0.0,"babies":0,"is_repeated_guest":0,"previous_cancellations":0,"previous_bookings_not_canceled":0,"agent":2.302585092994046,"company":0.0,"adr":5.017279836814924,"required_car_parking_spaces":0,"total_of_special_requests":0},{"hotel":1,"meal":0,"market_segment":2,"distribution_channel":2,"reserved_room_type":2,"deposit_type":0,"customer_type":0,"year":3,"month":7,"day":6,"lead_time":4.31748811353631,"arrival_date_week_number":3.332204510175204,"arrival_date_day_of_month":1.6094379124341003,"stays_in_weekend_nights":0,"stays_in_week_nights":2,"adults":2,"children":0.0,"babies":0,"is_repeated_guest":0,"previous_cancellations":0,"previous_bookings_not_canceled":0,"agent":2.302585092994046,"company":0.0,"adr":5.017279836814924,"required_car_parking_spaces":0,"total_of_special_requests":0}]' "http://127.0.0.1:5001"
```

And so the predictions array will have multiple elements:
```
{"predictions":[0,0]}
```

#### Option 2: Internet Browser

You can also use the browser to access the api by goind to the url: *http://127.0.0.1:5001* or *http://localhost:5001*
There you'll find a textarea input where you can paste a json to request predictions:
![api1](https://user-images.githubusercontent.com/68133293/197666935-71b165f4-a30d-4b76-b26e-ae32e7cd43ea.png)

And the predictions will appear at the bottom:
![api2](https://user-images.githubusercontent.com/68133293/197666996-42570f3b-043c-449c-91a3-c19a4233806d.png)

## Unit Tests

To run the tests access the api container as it was done at the beggining of the *Training the Model* section. You will also need to have trained and added a model to production as well otherwise some of the api tests will surely fail.

And then simply run:
```
pytest
```
It will run both the tests for api as for the train script.
