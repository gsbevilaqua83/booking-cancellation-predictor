import unittest
from http import client

from app import app, run_id


class ApiTest(unittest.TestCase):
    app = app
    api = "http://host.docker.internal:5001"
    index_endpoint = api + "/"
    run_id = run_id


    def update_currently_set_run_id(self, id):
        with open('current_model', 'w') as f:
            f.write(id)


    def test_1_no_input(self):
        client = self.app.test_client()
        r = client.post(self.index_endpoint)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json, {"error": "No input data provided"})


    def test_2_feature_names_mismatch(self):
        client = self.app.test_client()
        r = client.post(self.index_endpoint, json={"hotel":1})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json, {"error": "Feature names mismatch"})


    def test_3_unset_model(self):
        self.update_currently_set_run_id('') # setting run id to an invalid id
        client = self.app.test_client()
        r = client.post(self.index_endpoint, json={})
        self.update_currently_set_run_id(self.run_id) # set it back to the id it was before the test
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json, {"error": "No model set. Please run 'python set_model.py [run_id]' to set a model."})


    def test_4_predict(self):
        client = self.app.test_client()
        r = client.post(self.index_endpoint, json={"hotel":1,"meal":0,"market_segment":2,"distribution_channel":2,"reserved_room_type":2,"deposit_type":0,"customer_type":0,"year":3,"month":7,"day":6,"lead_time":4.31748811353631,"arrival_date_week_number":3.332204510175204,"arrival_date_day_of_month":1.6094379124341003,"stays_in_weekend_nights":0,"stays_in_week_nights":2,"adults":2,"children":0.0,"babies":0,"is_repeated_guest":0,"previous_cancellations":0,"previous_bookings_not_canceled":0,"agent":2.302585092994046,"company":0.0,"adr":5.017279836814924,"required_car_parking_spaces":0,"total_of_special_requests":0})
        self.assertEqual(r.status_code, 200)
        self.assertTrue("predictions" in r.json)


    def test_5_frontend(self):
        client = self.app.test_client()
        r = client.get(self.index_endpoint)
        self.assertEqual(r.status_code, 200)


    def test_6_invalid_input_data(self):
        client = self.app.test_client()
        r = client.post(self.index_endpoint, data="invalid")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json, {"error": "Could not json decode the input."})


    def test_7_invalid_input_json(self):
        client = self.app.test_client()
        r = client.post(self.index_endpoint, json='{"hotel":1,}')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json, {"error": "Could not json decode the input."})


    def test_8_predict_array_input(self):
        client = self.app.test_client()
        r = client.post(self.index_endpoint, json=[{"hotel":1,"meal":0,"market_segment":2,"distribution_channel":2,"reserved_room_type":2,"deposit_type":0,"customer_type":0,"year":3,"month":7,"day":6,"lead_time":4.31748811353631,"arrival_date_week_number":3.332204510175204,"arrival_date_day_of_month":1.6094379124341003,"stays_in_weekend_nights":0,"stays_in_week_nights":2,"adults":2,"children":0.0,"babies":0,"is_repeated_guest":0,"previous_cancellations":0,"previous_bookings_not_canceled":0,"agent":2.302585092994046,"company":0.0,"adr":5.017279836814924,"required_car_parking_spaces":0,"total_of_special_requests":0}])
        self.assertEqual(r.status_code, 200)
        self.assertTrue("predictions" in r.json)
        self.assertEqual(len(r.json["predictions"]), 1)


    def test_9_predict_array_multiple_input(self):
        client = self.app.test_client()
        r = client.post(self.index_endpoint, json=[{"hotel":1,"meal":0,"market_segment":2,"distribution_channel":2,"reserved_room_type":2,"deposit_type":0,"customer_type":0,"year":3,"month":7,"day":6,"lead_time":4.31748811353631,"arrival_date_week_number":3.332204510175204,"arrival_date_day_of_month":1.6094379124341003,"stays_in_weekend_nights":0,"stays_in_week_nights":2,"adults":2,"children":0.0,"babies":0,"is_repeated_guest":0,"previous_cancellations":0,"previous_bookings_not_canceled":0,"agent":2.302585092994046,"company":0.0,"adr":5.017279836814924,"required_car_parking_spaces":0,"total_of_special_requests":0},
                                                   {"hotel":1,"meal":0,"market_segment":2,"distribution_channel":2,"reserved_room_type":2,"deposit_type":0,"customer_type":0,"year":3,"month":7,"day":6,"lead_time":4.31748811353631,"arrival_date_week_number":3.332204510175204,"arrival_date_day_of_month":1.6094379124341003,"stays_in_weekend_nights":0,"stays_in_week_nights":2,"adults":2,"children":0.0,"babies":0,"is_repeated_guest":0,"previous_cancellations":0,"previous_bookings_not_canceled":0,"agent":2.302585092994046,"company":0.0,"adr":5.017279836814924,"required_car_parking_spaces":0,"total_of_special_requests":0}])
        self.assertEqual(r.status_code, 200)
        self.assertTrue("predictions" in r.json)
        self.assertEqual(len(r.json["predictions"]), 2)


if __name__ == "__main__":
    unittest.main()
