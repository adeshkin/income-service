from locust import HttpUser, between, task


class IncomeServiceUser(HttpUser):
    wait_time = between(1, 3)

    @task(4)
    def predict(self):
        payload = {
            "age": 47,
            "workclass": "Private",
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 45,
            "native_country": "United-States",
        }
        self.client.post("/v1/predict", json=payload)

    @task(1)
    def health(self):
        self.client.get("/health")
