"""
Content Warning: Contains potentially offensive text dealing with racism, misogyny, and violence. Examples of input prompts provided purely for the purposes of testing HAP (Hate, Abuse and Profanity) models.
"""

from locust import HttpUser, between, task


class WebsiteUser(HttpUser):
    wait_time = between(1, 5)

    # def on_start(self):
    #     self.client.post("/login", {
    #         "username": "test_user",
    #         "password": ""
    #     })

    @task
    def docs(self):
        self.client.get("/docs")

    @task
    def api(self):
        self.client.get("/openapi.json")

    @task
    def pii(self):
        self.client.post(
            "/api/v1/text/contents?pii_transformer",
            json={
                "contents": [
                    "My name is John Doe and my social security number is 123-45-6789."
                ]
            },
            headers={"detector-id": "pii", "Content-Type": "application/json"},
        )
