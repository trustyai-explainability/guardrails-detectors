import pytest
from fastapi.testclient import TestClient

class TestXGBDetectors:
    @pytest.fixture
    def client(self):
        from detectors.xgb.build.app import app
        from detectors.xgb.build.detector import Detector

        app.set_detector(Detector(), "detector")
        return TestClient(app)

    @pytest.mark.parametrize(
        "content,expected",
        [
            (["Congratulations! You've won a $1000 Walmart gift card. Click here to claim now."], True),
            (["Don't forget to bring your notebook to class tomorrow."], False),
        ]
    )

    def test_xgb_detectors(self, client, content, expected):
        payload = {
            "content": [content],
        }
        resp = client.post("api/v1/text/contexts", json=payload)
        assert resp.status_code == 200
        assert len(resp.json()[0]) > 0
        assert resp.json()[0][0]['spam_check'] == expected

