import pytest
from fastapi.testclient import TestClient

class TestRegexDetectors:
    @pytest.fixture
    def client(self):
        from detectors.built_in.app import app
        return TestClient(app)

    @pytest.mark.parametrize(
        "regex,content,expected",
        [
            ("email", "Contact me at test@example.com", "test@example.com"),
            ("credit-card", "Card: 4111-1111-1111-1111", "4111-1111-1111-1111"),
            ("credit-card", "Card: 4111 1111 1111 1111", "4111 1111 1111 1111"),
            ("ipv4", "My IP is 192.168.1.1", "192.168.1.1"),
            ("us-social-security-number", "SSN: 123-45-6789", "123-45-6789"),
            ("us-social-security-number", "SSN: 123 45 6789", "123 45 6789"),
            ("uk-post-code", "Postcode: SW1A 1AA", "SW1A 1AA"),
            ("uk-post-code", "Postcode: W1A1AA", "W1A1AA"),
            ("us-phone-number", "Call (123) 456-7890", "(123) 456-7890"),
            ("us-phone-number", "Call 123-456-7890", "123-456-7890"),
            ("us-phone-number", "Call 123-456 7890", "123-456 7890"),
            ("us-phone-number", "Call +1 (123) 456-7890", "+1 (123) 456-7890"),
        ]
    )
    def test_builtin_regex_detectors(self, client, regex, content, expected):
        payload = {
            "contents": [content],
            "detector_params": {"regex": [regex]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        assert len(resp.json()[0]) > 0
        assert expected == resp.json()[0][0]['text']

    @pytest.mark.parametrize(
        "regex,content",
        [
            ("email", "Contact me at test@exame.c "),
            ("credit-card", "Card: 4111111111111111"),
            ("us-social-security-number", "SSN: 123456789"),
            ("us-phone-number", "Call 1234567890"),
        ]
    )
    def test_builtin_regex_detectors_should_not_match(self, client, regex, content):
        payload = {
            "contents": [content],
            "detector_params": {"regex": [regex]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        assert resp.json()[0] == []


    def test_multiple_regexes(self, client):
        payload = {
            "contents": ["Email: a@b.com, SSN: 123-45-6789"],
            "detector_params": {"regex": ["email", "us-social-security-number"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        texts = [d["text"] for d in resp.json()[0]]
        assert "a@b.com" in texts
        assert "123-45-6789" in texts

    def test_custom_regex(self, client):
        payload = {
            "contents": ["foo bar baz"],
            "detector_params": {"regex": [r"ba."]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        texts = [d["text"] for d in resp.json()[0]]
        assert "bar" in texts
        assert "baz" in texts

    def test_no_match(self, client):
        payload = {
            "contents": ["nothing to see here"],
            "detector_params": {"regex": ["email", "us-phone-number"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        assert resp.json()[0] == []

    def test_invalid_regex(self, client):
        payload = {
            "contents": ["foo"],
            "detector_params": {"regex": ["["]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 500
        data = resp.json()
        assert "message" in data

    def test_registry_endpoint(self, client):
        resp = client.get("/registry")
        assert resp.status_code == 200
        data = resp.json()
        assert "regex" in data
        assert "email" in data["regex"]
        assert "Detect email" in data["regex"]["email"]

    def test_multiple_contents(self, client):
        payload = {
            "contents": [
                "Email: a@b.com",
                "SSN: 123-45-6789"
            ],
            "detector_params": {"regex": ["email", "us-zip-code", "us-social-security-number"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        results = resp.json()
        assert any("a@b.com" in d["text"] for d in results[0])
        assert any("123-45-6789" in d["text"] for d in results[1])



