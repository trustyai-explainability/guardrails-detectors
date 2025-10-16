import pytest
from fastapi.testclient import TestClient

class TestRegexDetectors:
    @pytest.fixture
    def client(self):
        from detectors.built_in.app import app
        from detectors.built_in.regex_detectors import RegexDetectorRegistry

        app.set_detector(RegexDetectorRegistry(), "regex")

        return TestClient(app)

    @pytest.mark.parametrize(
        "regex,content,expected",
        [
            ("email", "Contact me at test@example.com", "test@example.com"),
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
            ("email", "Contact me at test@exame.c "), # invalid luhn card
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

    @pytest.mark.parametrize(
        "content,expected",
        [
            # Visa
            ("Card: 4111 1111 1111 1111", "4111 1111 1111 1111"),
            ("Card: 4111-1111-1111-1111", "4111-1111-1111-1111"),
            ("Card: 4111111111111111", "4111111111111111"),
            # MasterCard
            ("Card: 5555 5555 5555 4444", "5555 5555 5555 4444"),
            ("Card: 5555-5555-5555-4444", "5555-5555-5555-4444"),
            ("Card: 5555555555554444", "5555555555554444"),
            # Amex
            ("Card: 3782 822463 10005", "3782 822463 10005"),
            ("Card: 3782-822463-10005", "3782-822463-10005"),
            ("Card: 378282246310005", "378282246310005"),
            # Discover
            ("Card: 6011 1111 1111 1117", "6011 1111 1111 1117"),
            ("Card: 6011-1111-1111-1117", "6011-1111-1111-1117"),
            ("Card: 6011111111111117", "6011111111111117"),
            # Diners Club
            ("Card: 3056 930902 5904", "3056 930902 5904"),
            ("Card: 3056-930902-5904", "3056-930902-5904"),
            ("Card: 30569309025904", "30569309025904"),
            # JCB
            ("Card: 3530 1113 3330 0000", "3530 1113 3330 0000"),
            ("Card: 3530-1113-3330-0000", "3530-1113-3330-0000"),
            ("Card: 3530111333300000", "3530111333300000"),
        ]
    )
    def test_credit_card_detector_patterns(self, client, content, expected):
        payload = {
            "contents": [content],
            "detector_params": {"regex": ["credit-card"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        found = [d["text"] for d in resp.json()[0]]
        assert expected in found

    @pytest.mark.parametrize(
        "content",
        [
            # Invalid Luhn
            "Card: 4111 1111 1111 1112",
            "Card: 5555-5555-5555-4440",
            "Card: 3782 822463 10006",
            "Card: 6011-1111-1111-1110",
            "Card: 3056-930902-5900",
            "Card: 3530-1113-3330-0001",
            # Not enough digits
            "Card: 4111 1111 1111",
            "Card: 5555-5555-5555",
            "Card: 3782 822463",
            "Card: 6011-1111-1111",
            "Card: 3056-930902",
            "Card: 3530-1113-3330",
        ]
    )
    def test_credit_card_detector_invalid(self, client, content):
        payload = {
            "contents": [content],
            "detector_params": {"regex": ["credit-card"]}
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

    def test_single_detector(self, client):
        payload = {
            "contents": [
                "Email: a@b.com",
                "SSN: 123-45-6789"
            ],
            "detector_params": {"regex": "email"}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        results = resp.json()
        assert any("a@b.com" in d["text"] for d in results[0])


    # === ERROR HANDLING & INVALID DETECTOR TYPES =================================================    
    def test_unregistered_detector_kind_ignored(self, client):
        """Test that requesting an unregistered detector kind fails"""
        payload = {
            "contents": ["test@example.com"],
            "detector_params": {"nonexistent_detector": ["some_value"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 400

    def test_mixed_valid_invalid_detector_kinds(self, client):
        """Test mixing valid and invalid detector kinds"""
        payload = {
            "contents": ["Contact me at test@example.com"],
            "detector_params": {
                "regex": ["email"],  # valid detector kind
                "nonexistent_detector": ["some_value"]  # invalid detector kind
            }
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 400
        

    def test_empty_detector_params(self, client):
        """Test with empty detector_params"""
        payload = {
            "contents": ["test@example.com"],
            "detector_params": {}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        # Should return empty list since no detectors are specified
        assert resp.json()[0] == []

    def test_null_regex_pattern(self, client):
        """Test with null regex pattern"""
        payload = {
            "contents": ["test@example.com"],
            "detector_params": {"regex": [None]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 500  # Should cause an error when processing None

    def test_malformed_regex_groups(self, client):
        """Test malformed regex with unmatched groups"""
        payload = {
            "contents": ["test content"],
            "detector_params": {"regex": ["(unclosed group"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 500  # Should cause regex compilation error

