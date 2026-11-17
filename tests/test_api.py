from fastapi.testclient import TestClient

from app.api import app


def test_home_renders():
    client = TestClient(app)
    result = client.get("/")
    assert result.status_code == 200
    assert "双语关键词一致性检查器" in result.text


def test_empty_input_is_validation_error():
    client = TestClient(app)
    result = client.post("/api/check", json={"english": "  ", "chinese": "目的"})
    assert result.status_code == 422
