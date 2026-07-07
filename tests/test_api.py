# tests/test_api.py
"""
PyTest unit verification test suite evaluating API router endpoints 
and schema data type validation boundaries using lifespan context hooks.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture(scope="module")
def client():
    """
    Module-scoped test client fixture that forces FastAPI to execute its 
    async lifespan startup events, ensuring memory artifacts are fully initialized.
    """
    with TestClient(app) as test_client:
        yield test_client

def test_api_valid_underwriting_inference(client):
    """Verify that submitting a valid request payload outputs expected model configurations."""
    mock_valid_payload = {
        "Id": 77,
        "Income": 85000.0,
        "Age": 33,
        "Experience": 12,
        "Married/Single": "married",
        "House_Ownership": "owned",
        "Car_Ownership": "yes",
        "Profession": "Manager",
        "CITY": "Seattle",
        "STATE": "Washington",
        "CURRENT_JOB_YRS": 5,
        "CURRENT_HOUSE_YRS": 4
    }
    response = client.post("/api/v1/predict", json=mock_valid_payload)
    assert response.status_code == 200
    
    response_json = response.json()
    assert response_json["Id"] == 77
    assert "underwriting_decision" in response_json
    assert "approval_probability" in response_json
    assert isinstance(response_json["primary_risk_drivers"], list)

def test_api_schema_boundary_failure(client):
    """Verify that input parameters breaking Pydantic numeric criteria return 422 errors."""
    mock_invalid_payload = {
        "Id": 88,
        "Income": -100.0,  # Out of bounds constraint rule breach
        "Age": 15,         # Out of bounds constraint rule breach
        "Experience": 0,
        "Married/Single": "single",
        "House_Ownership": "rented",
        "Car_Ownership": "no",
        "Profession": "Intern",
        "CITY": "Denver",
        "STATE": "Colorado",
        "CURRENT_JOB_YRS": 0,
        "CURRENT_HOUSE_YRS": 1
    }
    response = client.post("/api/v1/predict", json=mock_invalid_payload)
    assert response.status_code == 422