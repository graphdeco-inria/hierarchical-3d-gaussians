def test_api_healthcare_professionals_limit_2(client):
    response = client.get("/api/healthcare_professionals/top?limit=2")
    assert response.status_code == 200
    assert len(response.json) == 2
