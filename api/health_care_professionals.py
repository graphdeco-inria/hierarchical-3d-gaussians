from flask import jsonify
from flask_pydantic import validate
from pydantic import BaseModel

from api import api
from db.models.health_care_professional import HealthcareProfessional

class HealthcareProfessionalQuery(BaseModel):
    limit: int
    strategy: str = 'elephant'

@api.route("/healthcare-professionals/top", methods=["GET"])
@validate()
def get_top_hcps(query: HealthcareProfessionalQuery):
    try:
        limit = query.limit
        strategy = query.strategy
        if strategy != 'elephant':
            return jsonify({"error": "Invalid strategy"}), 400
    
        hcps = HealthcareProfessional.get_healthcare_professionals_with_gaps()
        top_hcps = hcps[:limit]
        return jsonify([{"hcp": hcp["hcp"]} for hcp in top_hcps]), 200
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid input"}), 400
