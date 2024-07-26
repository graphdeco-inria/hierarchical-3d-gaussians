from flask import Blueprint, jsonify

api = Blueprint("api", __name__)

from . import sales, health_care_professionals


@api.errorhandler(404)
def handle_bad_request(e):
    return jsonify({"error": "The route is not defined"})
