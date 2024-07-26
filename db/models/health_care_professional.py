from sqlalchemy.sql import func
from sqlalchemy import text
from ..shared import db


class HealthcareProfessional(db.Model):
    __tablename__ = "HealthcareProfessionals"
    id = db.Column(db.Integer, primary_key=True)
    createdAt = db.Column(db.DateTime, default=func.now())
    updatedAt = db.Column(db.DateTime, default=func.now(), onupdate=func.now())

    @staticmethod
    def get_healthcare_professionals_with_gaps():
        return [
            row._asdict() 
            for row in db.session.execute(text("""
        SELECT
            pd.hcp AS hcp,
            COALESCE(pd.total_procedures, 0) - COALESCE(sd.total_sales, 0) AS gap
        FROM
            (SELECT hcp, SUM(volume) AS total_procedures
             FROM ProcedureData
             GROUP BY hcp) pd
        LEFT JOIN
            (SELECT hcp, SUM(volume) AS total_sales
             FROM SaleData
             GROUP BY hcp) sd
        ON pd.hcp = sd.hcp
        ORDER BY gap DESC, pd.hcp ASC
        """))
        ]