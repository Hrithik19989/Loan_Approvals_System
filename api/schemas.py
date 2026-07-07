# api/schemas.py
from pydantic import BaseModel, Field
from typing import Literal

class LoanApplicationSchema(BaseModel):
    Id: int = Field(..., description="Unique applicant record index.")
    Income: float = Field(..., ge=0, description="Annual continuous income in currency unit.")
    Age: int = Field(..., ge=18, le=100, description="Applicant chronological age.")
    Experience: int = Field(..., ge=0, le=60, description="Total years of professional experience.")
    Married_Single: Literal["married", "single"] = Field(..., alias="Married/Single", description="Marital status configuration.")
    House_Ownership: Literal["rented", "owned", "norent_noown"] = Field(..., description="Residential baseline.")
    Car_Ownership: Literal["yes", "no"] = Field(..., description="Automotive personal asset status.")
    Profession: str = Field(..., description="Stated occupational role or industry title.")
    CITY: str = Field(..., description="Residential municipality identifier.")
    STATE: str = Field(..., description="Regional/provincial zoning descriptor.")
    CURRENT_JOB_YRS: int = Field(..., ge=0, le=50, description="Tenure with the current employer.")
    CURRENT_HOUSE_YRS: int = Field(..., ge=0, le=50, description="Duration living in current property.")

    class Config:
        populate_by_name = True

class LoanPredictionResponse(BaseModel):
    Id: int
    risk_assessment: Literal["Approved (Low Risk)", "Rejected (High Risk)"]
    approval_probability: float
    top_risk_drivers: dict[str, float]
