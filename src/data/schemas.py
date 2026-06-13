from pydantic import BaseModel, validator
from datetime import datetime
from typing import Optional

class Patient(BaseModel):
    patient_id: str
    age_group: str  # '15-24','25-34','35-44','45+'
    fistula_type: str  # 'VVF','RVF','mixed'
    repair_date: datetime
    region: str

    @validator('age_group')
    def validate_age_group(cls, v):
        allowed = ['15-24','25-34','35-44','45+']
        if v not in allowed:
            raise ValueError(f'age_group must be one of {allowed}')
        return v

class SymptomReportIn(BaseModel):
    patient_id: str
    raw_message: str
    language: Optional[str] = 'en'  # default
    timestamp: Optional[datetime] = None

class SymptomReportOut(BaseModel):
    report_id: int
    patient_id: str
    raw_message: str
    language: str
    timestamp: datetime
    predicted_class: Optional[int] = None
    prediction_confidence: Optional[float] = None
    action_taken: Optional[str] = None

class PredictionRequest(BaseModel):
    patient_id: str
    message: str
    language: str = 'en'

class PredictionResponse(BaseModel):
    report_id: int
    triage_class: int  # 0=routine,1=watchful,2=urgent
    confidence: float
    action_recommended: str
    explanation: Optional[str] = None