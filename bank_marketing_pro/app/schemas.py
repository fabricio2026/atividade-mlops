from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=18, le=100)
    job: Literal[
        "admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown",
    ]
    marital: Literal["divorced", "married", "single", "unknown"]
    education: Literal[
        "basic.4y", "basic.6y", "basic.9y", "high.school",
        "illiterate", "professional.course", "university.degree", "unknown",
    ]
    default: Literal["no", "yes", "unknown"]
    housing: Literal["no", "yes", "unknown"]
    loan: Literal["no", "yes", "unknown"]
    contact: Literal["cellular", "telephone"]
    month: Literal["jan", "feb", "mar", "apr", "may", "jun",
                   "jul", "aug", "sep", "oct", "nov", "dec"]
    day_of_week: Literal["mon", "tue", "wed", "thu", "fri"]
    duration: int = Field(ge=0, le=5000)
    campaign: int = Field(ge=1, le=50)
    pdays: int = Field(ge=0, le=999)
    previous: int = Field(ge=0, le=30)
    poutcome: Literal["failure", "nonexistent", "success"]
    emp_var_rate: float = Field(ge=-5.0, le=5.0)
    cons_price_idx: float = Field(ge=90.0, le=100.0)
    cons_conf_idx: float = Field(ge=-60.0, le=0.0)
    euribor3m: float = Field(ge=0.0, le=10.0)
    nr_employed: float = Field(ge=4000.0, le=6000.0)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    label: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    version: str


class InfoResponse(BaseModel):
    algorithm: str
    features: list[str]
    metrics: dict
    version: str
