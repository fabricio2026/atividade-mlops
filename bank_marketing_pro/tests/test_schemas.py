import pytest
from pydantic import ValidationError

from app.schemas import CustomerFeatures

VALID_DATA = {
    "age": 35,
    "job": "admin.",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "duration": 200,
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp_var_rate": -1.8,
    "cons_price_idx": 92.893,
    "cons_conf_idx": -46.2,
    "euribor3m": 1.313,
    "nr_employed": 5099.1,
}


def test_valid_input():
    customer = CustomerFeatures(**VALID_DATA)
    assert customer.age == 35
    assert customer.job == "admin."


def test_invalid_age_too_low():
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**VALID_DATA, "age": 5})


def test_invalid_age_too_high():
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**VALID_DATA, "age": 150})


def test_invalid_job():
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**VALID_DATA, "job": "astronaut"})


def test_invalid_month():
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**VALID_DATA, "month": "january"})


def test_invalid_poutcome():
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**VALID_DATA, "poutcome": "unknown"})


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**VALID_DATA, "extra_field": "value"})


def test_emp_var_rate_out_of_range():
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**VALID_DATA, "emp_var_rate": 10.0})


def test_campaign_zero_rejected():
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**VALID_DATA, "campaign": 0})
