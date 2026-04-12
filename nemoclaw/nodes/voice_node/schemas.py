"""
Direction Engine — data shapes.

Categories sourced from NYC Open Data Benefits Platform:
  Benefits and Programs Dataset (kvhd-5fmu / ACCESS NYC taxonomy)
"""
from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class NYCBenefitCategory(str, Enum):
    cash_and_expenses      = "Cash and Expenses"
    food_and_nutrition     = "Food and Nutrition"
    housing                = "Housing"
    utilities              = "Utilities"
    emergency_crisis       = "Emergency and Crisis Assistance"
    health_care            = "Health Care"
    mental_health          = "Mental Health"
    work_and_employment    = "Work and Employment"
    education_and_training = "Education and Training"
    legal_services         = "Legal Services"
    civil_rights           = "Civil Rights Protections"
    child_care             = "Child Care and Early Education"
    children_and_youth     = "Children and Youth"
    older_adults           = "Older Adults"
    disability             = "Disability"
    immigrants             = "Immigrants and Refugees"
    veterans               = "Veterans"
    family_services        = "Family and Children Services"
    criminal_justice       = "Criminal Justice"


class RoutingDecision(str, Enum):
    categorized     = "categorized"      # clear category match, ready for OpenClaw
    uncategorized   = "uncategorized"    # real complaint but no category match — still send to OpenClaw
    needs_more_info = "needs_more_info"  # partially clear, ask follow-up
    irrelevant      = "irrelevant"       # small talk / off-topic, skip OpenClaw


class DirectionResult(BaseModel):
    """
    Structured output of the direction engine.
    This is the payload handed to OpenClaw — someone else wires the tools.
    A single transcript can map to multiple categories
    (e.g. lost housing + can't afford food = Housing + Food and Nutrition).
    """
    decision:     RoutingDecision
    categories:   list[NYCBenefitCategory] = Field(default_factory=list)
    question:     str
    summary:      str
    response:     str   # what to say back to the user in plain language
    confidence:   float = Field(ge=0.0, le=1.0)
    missing_info: list[str] = Field(default_factory=list)
    transcript:   str
