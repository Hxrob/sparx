from pydantic import BaseModel


class StartRequest(BaseModel):
    transcript: str
    report_url: str


class FormField(BaseModel):
    field_id: str
    selector: str
    label: str
    type: str
    required: bool = False
    placeholder: str = ""
    options: list[str] = []
    current_value: str = ""


class PageContext(BaseModel):
    title: str = ""
    url: str = ""


class SuggestRequest(BaseModel):
    page_context: PageContext
    fields: list[FormField]


class FillItem(BaseModel):
    field_id: str
    value: str
    confidence: float = 0.0
    reason: str = ""


class SuggestResponse(BaseModel):
    assistant_message: str = ""
    known_facts: dict[str, str] = {}
    fills: list[FillItem] = []
    needs_user_input: list[str] = []
