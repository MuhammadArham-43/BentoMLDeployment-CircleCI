import os
import bentoml
from pydantic import BaseModel, StringConstraints
from typing import Annotated
from openai import OpenAI

class PromptRequest(BaseModel):
    prompt: Annotated[
        str,
        StringConstraints(min_length=1, max_length=1000)
    ]

class CompletionResponse(BaseModel):
    completion: str


bentoml_image = bentoml.images.Image(
    python_version="3.10",
).requirements_file("./requirements.txt")

@bentoml.service(
    image=bentoml_image,
)
class LLMService:
    def __init__(self):
        self.client = OpenAI()
    
    @bentoml.api(input_spec=PromptRequest, output_spec=CompletionResponse)
    def generate(self, prompt: str) -> CompletionResponse:
        response = self.client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
            messages=[{"role": "user", "content": prompt}]
        )
        completion_text = response.choices[0].message.content.strip()
        return CompletionResponse(completion=completion_text)