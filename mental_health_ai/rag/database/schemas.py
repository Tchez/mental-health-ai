from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Metadata(BaseModel):
    type: str = Field(
        ..., description='Type of the document (e.g., article, blog post)'
    )
    source: Optional[str] = Field(
        None, description='The source of the document (e.g., URL, site name)'
    )
    page_number: Optional[int] = Field(
        None, description='The page number of the document'
    )
    source_description: Optional[str] = Field(
        None, description='Description of the source'
    )
    date: Optional[datetime] = Field(
        None, description='The publication date of the document'
    )

    @field_validator('type')
    def type_must_not_be_empty(cls, value):
        if not value.strip():
            raise ValueError('Type must not be empty')
        return value

    @field_validator('source', 'source_description')
    def not_empty_if_provided(cls, value):
        if value is not None and not value.strip():
            raise ValueError('Field must not be empty if provided')
        return value


class WeaviateDocument(BaseModel):
    title: str = Field(..., description='Title of the document')
    page_content: str = Field(..., description='The content of the document')
    metadata: Metadata = Field(..., description='Metadata of the document')

    @field_validator('title', 'page_content')
    def not_empty(cls, value):
        if not value.strip():
            raise ValueError('Field must not be empty')
        return value


class DataModel(BaseModel):
    documents: List[WeaviateDocument]

    @field_validator('documents')
    def documents_must_not_be_empty(cls, value):
        if not value:
            raise ValueError('Documents list must not be empty')
        return value
