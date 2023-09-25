from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel

ModelTestParamType = dict[str, Any]


@dataclass
class BaseModelTestCase:
    name: str
    input: ModelTestParamType
    output: ModelTestParamType


class BaseModelTest:
    def generate(self, input: ModelTestParamType) -> BaseModel:
        raise NotImplementedError

    def test_model(self, model_test_case: BaseModelTestCase):
        input = model_test_case.input
        output = model_test_case.output

        model_outputs = self.generate(input)
        sanitized_model_outputs = self.sanitize_output(model_outputs)
        assert sanitized_model_outputs == output, repr(sanitized_model_outputs)

    def sanitize_output(self, output: BaseModel) -> ModelTestParamType:
        sanitized_output = {}

        for field in output.__fields__:
            value = getattr(output, field)
            sanitized_value = self.sanitize_value(value)
            sanitized_output[field] = sanitized_value

        return sanitized_output

    def sanitize_value(self, value):
        # HACK: type comparison for fal.toolkit.File behaves unexpectedly.
        if hasattr(value, "url"):
            return calculate_file_hash(value.url)
        elif isinstance(value, (list, tuple, set)):
            value_type = type(value)
            return value_type(self.sanitize_value(item) for item in value)
        elif isinstance(value, dict):
            return self.sanitize_dictionary(value)
        else:
            return value

    def sanitize_dictionary(self, input_dict):
        sanitized_dict = {}
        for key, value in input_dict.items():
            sanitized_value = self.sanitize_value(value)
            sanitized_dict[key] = sanitized_value
        return sanitized_dict


def calculate_file_hash(file_url: str):
    with httpx.stream(method="GET", url=file_url) as response:
        if response.status_code != 200:
            raise ValueError(
                f"Failed to download file from {file_url}. "
                "Status code: {response.status_code}"
            )

        sha256_hash = hashlib.sha256()

        for chunk in response.iter_bytes(chunk_size=64 * 1024):
            sha256_hash.update(chunk)

        return sha256_hash.hexdigest()
