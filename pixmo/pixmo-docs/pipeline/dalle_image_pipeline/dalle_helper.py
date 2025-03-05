import os
import json
import random
import warnings
import pandas as pd
from io import BytesIO

import openai
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential,
    stop_after_attempt,
)
from PIL import Image
from filelock import FileLock
from sqlitedict import SqliteDict
from datasets.fingerprint import Hasher
from datadreamer.utils.fingerprint_utils import stable_fingerprint


def _generate_dalle_image_with_description(api_key, description):
    client = openai.OpenAI(
        api_key=api_key,
    )
    rand = random.Random(description)
    response = client.images.generate(
        model="dall-e-3",
        prompt=description,
        n=1,
        style="natural",
        quality="hd",
        size=rand.choice(["1024x1024", "1792x1024", "1024x1792"]),
    )
    json_response = json.loads(response.model_dump_json())
    image_url = json_response["data"][0]["url"] 
    image_data = requests.get(image_url).content
    image = Image.open(BytesIO(image_data))
    return image

@retry(
    retry=retry_if_exception_type(openai.RateLimitError),
    wait=wait_exponential(multiplier=1, min=10, max=60),
    stop=stop_after_attempt(10),
    reraise=True,
)
@retry(
    retry=retry_if_exception_type(openai.InternalServerError),
    wait=wait_exponential(multiplier=1, min=3, max=300),
    stop=stop_after_attempt(10),
    reraise=True,
)
@retry(
    retry=retry_if_exception_type(openai.APIError),
    wait=wait_exponential(multiplier=1, min=3, max=300),
    stop=stop_after_attempt(10),
    reraise=True,
)
@retry(
    retry=retry_if_exception_type(openai.APIConnectionError),
    wait=wait_exponential(multiplier=1, min=3, max=300),
    stop=stop_after_attempt(10),
    reraise=True,
)
def _generate_dalle_image_with_description_wrapped_for_retries(api_key, description):
    return _generate_dalle_image_with_description(api_key, description)

def generate_dalle_image_with_description(cache_output_folder, api_key, description):
    key = Hasher.hash([stable_fingerprint(_generate_dalle_image_with_description), description])

    with FileLock(os.path.join(cache_output_folder, "dalle_cache.db.lock")):
        cache = SqliteDict(
            os.path.join(cache_output_folder, "dalle_cache.db"), journal_mode="WAL", autocommit=False
        )
        if key in cache:
            return cache[key]
    image = _generate_dalle_image_with_description_wrapped_for_retries(api_key, description)
    with FileLock(os.path.join(cache_output_folder, "dalle_cache.db.lock")):
            cache[key] = image
            cache.commit()
            return cache[key]