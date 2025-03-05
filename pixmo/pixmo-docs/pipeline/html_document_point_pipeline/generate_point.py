import tempfile
import os
import hashlib
import json
import random
from PIL import Image
from datasets.fingerprint import Hasher
from datadreamer.steps import DataSource, SuperStep, Prompt, zipped
from sqlitedict import SqliteDict
from filelock import FileLock
import dill
import faulthandler
import signal

from ..utils.render import render_html
from ..utils.utils import is_json_valid, extract_point_html, extract_points, process_image, modify_html, draw_points
from ..prompts.document_prompts import GENERATE_DOCUMENT_POINT_PROMPT, POINT_INTENTS, INTENT_PREFIXES

NUM_RENDER_WORKERS = 16
POINT_COLOR = "#72A0C1"

class GenerateDocumentPoint(SuperStep):
    CONFIG_HASH = Hasher.hash([GENERATE_DOCUMENT_POINT_PROMPT])

    def setup(self):
        self.register_input(
            "metadata", required=True, help="The metadata used to generate the topics."
        )
        self.register_input("topic", required=True, help="The topics.")
        self.register_input("data", required=True, help="The data.")
        self.register_input("code", required=True, help="The code.")
        self.register_input("image", required=True, help="The images.")
        self.register_arg("llm", required=True, help="The LLM to use.")
        self.register_arg(
            "batch_size", required=True, help="The batch size to use with the LLM."
        )
        self.register_output("metadata")
        self.register_output("topic")
        self.register_output("data")
        self.register_output("code")
        self.register_output("image")
        self.register_output("point_data")
        self.register_output("point_image")

    def run(self):
        combined_inputs = DataSource(
            "Combine inputs",
            {
                "metadata": list(self.inputs["metadata"]),
                "topic": list(self.inputs["topic"]),
                "data": list(self.inputs["data"]),
                "code": list(self.inputs["code"]),
                "image": list(self.inputs["image"]),
            },
        )

        # Create point prompts
        point_prompts_dataset = combined_inputs.map(
            lambda row: {
                "prompt": GENERATE_DOCUMENT_POINT_PROMPT.format(
                    topic=row["topic"], 
                    code=row["code"], 
                    figure_type=json.loads(row["metadata"])["figure_type"],
                    intent_type=random.choice(POINT_INTENTS),
                    prefix=random.choice(INTENT_PREFIXES),
                )
            },
            remove_columns=["image"],
            lazy=False,
            name="Create Generate Point Prompts",
        )

        # Generate Data
        generated_point = Prompt(
            name="Generate Data",
            inputs={
                "prompts": point_prompts_dataset.output["prompt"],
            },
            args={
                "llm": self.args["llm"],
                "batch_size": self.args["batch_size"],
                "post_process": extract_point_html,
                "temperature": 1.0,
                "top_p": 1.0,
            },
            outputs={
                "generations": "point_data",
            },
        ).select_columns(["point_data"], name="Get Generated Point Data")

        # Combine with generations with inputs
        combined = zipped(
            combined_inputs,
            generated_point,
            name="Combine with inputs",
        ).save(name="Save combine with inputs")
        
        # Register debugger for debugging frozen process
        faulthandler.register(signal.SIGUSR1.value)
        def process_point(row):

            # Register debugger for debugging frozen process
            if 'SETFAULTHANDLER' not in os.environ or os.environ['SETFAULTHANDLER'] != '1':
                faulthandler.register(signal.SIGUSR1.value)
            else:
                os.environ['SETFAULTHANDLER'] = '1'

            # Load result from cache if possible
            cache_lock = FileLock(
                os.path.join("/tmp/", "process_point_cache.flock")
            )
            cache_db = SqliteDict(
                os.path.join("/tmp/", "process_point_cache.db"), journal_mode="WAL", autocommit=True
            )
            hash = Hasher.hash(row)
            
            # Found the result in cache
            if hash in cache_db:
                cached = dill.loads(cache_db[hash])
                row["point_data"] = cached["point_data"]
                if cached["point_image_is_none"]:
                    row["point_image"] = None
                else:
                    row["point_image"] = draw_points(row["image"], [point["point_coordinates"][0] for point in json.loads(row["point_data"])])
                return row

            # Result was not in cache, compute it
            for i in range(len(row["point_data"])):
                try:
                    modified_html = modify_html(row["code"], row["point_data"][i]["modified_lines"])
                    point_image = render_html(modified_html, random_width=False)
                    point_coordinates = extract_points(point_image, point_color=POINT_COLOR)
                    row["point_data"][i]["point_coordinates"] = point_coordinates
                except Exception as e:
                    row["point_data"][i]["point_coordinates"] = None
                    continue

            # remove data that has no point coordinates
            row["point_data"] = [point for point in row["point_data"] if point["point_coordinates"] is not None]

            if len(row["point_data"]) == 0: row["point_image"] = None
            else: row["point_image"] = draw_points(row["image"], [point["point_coordinates"][0] for point in row["point_data"]])
            
            row["point_data"] = json.dumps(row["point_data"]) # convert the data column to a string

            # Save to cache for the future
            with cache_lock:
                cache_db[hash] = dill.dumps({
                    "point_data": row["point_data"],
                    "point_image_is_none": row["point_image"] is None
                })
                cache_db.commit()
            return row
        
        combined_processed = combined.map(
            process_point,
            lazy=False,
            save_num_proc=NUM_RENDER_WORKERS,
            name="Process Point Data",
        )

        # Remove any invalid rows
        filtered = combined_processed.filter(
            lambda row: row["point_image"] is not None,
            lazy=False,
            name="Remove invalid rows",
        )

        # Return result
        return filtered.output

    @property
    def version(self):
        hash = lambda x: int(hashlib.md5(x.encode('utf-8')).hexdigest(), 16)
        return hash(GenerateDocumentPoint.CONFIG_HASH)
