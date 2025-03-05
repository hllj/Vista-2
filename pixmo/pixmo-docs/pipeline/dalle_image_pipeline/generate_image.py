import os
import json
import tempfile

from datadreamer import DataDreamer
from datasets.fingerprint import Hasher
from datadreamer.steps import DataSource, SuperStep, zipped
from datadreamer.utils.fingerprint_utils import stable_fingerprint
from .dalle_helper import _generate_dalle_image_with_description, generate_dalle_image_with_description

NUM_RENDER_WORKERS = 24


class GenerateImage(SuperStep):
    CONFIG_HASH = Hasher.hash([
        stable_fingerprint(_generate_dalle_image_with_description),
        stable_fingerprint(generate_dalle_image_with_description),
    ])

    def setup(self):
        self.register_input(
            "metadata", required=True, help="The metadata used to generate the topics."
        )
        self.register_input("topic", required=True, help="The topics.")
        self.register_input("data", required=True, help="The description.")
        self.register_arg("llm", required=True, help="The LLM to use.")
        self.register_arg(
            "batch_size", required=True, help="The batch size to use with the LLM."
        )
        self.register_output("metadata")
        self.register_output("topic")
        self.register_output("data")
        self.register_output("code")
        self.register_output("image")

    def run(self):
        combined_inputs = DataSource(
            "Combine inputs",
            {
                "metadata": list(self.inputs["metadata"]),
                "topic": list(self.inputs["topic"]),
                "data": list(self.inputs["data"]),
            },
        )

        def generate_code(row):
            image_type = json.loads(row["metadata"])["image_type"]
            row["code"] = f"Generate a {image_type} based on the description: {row['data']}"
            return row

        # Combine with code
        combined = combined_inputs.map(
            generate_code,
            lazy=False,
            name="Combine with code",
        )

        # Generate Images
        def generate_image(row):
            try:
                row["image"] = generate_dalle_image_with_description(os.path.join(DataDreamer.get_output_folder_path(), ".cache"), self.args["llm"].api_key, row["data"])
            except Exception as e:
                print("Error:", e)
                row["image"] = None
            return row

        code_and_images = combined.map(
            generate_image,
            lazy=False,
            save_num_proc=NUM_RENDER_WORKERS,
            name="Generate Images",
        )

        # Remove any invalid images
        filtered = code_and_images.filter(
            lambda row: row["image"] is not None,
            lazy=False,
            name="Remove invalid images",
        )
        if filtered.output.num_rows < code_and_images.output.num_rows:
            self.logger.info(
                f"Warning: Could only generate valid images for {filtered.output.num_rows} out of {code_and_images.output.num_rows} total rows."
            )

        # Return result
        return filtered.output

    @property
    def version(self):
        return hash(GenerateImage.CONFIG_HASH)
