import json

from datasets.fingerprint import Hasher
from datadreamer.steps import DataSource, SuperStep, Prompt, zipped

from ..prompts.image_prompts import GENERATE_IMAGE_DESCRIPTION_PROMPT
from ..utils.utils import extract_csv, is_csv_valid

class GenerateImageDescription(SuperStep):
    CONFIG_HASH = Hasher.hash([GENERATE_IMAGE_DESCRIPTION_PROMPT])

    def setup(self):
        self.register_input(
            "metadata", required=True, help="The metadata used to generate the topics."
        )
        self.register_input("topic", required=True, help="The topics.")
        self.register_arg("llm", required=True, help="The LLM to use.")
        self.register_arg(
            "batch_size", required=True, help="The batch size to use with the LLM."
        )
        self.register_output("metadata")
        self.register_output("topic")
        self.register_output("data")

    def run(self):
        combined_inputs = DataSource(
            "Combine inputs",
            {
                "metadata": list(self.inputs["metadata"]),
                "topic": list(self.inputs["topic"]),
            },
        )

        # Create prompts
        prompts_dataset = combined_inputs.map(
            lambda row: {
                "prompt": GENERATE_IMAGE_DESCRIPTION_PROMPT.format(
                    topic=row["topic"],
                    image_type=json.loads(row["metadata"])["image_type"],
                    persona=json.loads(row["metadata"])["persona"],
                )
            },
            lazy=False,
            name="Create Generate Descriptions Prompts",
        )

        # Generate Descriptions
        generated_descriptions = Prompt(
            name="Generate",
            inputs={
                "prompts": prompts_dataset.output["prompt"],
            },
            args={
                "llm": self.args["llm"],
                "batch_size": self.args["batch_size"],
                "post_process": extract_csv,
                "temperature": 1.0,
                "top_p": 1.0,
            },
            outputs={
                "generations": "data",
            },
        ).select_columns(["data"], name="Get Generated Descriptions")

        # Combine with generations with inputs
        combined = zipped(combined_inputs, generated_descriptions, name="Combine with inputs").save()

        # Return result
        return combined.output

    @property
    def version(self):
        return hash(GenerateImageDescription.CONFIG_HASH)
