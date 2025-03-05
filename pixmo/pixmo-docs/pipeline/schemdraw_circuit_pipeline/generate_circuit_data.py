import json

from datasets.fingerprint import Hasher
from datadreamer.steps import DataSource, SuperStep, Prompt, zipped

from ..prompts.misc_prompts import GENERATE_CIRCUIT_DATA_PROMPT
from ..utils.utils import extract_json, is_json_valid

class GenerateCircuitData(SuperStep):
    CONFIG_HASH = Hasher.hash([GENERATE_CIRCUIT_DATA_PROMPT])

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

        # add dummy data
        combined_inputs = combined_inputs.map(
            lambda row: {"data": "dummy"},
            lazy=False,
            name="Add dummy data",
        )

        return combined_inputs.output

    @property
    def version(self):
        return hash(GenerateCircuitData.CONFIG_HASH)
