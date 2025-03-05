import json

from datasets.fingerprint import Hasher
from datadreamer.steps import DataSource, SuperStep, Prompt, zipped

from ..prompts.chart_prompts import GENERATE_CHART_DATA_JSON_PROMPT
from ..utils.utils import extract_json, is_json_valid

class GenerateChartData(SuperStep):
    CONFIG_HASH = Hasher.hash([GENERATE_CHART_DATA_JSON_PROMPT])

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
                "prompt": GENERATE_CHART_DATA_JSON_PROMPT.format(
                    topic=row["topic"],
                    figure_type=json.loads(row["metadata"])["figure_type"],
                    persona=json.loads(row["metadata"])["persona"],
                )
            },
            lazy=False,
            name="Create Generate Data Prompts",
        )

        # Generate Data
        generated_data = Prompt(
            name="Generate",
            inputs={
                "prompts": prompts_dataset.output["prompt"],
            },
            args={
                "llm": self.args["llm"],
                "batch_size": self.args["batch_size"],
                "post_process": extract_json,
                "temperature": 1.0,
                "top_p": 1.0,
            },
            outputs={
                "generations": "data",
            },
        ).select_columns(["data"], name="Get Generated Data")

        # Combine with generations with inputs
        combined = zipped(combined_inputs, generated_data, name="Combine with inputs")

        # Remove any invalid JSON objects
        filtered = combined.filter(
            lambda row: is_json_valid(row["data"]),
            lazy=False,
            name="Remove invalid JSON objects",
        )
        if filtered.output.num_rows < combined.output.num_rows:
            self.logger.info(
                f"Warning: Could only generate valid data for {filtered.output.num_rows} out of {combined.output.num_rows} total rows."
            )

        # Return result
        return filtered.output

    @property
    def version(self):
        return hash(GenerateChartData.CONFIG_HASH)
