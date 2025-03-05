import json

from datasets.fingerprint import Hasher
from datadreamer.steps import DataSource, SuperStep, Prompt, zipped

from ..utils.utils import is_json_valid
from ..prompts.misc_prompts import GENERATE_GRAPHIC_QA_PROMPT

class GenerateGraphicQA(SuperStep):
    CONFIG_HASH = Hasher.hash([GENERATE_GRAPHIC_QA_PROMPT])

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
        self.register_output("qa")

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

        # Create Q&A prompts
        qa_prompts_dataset = combined_inputs.map(
            lambda row: {
                "prompt": GENERATE_GRAPHIC_QA_PROMPT.format(
                    topic=row["topic"], data=row["data"], code=row["code"], persona=json.loads(row["metadata"])["persona"], figure_type=json.loads(row["metadata"])["figure_type"]
                )
            },
            remove_columns=["image"],
            lazy=False,
            name="Create Generate Q&A Prompts",
        )

        # Generate Q&A
        generated_qa = Prompt(
            name="Generate Q&A",
            inputs={
                "prompts": qa_prompts_dataset.output["prompt"],
            },
            args={
                "llm": self.args["llm"],
                "batch_size": self.args["batch_size"],
                "temperature": 1.0,
                "top_p": 1.0,
            },
            outputs={
                "generations": "qa",
            },
        ).select_columns(["qa"], name="Get Generated Q&A")

        # Combine with generations with inputs
        combined = zipped(
            combined_inputs,
            generated_qa,
            name="Combine with inputs",
        )

        def process_qa(row):
            response = row["qa"]
            lines = response.split("\n\n")
            qa = []
            for line in lines:
                qa_obj = {}
                parts = line.split("|")
                if len(parts) == 3:
                    qa_obj["question"] = parts[0].strip()
                    qa_obj["answer"] = parts[1].strip()
                    qa_obj["explanation"] = parts[2].strip()
                    qa.append(qa_obj)
            row["qa"] = json.dumps(qa)
            return row
        
        combined_processed = combined.map(
            process_qa,
            lazy=False,
            name="Process Q&A",
        )

        if combined_processed.output.num_rows < combined.output.num_rows:
            self.logger.info(
                f"Warning: Could only generate valid Q&A for {combined_processed.output.num_rows} out of {combined.output.num_rows} total rows."
            )

        # Return result
        return combined_processed.output

    @property
    def version(self):
        return hash(GenerateGraphicQA.CONFIG_HASH)
