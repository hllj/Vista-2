import os
from datasets.fingerprint import Hasher
from datadreamer.steps import SuperStep
from .generate_document_topics import GenerateDocumentTopics
from .generate_document_data import GenerateDocumentData
from .generate_document import GenerateDocument
from .generate_qa import GenerateDocumentQA


class DOCXDocumentPipeline(SuperStep):
    def setup(self):
        self.register_arg("llm", required=True, help="The LLM to use.")
        self.register_arg("code_llm", required=True, help="The LLM to use for code generation.")
        self.register_arg(
            "batch_size", required=True, help="The batch size to use with the LLM."
        )
        self.register_arg(
            "code_batch_size", required=True, help="The batch size to use with the code LLM."
        )
        self.register_arg("n", required=True, help="The number of topics to generate.")
        self.register_arg("seed", required=True, help="The seed to use for generation.")
        self.register_arg("figure_types", required=True, help="The figure types to use.")
        self.register_arg("qa", required=True, help="Whether to generate Q&A.")

        self.register_output("metadata")
        self.register_output("topic")
        self.register_output("data")
        self.register_output("code")
        self.register_output("image")

        if os.environ["GENERATE_QA"] == "true": self.register_output("qa")

    def run(self):
        # Generate Topics
        generated_document_topics = GenerateDocumentTopics(
            "Generate Document Topics",
            args={
                "pipeline": self.__class__.__name__,
                "llm": self.args["llm"],
                "batch_size": self.args["batch_size"],
                "n": self.args["n"],
                "seed": self.args["seed"],
                "figure_types": self.args["figure_types"],
            },
        )

        # Generate Data
        generated_data = GenerateDocumentData(
            "Generate Data",
            inputs={
                "metadata": generated_document_topics.output["metadata"],
                "topic": generated_document_topics.output["topic"],
            },
            args={
                "llm": self.args["llm"],
                "batch_size": self.args["batch_size"],
            },
        )

        # Generate Documents
        generated_documents = GenerateDocument(
            "Generate Code",
            inputs={
                "metadata": generated_data.output["metadata"],
                "topic": generated_data.output["topic"],
                "data": generated_data.output["data"],
            },
            args={
                "llm": self.args["code_llm"],
                "batch_size": self.args["code_batch_size"],
            },
        )

        if not self.args["qa"]:
            return generated_documents.output
        else:
            # Generate Q&A
            generated_qa = GenerateDocumentQA(
                "Generate Q&A",
                inputs={
                    "metadata": generated_documents.output["metadata"],
                    "topic": generated_documents.output["topic"],
                    "data": generated_documents.output["data"],
                    "code": generated_documents.output["code"],
                    "image": generated_documents.output["image"],
                },
                args={
                    "llm": self.args["llm"],
                    "batch_size": self.args["batch_size"],
                },
            )

            # Return result
            return generated_qa.output

    @property
    def version(self):
        return Hasher.hash(
            [
                GenerateDocumentTopics.CONFIG_HASH,
                GenerateDocumentData.CONFIG_HASH,
                GenerateDocument.CONFIG_HASH,
                GenerateDocumentQA.CONFIG_HASH,
            ]
        )
