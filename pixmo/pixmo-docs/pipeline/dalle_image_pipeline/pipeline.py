import os
from datasets.fingerprint import Hasher
from datadreamer.steps import SuperStep
from .generate_image_topics import GenerateImageTopics
from .generate_image_description import GenerateImageDescription
from .generate_image import GenerateImage
from .generate_qa import GenerateImageQA


class DALLEImagePipeline(SuperStep):
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
        self.register_arg("figure_types", required=True, help="The image types to use.")
        self.register_arg("qa", required=True, help="Whether to generate Q&A.")

        self.register_output("metadata")
        self.register_output("topic")
        self.register_output("data")
        self.register_output("code")
        self.register_output("image")

        if os.environ["GENERATE_QA"] == "true": self.register_output("qa")

    def run(self):
        # Generate Topics
        generated_image_topics = GenerateImageTopics(
            "Generate Image Topics",
            args={
                "pipeline": self.__class__.__name__,
                "llm": self.args["llm"],
                "batch_size": self.args["batch_size"],
                "n": self.args["n"],
                "seed": self.args["seed"],
                "image_types": self.args["figure_types"],
            },
        )
        

        # Generate Descriptions
        generated_data = GenerateImageDescription(
            "Generate Descriptions",
            inputs={
                "metadata": generated_image_topics.output["metadata"],
                "topic": generated_image_topics.output["topic"],
            },
            args={
                "llm": self.args["llm"],
                "batch_size": self.args["batch_size"],
            },
        )

        # Generate Images
        generated_images = GenerateImage(
            "Generate Image",
            inputs={
                "metadata": generated_data.output["metadata"],
                "topic": generated_data.output["topic"],
                "data": generated_data.output["data"],
            },
            args={
                "llm": self.args["llm"],
                "batch_size": self.args["batch_size"],
            },
        )

        if not self.args["qa"]:
            return generated_images.output
        else:
            # Generate Q&A
            generated_qa = GenerateImageQA(
                "Generate Q&A",
                inputs={
                    "metadata": generated_images.output["metadata"],
                    "topic": generated_images.output["topic"],
                    "data": generated_images.output["data"],
                    "code": generated_images.output["code"],
                    "image": generated_images.output["image"],
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
                GenerateImageTopics.CONFIG_HASH,
                GenerateImageDescription.CONFIG_HASH,
                GenerateImage.CONFIG_HASH,
                GenerateImageQA.CONFIG_HASH,
            ]
        )
