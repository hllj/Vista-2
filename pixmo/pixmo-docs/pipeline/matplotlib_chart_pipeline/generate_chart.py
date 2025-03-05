import os
import tempfile
import json
import random
import warnings
import pandas as pd
from io import StringIO
import signal

from PIL import Image
from datasets.fingerprint import Hasher
from datadreamer.steps import DataSource, SuperStep, Prompt, zipped

from ..prompts.chart_prompts import GENERATE_CHART_CODE_MATPLOTLIB_PROMPT
from ..utils.utils import extract_code, process_image

NUM_RENDER_WORKERS = 1

class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException()


class GenerateChart(SuperStep):
    CONFIG_HASH = Hasher.hash([GENERATE_CHART_CODE_MATPLOTLIB_PROMPT])

    def setup(self):
        self.register_input(
            "metadata", required=True, help="The metadata used to generate the topics."
        )
        self.register_input("topic", required=True, help="The topics.")
        self.register_input("data", required=True, help="The data.")
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

        matplotlib_styles = ['classic', 'dark_background', 'Solarize_Light2', 'bmh', 'fast', '_mpl-gallery', '_mpl-gallery-nogrid',
                             'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', '_classic_test_patch',
                             'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-muted', 'tableau-colorblind10',
                             'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel',
                             'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid']

        # Create prompts
        prompts_dataset = combined_inputs.map(
            lambda row: {
                "prompt": GENERATE_CHART_CODE_MATPLOTLIB_PROMPT.format(
                    topic=row["topic"],
                    figure_type=json.loads(row["metadata"])["figure_type"],
                    data=row["data"],
                    persona=json.loads(row["metadata"])["persona"],
                    style_name=random.choice(['default', random.choice(matplotlib_styles)]),
                )
            },
            lazy=False,
            name="Create Generate Code Prompts",
        )

        # Generate Code
        generated_code = Prompt(
            name="Generate",
            inputs={
                "prompts": prompts_dataset.output["prompt"],
            },
            args={
                "llm": self.args["llm"],
                "batch_size": self.args["batch_size"],
                "post_process": extract_code,
                "temperature": 1.0,
                "top_p": 1.0,
            },
            outputs={
                "generations": "code",
            },
        ).select_columns(["code"], name="Get Generated Code")

        # Combine with generations with inputs
        combined = zipped(
            combined_inputs, generated_code, name="Combine with inputs"
        ).save(name="Save combine with inputs")

        # Generate Images
        def execute_code_and_generate_image(row, timeout=20):
            original_dir = os.getcwd()
            os.chdir(tempfile.mkdtemp())
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)  # set the timeout
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    exec(row["code"], globals())
                    image = generate_plot(pd.read_csv(StringIO(row["data"]))) # noqa: F821
                    globals().pop("generate_plot", None)
                    
                    if not isinstance(image, Image.Image):
                        raise TypeError()
                    
                    row["image"] = process_image(image)
            except TimeoutException:
                print(f"Error: Code execution exceeded {timeout} seconds.")
                row["image"] = None
            except Exception as e:
                print(f"Error: {e}")
                row["image"] = None
            finally:
                signal.alarm(0)  # disable the alarm
                os.chdir(original_dir)
            
            return row

        code_and_images = combined.map(
            execute_code_and_generate_image,
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
        return hash(GenerateChart.CONFIG_HASH)
