import os
import json
import random
from ..utils.utils import PERSONAS
from ..prompts.misc_prompts import GENERATE_CIRCUIT_TOPICS_PROMPT, NUM_TOPICS

from ordered_set import OrderedSet
from datasets.fingerprint import Hasher
from datadreamer.steps import DataSource, SuperStep, Prompt, zipped

class GenerateCircuitTopics(SuperStep):
    CONFIG_HASH = Hasher.hash(
        [PERSONAS, GENERATE_CIRCUIT_TOPICS_PROMPT, NUM_TOPICS]
    )

    def setup(self):
        self.register_arg("pipeline", required=True, help="The current pipeline.")
        self.register_arg("llm", required=True, help="The LLM to use.")
        self.register_arg(
            "batch_size", required=True, help="The batch size to use with the LLM."
        )
        self.register_arg("n", required=True, help="The number of personas to use.")
        self.register_arg("seed", required=True, help="The seed to use for generation.")
        self.register_arg("figure_types", required=True, help="The figure types to use.")
        self.register_output("metadata")
        self.register_output("topic")


    def run(self):
        USED_PERSONAS = []
        if os.path.exists("selected_personas.txt"):
            with open("selected_personas.txt", "r") as f:
                for line in f: USED_PERSONAS.append(line.strip())

        UNUSED_PERSONAS = list(OrderedSet(PERSONAS) - OrderedSet(USED_PERSONAS))

        if len(UNUSED_PERSONAS) < self.args["n"]:
            USED_PERSONAS = PERSONAS
            with open("selected_personas.txt", "w") as f: f.write("")
        
        random.seed(self.args["seed"])
        selected_personas = random.sample(UNUSED_PERSONAS, self.args["n"])

        with open("selected_personas.txt", "a") as f:
            for persona in selected_personas: f.write(persona + "\n")

        # Get triplets
        triplets = []
        for persona in selected_personas:
            triplets.append(
                {
                    "metadata": json.dumps(
                        {
                            "_pipeline": self.args["pipeline"],
                            "persona": persona,
                            "figure_type": random.choice(self.args["figure_types"]),
                        }
                    )
                }
            )

        triplets_dataset = DataSource("Load Generate Topics Triplets", triplets)

        # Create prompts
        prompts_dataset = triplets_dataset.map(
            lambda row: {
                "prompt": GENERATE_CIRCUIT_TOPICS_PROMPT.format(
                    num_topics=NUM_TOPICS,
                    persona=json.loads(row["metadata"])["persona"],
                    figure_type=json.loads(row["metadata"])["figure_type"],
                )
            },
            lazy=False,
            name="Create Generate Topics Prompts",
        )

        # Generate Topics
        generated_topics = Prompt(
            name="Generate",
            inputs={
                "prompts": prompts_dataset.output["prompt"],
            },
            args={
                "llm": self.args["llm"],
                "batch_size": self.args["batch_size"],
                "temperature": 1.0,
                "top_p": 1.0,
            },
            outputs={
                "generations": "topic",
            },
        ).select_columns(["topic"], name="Get Generated Topics")

        # Combine the metadata with the topic
        combined = zipped(triplets_dataset, generated_topics, name="Keep Metadata")

        # Extract each topic into its own row
        combined_and_extracted = combined.map(
            lambda row: {
                "topic": [t.strip() for t in row["topic"][0].strip().split("|")],
                "metadata": row["metadata"] * (row["topic"][0].count("|") + 1),
            },
            batched=True,
            batch_size=1,
            lazy=False,
            name="Extract Topics",
        )

        # Warn if < n topics
        if combined_and_extracted.output.num_rows < self.args["n"]:
            self.logger.info(
                f"Warning: Only generated {combined_and_extracted.output.num_rows} vs. {self.args['n']} topics as invalid ones may have been removed."
            )

        # Shuffle the data
        shuffled = combined_and_extracted.shuffle(seed=self.args["seed"])

        # Keep only n if more
        subset = shuffled.take(self.args["n"], "Keep only desired number of topics")

        # Return result
        return subset.output

    @property
    def version(self):
        return hash(GenerateCircuitTopics.CONFIG_HASH)
