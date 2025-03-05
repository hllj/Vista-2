import os
from argparse import ArgumentParser

from pipeline import run_datadreamer_session
from pipeline.utils.gpt4o_support import datadreamer_gpt4o_support


def main(args):
    with datadreamer_gpt4o_support():
        run_datadreamer_session(args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-o",
        "--openai_api_key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="The OpenAI secret key.",
    )
    parser.add_argument(
        "-a",
        "--anthropic_api_key",
        type=str,
        default=os.getenv("ANTHROPIC_API_KEY"),
        help="The Anthropic secret key.",
    )
    parser.add_argument(
        "-l",
        "--llm",
        type=str,
        default="gpt-4o",
        help="LLM to use (gpt-4 or claude-sonnet).",
    )
    parser.add_argument(
        "-c",
        "--code_llm",
        type=str,
        default="claude-sonnet",
        help="LLM to use (gpt-4 or claude-sonnet) for code generation.",
    )
    parser.add_argument(
        "-p",
        "--pipelines",
        type=str,
        default="MatplotlibChartPipeline",
        help="Which pipelines to run comma-separated.",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=str,
        default="1",
        help="The number of visualizations to generate per pipeline. (either a single number or a comma-separated list of numbers)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="The seed to use for generation.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=24,
        help="The number of requests to make to the LLM in parallel.",
    )
    parser.add_argument(
        "-cb",
        "--code_batch_size",
        type=int,
        default=24,
        help="The number of requests to make to the coding LLM in parallel.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Force regenerate.",
    )
    parser.add_argument(
        "-m",
        "--name",
        type=str,
        default="scifi",
        help="The name of the dataset to push to huggingface.",
    )
    parser.add_argument(
        "-t",
        "--types",
        type=str,
        default="bar chart",
        help="The types of visualizations to generate.",
    )
    parser.add_argument(
        "-q",
        "--qa",
        type=bool,
        default=True,
        help="whether to generate QA for the visualizations.",
    )

    args = parser.parse_args()

    print("LLM:", args.llm)
    print("Code LLM:", args.code_llm)
    print("Pipelines:", args.pipelines)
    print("Num:", args.num)
    print("Seed:", args.seed)
    print("Batch Size:", args.batch_size)
    print("Code Batch Size:", args.code_batch_size)
    print("Name:", args.name)
    print("Types:", args.types)

    main(args)
