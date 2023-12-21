#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging

import pandas as pd

import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info(f"Downloading the input artifacts: '{args.input_artifact}'")

    local_path = wandb.use_artifact(args.input_artifact).file()
    df = pd.read_csv(local_path)

    logger.info("Applying basic cleaning to the dataset ...")

    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()
    df["last_review"] = pd.to_datetime(df["last_review"])
    idx = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
    df = df[idx].copy()

    logger.info("Saving the dataset ...")

    df.to_csv("clean_sample.csv", index=False)
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This steps cleans the data")

    parser.add_argument(
        "--input_artifact", type=str, help="Dataset to be cleaned.", required=True
    )

    parser.add_argument(
        "--output_artifact", type=str, help="Clean dataset's name.", required=True
    )

    parser.add_argument("--output_type", type=str, help="Output dtyped", required=True)

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description about the output",
        required=True,
    )

    parser.add_argument(
        "--min_price", type=float, help="Minimum allowed price.", required=True
    )

    parser.add_argument(
        "--max_price", type=float, help="Maximum allowed price.", required=True
    )

    args = parser.parse_args()

    go(args)
