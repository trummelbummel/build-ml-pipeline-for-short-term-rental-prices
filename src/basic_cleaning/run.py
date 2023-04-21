#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import os
import logging
import wandb
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logging.info('Downloading raw dataset from Wandb.')
    if not os.path.exists('../../components/get_data/data/' + args.input_artifact):
        logging.error('ERROR: Please include the download step in the pipeline.')

    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)
    logging.info('Downloading raw dataset SUCCESS.')

    logging.info('Preprocessing raw dataset.')
    df = df.drop_duplicates('name')
    assert len(df.name.unique()) == len(df)
    df.index = df.name
    df['name'].fillna(df['host_id'], inplace=True)
    df['lognorm_minimum_nights'] = np.log(df.minimum_nights)
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    df['last_review'] = pd.to_datetime(df['last_review'])
    df.availability_365.fillna(df.availability_365.median(), inplace=True)
    keepcols = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    df_new = df[keepcols]
    logging.info('Preprocessing SUCCESS.')
    logging.info(f'Keeping columns {keepcols}')

    logging.info('Uploading preprocessed dataset.')


    df_new.to_csv(args.output_artifact, index=False)
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)

    logging.info('Uploading SUCCESS.')
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")
    parser.add_argument(
        "--input_artifact",
        type=str,
        help='Path for input artifact download.',
        required=True,
        default="sample"
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help='Path to output artifact location.',
        required=True,
        default="clean_sample"
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help='Type of file that is output. e.g. csv',
        required=True,
        default='csv'
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help='Type of file that is input e.g. csv',
        required=True,
        default='Preprocessed Input Datafile.'
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help='Minimum price to be considered not an outlier.',
        required=True,
        default=10
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help='Maximum price to be considered not an outlier.',
        required=True,
        default=350,
    )

    args = parser.parse_args()

    go(args)
