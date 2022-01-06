import os
from glob import glob
from functools import cached_property
import json
import datetime

import luigi
import pandas as pd
from tqdm import tqdm
from gluonts.model.forecast import Forecast
import torch
import pytorch_lightning as pl

from datah_challenge.task.training import BaseTraining, DeepARTraining, TemporalFusionTransformerTraining
from datah_challenge.dataset import JsonGzDataset, SplitFeaturesIntoFields, SameSizeTransformedDataset
from datah_challenge.task.data_preparation import PrepareGluonTimeSeriesDatasets


class GenerateSubmission(luigi.Task):
    task_path: str = luigi.Parameter()

    num_samples: int = luigi.IntParameter(default=100)
    seed: int = luigi.IntParameter(default=42)

    @cached_property
    def training(self) -> BaseTraining:
        with open(os.path.join(self.task_path, "params.json"), "r") as params_file:
            params = json.load(params_file)
        training_class = {
            DeepARTraining.__name__: DeepARTraining,
            TemporalFusionTransformerTraining.__name__: TemporalFusionTransformerTraining,
        }[os.path.split(os.path.split(self.task_path)[0])[1]]
        return training_class(**params)

    def requires(self):
        return PrepareGluonTimeSeriesDatasets(
            categorical_variables=self.training.categorical_variables,
            id_variables=self.training.id_variables,
        )

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                self.task_path,
                f"submission_num-samples={self.num_samples}_seed={self.seed}.csv",
            )
        )

    def run(self):
        pl.seed_everything(self.seed, workers=True)

        paths = glob(os.path.join(self.input().path, "*.json.gz"))

        dataset = JsonGzDataset(paths, freq="W")

        if isinstance(self.training, TemporalFusionTransformerTraining):
            dataset = SameSizeTransformedDataset(
                dataset,
                transformation=SplitFeaturesIntoFields(
                    self.training.categorical_variables,
                    self.training.real_variables,
                ),
            )

        predictor = self.training.get_trained_predictor(torch.device("cuda"))
        predictor.batch_size = 512

        rows = []
        for forecast in tqdm(predictor.predict(dataset, num_samples=self.num_samples), total=len(dataset)): # type: Forecast
            delta = datetime.timedelta(weeks=self.training.test_steps - 1)

            dates = pd.date_range(
                forecast.start_date, forecast.start_date+delta, freq="W"
            )

            for date, pred in zip(dates, forecast.mean):
                rows.append({
                    "ID": "%s_%s" % (date.strftime('%Y-%m-%d'), forecast.item_id.replace("_partition=", "")),
                    "QTT": float(pred)
                })

        df = pd.DataFrame(data=rows)

        sample_df = pd.read_csv("assets/submission_sample.csv")
        df = df[df["ID"].isin(sample_df["ID"])]

        remaining_df = sample_df[~(sample_df["ID"].isin(df["ID"]))]
        remaining_df["QTT"] = 1

        df = pd.concat([df, remaining_df])

        df.sort_values(by="ID").to_csv(self.output().path, index=False)





