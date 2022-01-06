import abc
import json
import os
import pickle
import shutil
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Dict, Type, cast, Any
from glob import glob

import luigi
import pandas as pd
import torch
import pytorch_lightning as pl
import wandb
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import (
    get_lags_for_frequency,
    TimeFeature,
    DayOfWeek,
    DayOfMonth,
    DayOfYear,
    WeekOfYear,
)
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.modules.distribution_output import (
    DistributionOutput,
    GammaOutput,
    BetaOutput,
)
from pts.modules import (
    NegativeBinomialOutput,
    PoissonOutput,
    ZeroInflatedPoissonOutput,
    ZeroInflatedNegativeBinomialOutput,
    NormalOutput,
    StudentTOutput,
)
from sklearn.preprocessing import OrdinalEncoder

from datah_challenge.dataset import (
    JsonGzDataset,
    SameSizeTransformedDataset,
    FilterTimeSeriesTransformation,
    TruncateTargetTransformation,
    SplitFeaturesIntoFields,
)
from datah_challenge.gluonts.custom import CustomDeepAREstimator
from datah_challenge.gluonts.distribution import BimodalBetaOutput, BiStudentTMixtureOutput
from datah_challenge.gluonts.model.causal_deepar import CausalDeepAREstimator
from datah_challenge.gluonts.model.tft.estimator import TemporalFusionTransformerEstimator
from datah_challenge.gluonts.time_feature import (
    DayOfWeekSin,
    DayOfWeekCos,
    DayOfMonthSin,
    DayOfMonthCos,
    DayOfYearSin,
    DayOfYearCos,
    WeekOfYearSin,
    WeekOfYearCos,
)
from datah_challenge.path import get_assets_path
from datah_challenge.task.data_preparation import FilterGluonTimeSeriesDatasets, CountCardinalities
from datah_challenge.utils import save_params, calculate_split
from datah_challenge.wandb import WandbWithBestMetricLogger

_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


_DISTRIBUTIONS: Dict[str, Type[DistributionOutput]] = {
    "negative_binomial": NegativeBinomialOutput,
    "zero_inflated_negative_binomial": ZeroInflatedNegativeBinomialOutput,
    "poisson": PoissonOutput,
    "zero_inflated_poisson": ZeroInflatedPoissonOutput,
    "normal": NormalOutput,
    "student_t": StudentTOutput,
    "beta": BetaOutput,
    "bimodal_beta": BimodalBetaOutput,
    "gamma": GammaOutput,
    "bi_student_t_mixture": BiStudentTMixtureOutput,
}

_TIME_FEATURES: Dict[str, Type[TimeFeature]] = {
    "day_of_week": DayOfWeek,
    "day_of_month": DayOfMonth,
    "day_of_year": DayOfYear,
    "week_of_year": WeekOfYear,
    "day_of_week_sin": DayOfWeekSin,
    "day_of_week_cos": DayOfWeekCos,
    "day_of_month_sin": DayOfMonthSin,
    "day_of_month_cos": DayOfMonthCos,
    "day_of_year_sin": DayOfYearSin,
    "day_of_year_cos": DayOfYearCos,
    "week_of_year_sin": WeekOfYearSin,
    "week_of_year_cos": WeekOfYearCos,
}


class BaseTraining(luigi.Task, metaclass=abc.ABCMeta):
    categorical_variables: List[str] = luigi.ListParameter(
        default=[
            "S101",
            "S102",
            "S103",
            "S100",
            "I101",
            "I102",
            "I103",
            "I100",
            "C100",
            "C101",
        ]
    )
    id_variables: List[str] = luigi.ListParameter(
        default=[
            "S100",
            "I100",
            "C100",
            "C101",
        ]
    )

    split_index: int = luigi.IntParameter(default=0)
    split_steps: int = luigi.IntParameter(default=8)
    test_steps: int = luigi.IntParameter(default=8)

    context_length: int = luigi.IntParameter(default=4)
    time_features: List[str] = luigi.ListParameter(
        default=["day_of_week", "day_of_month", "day_of_year"]
    )

    precision: int = luigi.IntParameter(default=32)

    batch_size: int = luigi.IntParameter(default=32)
    accumulate_grad_batches: int = luigi.IntParameter(default=1)
    max_epochs: int = luigi.IntParameter(default=100)
    num_batches_per_epoch: int = luigi.IntParameter(default=-1)
    lr: float = luigi.FloatParameter(default=1e-3)
    weight_decay: float = luigi.FloatParameter(default=1e-8)
    gradient_clip_val: float = luigi.FloatParameter(default=10.0)
    early_stopping_patience: int = luigi.IntParameter(default=5)

    num_workers: int = luigi.IntParameter(default=0)
    num_prefetch: int = luigi.IntParameter(default=2)

    seed: int = luigi.IntParameter(default=42)
    use_gpu: bool = luigi.BoolParameter(default=False)

    cache_dataset: bool = luigi.BoolParameter(default=False)
    preload_dataset: bool = luigi.BoolParameter(default=False)
    check_data: bool = luigi.BoolParameter(default=False)

    def requires(self):
        return FilterGluonTimeSeriesDatasets(
            categorical_variables=self.categorical_variables,
            id_variables=self.id_variables,
            split_index=self.split_index,
            split_steps=self.split_steps,
            test_steps=self.test_steps,
        ), CountCardinalities(categorical_variables=self.categorical_variables)

    def output(self):
        return luigi.LocalTarget(
            os.path.join("output", self.__class__.__name__, self.task_id)
        )

    @cached_property
    def dataset_path(self) -> str:
        return self.input()[0].path

    @cached_property
    def dataset(self) -> Dataset:
        paths = glob(os.path.join(self.dataset_path, "*.json.gz"))
        return JsonGzDataset(paths, freq="W")

    @cached_property
    def train_dataset(self) -> Dataset:
        if self.split_index >= 0:
            split = calculate_split(
                self.split_index, self.split_steps, self.test_steps
            )
            return SameSizeTransformedDataset(
                self.dataset,
                transformation=FilterTimeSeriesTransformation(start=0, end=split),
            )
        else:
            return self.dataset

    @cached_property
    def val_dataset(self) -> Optional[Dataset]:
        if self.split_index >= 0:
            return self.dataset
        else:
            return None

    @cached_property
    def cardinalities(self) -> Dict[str, int]:
        with open(self.input()[1].path, "r") as f:
            return json.load(f)

    def _serialize(self, predictor: PyTorchPredictor):
        print("Serializing predictor...")
        predictor_path = os.path.join(self.output().path, "predictor")
        os.mkdir(predictor_path)
        predictor.serialize(Path(predictor_path))
        print("Serialized predictor...")

    def get_trained_predictor(self, device: torch.device) -> PyTorchPredictor:
        predictor_path = os.path.join(self.output().path, "predictor")
        predictor = PyTorchPredictor.deserialize(Path(predictor_path), device=device,)
        predictor.prediction_net.to(device)
        return predictor

    @property
    def num_feat_dynamic_real(self) -> int:
        return 0

    @property
    @abc.abstractmethod
    def wandb_project(self) -> str:
        pass

    @abc.abstractmethod
    def create_estimator(
        self,
        wandb_logger: WandbWithBestMetricLogger,
        early_stopping: pl.callbacks.EarlyStopping,
    ) -> PyTorchLightningEstimator:
        pass

    def run(self):
        os.makedirs(self.output().path, exist_ok=True)

        save_params(self.output().path, self.param_kwargs)

        pl.seed_everything(self.seed, workers=True)

        monitor = "val_loss"

        wandb_logger = WandbWithBestMetricLogger(
            name=self.task_id,
            save_dir=self.output().path,
            project=self.wandb_project,
            log_model=False,
            monitor=monitor,
            mode="min",
        )
        wandb_logger.log_hyperparams(self.param_kwargs)

        early_stopping = pl.callbacks.EarlyStopping(
            monitor=monitor,
            mode="min",
            patience=self.early_stopping_patience,
            verbose=True,
        ) if self.split_index >= 0 else None
        estimator = self.create_estimator(wandb_logger, early_stopping)

        train_output = estimator.train_model(
            self.train_dataset,
            validation_data=self.val_dataset,
            num_workers=self.num_workers,
            cache_data=self.cache_dataset,
        )

        self._serialize(train_output.predictor)

        if self.split_index >= 0:
            train_output.predictor.batch_size = 512
            val_forecast_it, val_ts_it = make_evaluation_predictions(
                dataset=self.val_dataset, predictor=train_output.predictor, num_samples=100
            )
            evaluator = Evaluator(quantiles=[], calculate_owa=False, num_workers=min(8, os.cpu_count()))
            agg_metrics, item_metrics = evaluator(val_ts_it, val_forecast_it, num_series=len(self.val_dataset))

            with open(os.path.join(self.output().path, "metrics.json"), "w") as f:
                json.dump(agg_metrics, f, indent=4)

            wandb_logger.experiment.log(agg_metrics)

        predictor_artifact = wandb.Artifact(
            name=f"artifact-{wandb_logger.experiment.id}", type="model"
        )
        predictor_artifact.add_dir(os.path.join(self.output().path, "predictor"))
        wandb_logger.experiment.log_artifact(predictor_artifact)


class DeepARTraining(BaseTraining, metaclass=abc.ABCMeta):
    distribution: str = luigi.ChoiceParameter(
        choices=_DISTRIBUTIONS.keys(), default="negative_binomial"
    )

    lags_seq_ub: int = luigi.IntParameter(default=52)

    num_layers: int = luigi.IntParameter(default=2)
    hidden_size: int = luigi.IntParameter(default=40)
    dropout_rate: float = luigi.FloatParameter(default=0.1)
    embedding_dimension: List[int] = luigi.ListParameter(default=None)
    num_parallel_samples: int = luigi.IntParameter(default=100)

    @cached_property
    def cardinality(self) -> List[int]:
        return [
            self.cardinalities[variable]
            for variable in self.categorical_variables
        ]

    @property
    def wandb_project(self) -> str:
        return "datah-challenge"

    def get_estimator_params(
        self,
        wandb_logger: WandbWithBestMetricLogger,
        early_stopping: Optional[pl.callbacks.EarlyStopping],
    ) -> Dict[str, Any]:
        return dict(
            freq="W",
            prediction_length=self.test_steps,
            context_length=self.context_length,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            dropout_rate=self.dropout_rate,
            num_feat_dynamic_real=self.num_feat_dynamic_real,
            num_feat_static_cat=len(self.categorical_variables),
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            distr_output=_DISTRIBUTIONS[self.distribution](),
            scaling=True,
            lags_seq=get_lags_for_frequency("W", lag_ub=self.lags_seq_ub),
            time_features=[
                _TIME_FEATURES[time_feature]() for time_feature in self.time_features
            ],
            num_parallel_samples=self.num_parallel_samples,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            num_batches_per_epoch=self.num_batches_per_epoch
            if self.num_batches_per_epoch > 0
            else len(self.train_dataset) // self.batch_size,
            trainer_kwargs=dict(
                max_epochs=self.max_epochs,
                accumulate_grad_batches=self.accumulate_grad_batches,
                gradient_clip_val=self.gradient_clip_val,
                logger=wandb_logger,
                callbacks=[early_stopping] if early_stopping is not None else [],
                default_root_dir=self.output().path,
                gpus=torch.cuda.device_count() if self.use_gpu else 0,
                precision=self.precision,
                num_sanity_val_steps=0,
                deterministic=True,
            ),
        )

    def create_estimator(
        self,
        wandb_logger: WandbWithBestMetricLogger,
        early_stopping: Optional[pl.callbacks.EarlyStopping],
    ) -> DeepAREstimator:
        return CustomDeepAREstimator(
            **self.get_estimator_params(wandb_logger, early_stopping)
        )


class TemporalFusionTransformerTraining(BaseTraining):
    dropout_rate: float = luigi.FloatParameter(default=0.1)
    embed_dim: int = luigi.IntParameter(default=32)
    num_heads: int = luigi.IntParameter(default=4)
    num_outputs: int = luigi.IntParameter(default=9)
    variable_dim: Optional[int] = luigi.IntParameter(default=None)

    @property
    def wandb_project(self) -> str:
        return "datah-challenge"

    @property
    def real_variables(self) -> List[str]:
        return []

    @cached_property
    def train_dataset(self) -> Dataset:
        return SameSizeTransformedDataset(
            super().train_dataset,
            transformation=SplitFeaturesIntoFields(
                self.categorical_variables,
                self.real_variables,
            ),
        )

    @cached_property
    def val_dataset(self) -> Dataset:
        return SameSizeTransformedDataset(
            super().val_dataset,
            transformation=SplitFeaturesIntoFields(
                self.categorical_variables,
                self.real_variables,
            ),
        )

    def create_estimator(
        self,
        wandb_logger: WandbWithBestMetricLogger,
        early_stopping: pl.callbacks.EarlyStopping,
    ) -> TemporalFusionTransformerEstimator:
        return TemporalFusionTransformerEstimator(
            freq="W",
            prediction_length=self.test_steps,
            context_length=self.context_length,
            dropout_rate=self.dropout_rate,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_outputs=self.num_outputs,
            variable_dim=self.variable_dim,
            time_features=[
                _TIME_FEATURES[time_feature]() for time_feature in self.time_features
            ],
            static_cardinalities={
                variable: self.cardinalities[variable]
                for variable in self.categorical_variables
            },
            dynamic_feature_dims={
                variable: 1
                for variable in self.real_variables
            },
            past_dynamic_features=self.real_variables,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            num_batches_per_epoch=self.num_batches_per_epoch
            if self.num_batches_per_epoch > 0
            else len(self.train_dataset) // self.batch_size,
            trainer_kwargs=dict(
                max_epochs=self.max_epochs,
                accumulate_grad_batches=self.accumulate_grad_batches,
                gradient_clip_val=self.gradient_clip_val,
                logger=wandb_logger,
                callbacks=[early_stopping],
                default_root_dir=self.output().path,
                gpus=torch.cuda.device_count() if self.use_gpu else 0,
                precision=self.precision,
                num_sanity_val_steps=0,
                deterministic=True,
            ),
        )
