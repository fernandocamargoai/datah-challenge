import abc
import functools
import json
import os
import pickle
import shutil
from glob import glob
from multiprocessing.pool import Pool
from typing import List, Tuple, Dict
import warnings
import datetime

import luigi
import numpy as np
import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, StringIndexerModel
from pyspark.sql import SparkSession, DataFrame, Window, WindowSpec
from pyspark.sql.functions import (
    col,
    lit,
    sum as sum_,
    min as min_,
    max as max_,
    udf,
    datediff,
    lead,
    explode,
    count,
    row_number,
    concat,
    concat_ws,
    collect_set,
    date_add,
    next_day,
    last_day,
    when,
    dayofmonth,
    trunc,
    to_date,
)
from pyspark.sql.types import StructType, StructField, DateType, ArrayType, IntegerType
import psutil
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import Dataset, DataEntry, ListDataset
from luigi.contrib.spark import PySparkTask
from sklearn.preprocessing import OrdinalEncoder, scale
from tqdm import tqdm

from datah_challenge.path import get_assets_path, get_extra_data_path
from datah_challenge.utils import save_json_gzip, save_params, calculate_split, load_json_gzip, symlink_to_dir


class BasePySparkTask(PySparkTask, metaclass=abc.ABCMeta):
    def setup(self, conf: SparkConf):
        conf.set("spark.local.dir", os.path.join("output", "spark"))
        conf.set("spark.sql.warehouse.dir", os.path.join("output", "spark-warehouse"))
        conf.set("spark.driver.maxResultSize", f"{int(self._get_available_memory())}g")
        conf.set("spark.executor.memory", f"{int(self._get_available_memory())}g")
        conf.set(
            "spark.executor.memoryOverhead",
            f"{int(self._get_available_memory() * 0.25)}g",
        )

    @property
    def driver_memory(self):
        return f"{int(self._get_available_memory())}g"

    def _get_available_memory(self) -> str:
        return psutil.virtual_memory().available / (1024 * 1024 * 1024) * 0.85

    def main(self, sc: SparkContext, *args):
        self.spark = SparkSession(sc)
        self.run_with_spark()

    @abc.abstractmethod
    def run_with_spark(self):
        pass


def _get_next_dates(start_date: datetime.date, diff: int) -> List[datetime.date]:
    return [
        start_date + datetime.timedelta(days=days)
        for days in range(1, diff)
        if (days % 7) == 0
    ]


def _get_fill_dates_df(
    df: DataFrame,
    id_variables: List[str],
) -> DataFrame:
    get_next_dates_udf = udf(_get_next_dates, ArrayType(DateType()))

    window = Window.partitionBy(*id_variables).orderBy("DATE")

    return (
        df.withColumn("_diff", datediff(lead("DATE", 1).over(window), "DATE"))
        .filter(col("_diff") > 1)
        .withColumn("_next_dates", get_next_dates_udf("DATE", "_diff"))
        .withColumn("QTT", lit("0"))
        .withColumn("DATE", explode("_next_dates"))
        .drop("_diff", "_next_dates")
    )


def _save_dataset_item(
    filepath: str,
    categorical_variables: List[str],
    output_dir: str,
):
    df = pd.read_parquet(filepath)

    first_row = df.iloc[0]

    ts_id = os.path.split(filepath)[1].replace(".parquet", "")

    try:
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.set_index("DATE").sort_index()

        categorical_values = [
            int(first_row[variable]) for variable in categorical_variables
        ]

        data_entry = {
            FieldName.ITEM_ID: ts_id,
            FieldName.TARGET: df["QTT"].values.tolist(),
            FieldName.START: str(df.index[0]),
            FieldName.FEAT_STATIC_CAT: categorical_values,
        }

        save_json_gzip(
            data_entry,
            os.path.join(output_dir, "{}.json.gz".format(ts_id)),
        )
    except Exception:
        print(f"Error when processing TS: {ts_id}")
        raise


class PreProcessDataset(BasePySparkTask):
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

    def input(self) -> Tuple[luigi.LocalTarget, luigi.LocalTarget, luigi.LocalTarget]:
        assets_path = get_assets_path()
        return (
            luigi.LocalTarget(os.path.join(assets_path, "sales.csv")),
            luigi.LocalTarget(os.path.join(assets_path, "meta-store.csv")),
            luigi.LocalTarget(os.path.join(assets_path, "meta-item.csv")),
        )

    def output(self):
        dir_path = os.path.join("output", self.__class__.__name__, self.task_id)
        return (
            luigi.LocalTarget(dir_path),
            luigi.LocalTarget(os.path.join(dir_path, "data.parquet")),
        )

    def _add_max_date_for_all_groups(
        self, df: DataFrame, max_date_created: datetime.date
    ) -> DataFrame:
        window = Window.partitionBy(*self.id_variables).orderBy(col("DATE").desc())
        last_row_per_group_df: DataFrame = (
            df.withColumn("_row_number", row_number().over(window))
            .where(col("_row_number") == 1)
            .drop("_row_number")
        )

        filled_last_date_groups_df = (
            last_row_per_group_df[
                last_row_per_group_df["DATE"] != max_date_created
            ]
            .withColumn("DATE", lit(max_date_created))
            .withColumn("QTT", lit(0))
        )
        df = df.union(filled_last_date_groups_df)
        return df

    def _fill_dates(self, df: DataFrame) -> DataFrame:
        fill_df = _get_fill_dates_df(df, self.id_variables)
        df = df.union(fill_df)
        return df

    def _create_partition_column_by_id_variables(self, df: DataFrame) -> DataFrame:
        return df.withColumn(
            "_partition", concat(concat_ws("_", *self.id_variables), lit(".parquet"))
        )

    def run_with_spark(self):
        os.makedirs(self.output()[0].path, exist_ok=True)

        save_params(self.output()[0].path, self.param_kwargs)

        df = self.spark.read.csv(self.input()[0].path, inferSchema=True, header=True)
        df = df.withColumn("DATE", to_date(df["DATE"]))

        max_date_created = df.select(
            max_("DATE")
        ).first()[0]

        df = self._add_max_date_for_all_groups(df, max_date_created)
        df = self._fill_dates(df)
        df = self._create_partition_column_by_id_variables(df)

        store_df = self.spark.read.csv(
            self.input()[1].path, inferSchema=True, header=True
        )
        item_df = self.spark.read.csv(
            self.input()[2].path, inferSchema=True, header=True
        )
        df = df.join(store_df, how="left", on="S100")
        df = df.join(item_df, how="left", on="I100")
        for variable in self.categorical_variables:
            # Increment categorical variables (that are already indexed) by one to use 0 as unknown
            df.withColumn(variable, col(variable) + 1)

        df.repartition("_partition").write.partitionBy("_partition").parquet(
            self.output()[1].path
        )


class CountCardinalities(luigi.Task):
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

    def input(self) -> Tuple[luigi.LocalTarget, luigi.LocalTarget, luigi.LocalTarget]:
        assets_path = get_assets_path()
        return (
            luigi.LocalTarget(os.path.join(assets_path, "sales.csv")),
            luigi.LocalTarget(os.path.join(assets_path, "meta-store.csv")),
            luigi.LocalTarget(os.path.join(assets_path, "meta-item.csv")),
        )

    def output(self):
        dir_path = os.path.join("output", self.__class__.__name__, self.task_id)
        return luigi.LocalTarget(os.path.join(dir_path, "cardinalities.json"))

    def run(self):
        os.makedirs(os.path.split(self.output().path)[0])

        df = pd.read_csv(self.input()[0].path)
        store_df = pd.read_csv(self.input()[1].path)
        item_df = pd.read_csv(self.input()[2].path)

        df = pd.merge(df, store_df, on="S100")
        df = pd.merge(df, item_df, on="I100")

        for variable in self.categorical_variables:
            # Increment categorical variables (that are already indexed) by one to use 0 as unknown
            df[variable] += 1

        with open(self.output().path, "w") as f:
            json.dump((df[list(self.categorical_variables)].max() + 1).to_dict(), f)


class PrepareGluonTimeSeriesDatasets(luigi.Task):
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

    def requires(self):
        return PreProcessDataset(
            categorical_variables=self.categorical_variables,
            id_variables=self.id_variables,
        )

    def output(self):
        return luigi.LocalTarget(
            os.path.join("output", self.__class__.__name__, self.task_id)
        )

    def run(self):
        os.makedirs(self.output().path)

        filepaths = glob(os.path.join(self.input()[1].path, "*.parquet"))

        list(
            tqdm(
                map(
                    functools.partial(
                        _save_dataset_item,
                        categorical_variables=self.categorical_variables,
                        output_dir=self.output().path,
                    ),
                    filepaths,
                ),
                total=len(filepaths),
            )
        )


def _link_filtered_json_file(
    json_file_path: str, output_dir: str, min_target_length: int
):
    data = load_json_gzip(json_file_path, DataEntry)
    if len(data["target"]) > min_target_length:
        symlink_to_dir(json_file_path, output_dir)


class FilterGluonTimeSeriesDatasets(luigi.Task):
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
    split_steps: int = luigi.IntParameter(default=4)
    test_steps: int = luigi.IntParameter(default=4)

    def requires(self):
        return PrepareGluonTimeSeriesDatasets(
            categorical_variables=self.categorical_variables,
            id_variables=self.id_variables,
        )

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                "output",
                self.__class__.__name__,
                "%s-split_index=%d-split_steps=%d-test_steps=%d"
                % (self.task_id, self.split_index, self.split_steps, self.test_steps),
            )
        )

    def run(self):
        os.makedirs(self.output().path)

        min_target_length = abs(
            calculate_split(self.split_index, self.split_steps, self.test_steps)
        )

        with Pool(os.cpu_count()) as pool:
            json_file_paths = glob(
                os.path.join(self.input().path, "*.json.gz")
            )
            list(
                tqdm(
                    pool.map(
                        functools.partial(
                            _link_filtered_json_file,
                            output_dir=self.output().path,
                            min_target_length=min_target_length,
                        ),
                        json_file_paths,
                    ),
                    total=len(json_file_paths),
                )
            )
