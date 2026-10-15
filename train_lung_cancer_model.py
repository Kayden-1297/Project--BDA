#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Tuple

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lung cancer prediction model with Big Data (PySpark) concepts."
    )
    parser.add_argument("--dataset-path", required=True, help="Path to dataset (csv or parquet).")
    parser.add_argument(
        "--target-col",
        default="LUNG_CANCER",
        help="Target column name (default: LUNG_CANCER).",
    )
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "csv", "parquet"],
        default="auto",
        help="Dataset format (default: auto).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio between 0 and 1 (default: 0.2).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory to store model and metrics (default: artifacts).",
    )
    return parser.parse_args()


def read_dataset(spark: SparkSession, dataset_path: str, dataset_format: str) -> DataFrame:
    if dataset_format == "auto":
        if dataset_path.lower().endswith(".parquet"):
            dataset_format = "parquet"
        else:
            dataset_format = "csv"

    if dataset_format == "parquet":
        return spark.read.parquet(dataset_path)
    return spark.read.option("header", "true").option("inferSchema", "true").csv(dataset_path)


def find_feature_columns(df: DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for field in df.schema.fields:
        if field.name in {target_col, "label"}:
            continue
        if isinstance(field.dataType, NumericType):
            numeric_cols.append(field.name)
        elif isinstance(field.dataType, StringType):
            categorical_cols.append(field.name)
    return numeric_cols, categorical_cols


def build_feature_stages(numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[List, str]:
    stages: List = []
    assembled_inputs: List[str] = []

    if numeric_cols:
        imputed_cols = [f"{c}_imputed" for c in numeric_cols]
        stages.append(
            Imputer(inputCols=numeric_cols, outputCols=imputed_cols, strategy="median")
        )
        assembled_inputs.extend(imputed_cols)

    if categorical_cols:
        indexed_cols = [f"{c}_idx" for c in categorical_cols]
        encoded_cols = [f"{c}_ohe" for c in categorical_cols]
        stages.append(
            StringIndexer(
                inputCols=categorical_cols,
                outputCols=indexed_cols,
                handleInvalid="keep",
            )
        )
        stages.append(
            OneHotEncoder(inputCols=indexed_cols, outputCols=encoded_cols, handleInvalid="keep")
        )
        assembled_inputs.extend(encoded_cols)

    if not assembled_inputs:
        raise ValueError("No usable feature columns found. Add numeric or categorical predictors.")

    stages.append(VectorAssembler(inputCols=assembled_inputs, outputCol="raw_features"))
    stages.append(StandardScaler(inputCol="raw_features", outputCol="features", withMean=False))
    return stages, "features"


def evaluate_predictions(predictions: DataFrame) -> dict:
    auc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    ).evaluate(predictions)
    accuracy = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    ).evaluate(predictions)
    f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    ).evaluate(predictions)
    return {"auc": auc, "accuracy": accuracy, "f1": f1}


def distinct_label_count(df: DataFrame) -> int:
    return df.select("label").distinct().count()


def min_label_class_size(df: DataFrame) -> int:
    return df.groupBy("label").count().agg(F.min("count").alias("min_count")).collect()[0]["min_count"]


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")

    spark = (
        SparkSession.builder.appName("LungCancerBigDataModel")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )

    df = read_dataset(spark, args.dataset_path, args.dataset_format)
    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in dataset columns: {df.columns}")

    df = df.dropDuplicates().na.drop(subset=[args.target_col])

    label_indexer = StringIndexer(
        inputCol=args.target_col, outputCol="label", handleInvalid="skip"
    ).fit(df)
    df = label_indexer.transform(df)
    if distinct_label_count(df) < 2:
        raise ValueError(
            f"Target column '{args.target_col}' contains only one class after preprocessing."
        )

    numeric_cols, categorical_cols = find_feature_columns(df, args.target_col)
    feature_stages, feature_col = build_feature_stages(numeric_cols, categorical_cols)

    train_df, test_df = df.randomSplit([1 - args.test_size, args.test_size], seed=args.seed)
    if distinct_label_count(train_df) < 2 or distinct_label_count(test_df) < 2:
        fractions = {
            row["label"]: (1 - args.test_size) for row in df.select("label").distinct().collect()
        }
        sampled_train = df.stat.sampleBy("label", fractions=fractions, seed=args.seed)
        train_df = sampled_train
        test_df = df.subtract(sampled_train)

    if distinct_label_count(train_df) < 2 or distinct_label_count(test_df) < 2:
        raise ValueError(
            "Train/test split does not contain both classes. Provide more balanced data "
            "or reduce --test-size."
        )

    evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )

    lr = LogisticRegression(featuresCol=feature_col, labelCol="label", maxIter=100)
    rf = RandomForestClassifier(featuresCol=feature_col, labelCol="label", numTrees=100)

    common_stages = feature_stages

    lr_pipeline = Pipeline(stages=common_stages + [lr])
    lr_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.01, 0.1])
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
        .build()
    )
    num_folds = min(3, min_label_class_size(train_df))
    if num_folds < 2:
        num_folds = 2
    lr_cv = CrossValidator(
        estimator=lr_pipeline,
        estimatorParamMaps=lr_grid,
        evaluator=evaluator,
        numFolds=num_folds,
        seed=args.seed,
    )

    rf_pipeline = Pipeline(stages=common_stages + [rf])
    rf_grid = (
        ParamGridBuilder().addGrid(rf.maxDepth, [5, 10]).addGrid(rf.minInstancesPerNode, [1, 3]).build()
    )
    rf_cv = CrossValidator(
        estimator=rf_pipeline,
        estimatorParamMaps=rf_grid,
        evaluator=evaluator,
        numFolds=num_folds,
        seed=args.seed,
    )

    lr_model = lr_cv.fit(train_df)
    rf_model = rf_cv.fit(train_df)

    lr_predictions = lr_model.transform(test_df)
    rf_predictions = rf_model.transform(test_df)
    lr_metrics = evaluate_predictions(lr_predictions)
    rf_metrics = evaluate_predictions(rf_predictions)

    lr_rank = (lr_metrics["auc"], lr_metrics["f1"], lr_metrics["accuracy"])
    rf_rank = (rf_metrics["auc"], rf_metrics["f1"], rf_metrics["accuracy"])
    best_name = "logistic_regression"
    best_model = lr_model.bestModel
    best_metrics = lr_metrics
    if rf_rank > lr_rank:
        best_name = "random_forest"
        best_model = rf_model.bestModel
        best_metrics = rf_metrics

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "best_model")
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    best_model.write().overwrite().save(model_path)

    output = {
        "best_model": best_name,
        "target_col": args.target_col,
        "feature_columns": {"numeric": numeric_cols, "categorical": categorical_cols},
        "logistic_regression_metrics": lr_metrics,
        "random_forest_metrics": rf_metrics,
        "selected_model_metrics": best_metrics,
        "rows": {
            "total": df.count(),
            "train": train_df.count(),
            "test": test_df.count(),
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
