import pytest
from pyspark.sql import SparkSession

from pyspark_job import clean_data


@pytest.fixture(scope="session")
def spark():
    return (
        SparkSession.builder
        .master("local[2]")
        .appName("test_clean_data")
        .getOrCreate()
    )


def test_valid_records_are_kept(spark):
    data = [
        (1, "Ahmed", 100.0),
        (2, "Mohamed", 200.0)
    ]

    df = spark.createDataFrame(
        data,
        ["id", "name", "amount"]
    )

    result = clean_data(df)

    assert result.count() == 2


def test_records_with_invalid_amount_are_removed(spark):
    data = [
        (1, "Ahmed", 100.0),
        (2, "Mohamed", 0.0),
        (3, "Ali", -50.0)
    ]

    df = spark.createDataFrame(
        data,
        ["id", "name", "amount"]
    )

    result = clean_data(df)


    assert result.count() == 1


def test_records_with_null_names_are_removed(spark):
    data = [
        (1, "Ahmed", 100.0),
        (2, None, 200.0)
    ]

    df = spark.createDataFrame(
        data,
        ["id", "name", "amount"]
    )

    result = clean_data(df)

    assert result.count() == 1


def test_amount_with_tax_is_calculated_correctly(spark):
    data = [
        (1, "Ahmed", 100.0),
        (2, "Mohamed", 50.0)
    ]

    df = spark.createDataFrame(
        data,
        ["id", "name", "amount"]
    )

    result = clean_data(df)

    rows = result.orderBy("id").collect()

    assert rows[0]["amount_with_tax"] == pytest.approx(120.0)
    assert rows[1]["amount_with_tax"] == pytest.approx(60.0)
