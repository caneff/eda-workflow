from pathlib import Path

import pandas
import pytest


@pytest.fixture
def sample_eda_dataframe() -> pandas.DataFrame:
    return pandas.DataFrame({
        "region": ["North", "South", "North"],
        "sales": [10, None, 30],
        "units": [1, 2, None],
    })


@pytest.fixture
def sample_eda_csv_path(
    tmp_path: Path,
    sample_eda_dataframe: pandas.DataFrame,
) -> Path:
    csv_path = tmp_path / "sample_eda.csv"
    sample_eda_dataframe.to_csv(csv_path, index=False)
    return csv_path
