# Unit tests for functions in src/data_prep/utils.py
# Please note that this is just the start to illustrate the idea
# With more time, I'd want to fully test out the whole pipeline, including the
# models, and I would want to check test coverage (e.g. with pytest-cov)
# The tests/ directory is structured to follow the same layout as src/
import pytest
import pandas as pd
import pandas as pd
from src.data_prep.utils import *


def test_convert_to_float():
    assert (
        convert_to_float(pd.Series(["0", " 1", "-2.5  "]))
        == pd.Series([0.0, 1.0, -2.5])
    ).all() == True


def test_clean_vote_feature():
    assert (
        clean_vote_feature(pd.Series(["1,000 ", "11", "  7  ", np.nan, np.nan]))
        == pd.Series([1000, 11, 7, 0, 0])
    ).all() == True
