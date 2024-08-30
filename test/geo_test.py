# geo_test.py
#     this file contations the test cases for the geo.py file
# by: Noah Syrkis

# %% Imports
import parabellum.geo as geo
import pytest


# %% Test Cases
@pytest.mark.parametrize("place, expected", [
    ("New York City", (40.7127281, -74.0060152)),
    ("Chicago", (41.8755616, -87.6244212))])
def test_get_coordinates(place, expected):
    assert geo.get_coordinates(place) == expected
