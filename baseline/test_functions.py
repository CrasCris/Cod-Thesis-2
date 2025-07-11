from functions import smape

import numpy as np
import pytest

def test_smape():
    pred = np.array([10,20,5])
    real = np.array([12,18,6])
    assert smape(pred,real) == pytest.approx(15.63, abs=1e-3)

def test_smape2():
    real = np.array([10,20,5])
    pred = np.array([12,18,6])
    assert smape(pred,real) == pytest.approx(15.63, abs=1e-3)

def test_smap3():
    real = np.ones(10)
    pred = 0.9*real
    assert smape(pred,real) == pytest.approx(1000/95, abs=1e-3)
