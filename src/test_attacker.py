"""Tests for the evolutionary attacker. Checks that attack transactions are
featurized into the detector's feature space correctly, and that the evolution
responds to the detector it faces (a wall catches everything; an open door
catches nothing).

    python src/test_attacker.py
"""
from __future__ import annotations

import numpy as np

from attacker import EvolutionaryAttacker, VictimPool, attack_rows
from features import ALL_FEATURES


def _tiny_pool():
    return VictimPool(
        ts=[np.array([0.0, 3600.0, 7200.0]), np.array([0.0, 86400.0])],
        amt_prefix=[np.array([0.0, 10.0, 20.0, 30.0]), np.array([0.0, 50.0, 100.0])],
        last_t=np.array([7200.0, 86400.0]),
        home_lat=np.array([40.0, 34.0]),
        home_lon=np.array([-74.0, -118.0]),
        last_lat=np.array([40.0, 34.0]),
        last_lon=np.array([-74.0, -118.0]),
        median_amt=np.array([10.0, 50.0]),
    )


def test_attack_rows_shape_and_finite():
    pool = _tiny_pool()
    genome = {"amount_mult": 10.0, "burst_size": 3.0, "gap_s": 120.0,
              "distance_km": 500.0, "category": 5.0}
    X, amt = attack_rows(genome, pool, mr_global=0.01)
    assert X.shape == (2 * 3, len(ALL_FEATURES))  # 2 victims x burst 3
    assert np.isfinite(X).all()
    assert (amt > 0).all()
    # amount_ratio_user_median column should equal amount_mult (amount / median).
    j = ALL_FEATURES.index("amount_ratio_user_median")
    assert np.allclose(X[:, j], 10.0)


def test_evolution_responds_to_defense():
    pool = _tiny_pool()
    tau = 0.5
    # A wall: every transaction scored as fraud -> nothing gets through.
    wall = EvolutionaryAttacker(pop_size=8, generations=3, seed=1)
    _, _, _, profit_wall = wall.step(lambda X: np.ones(len(X)), tau, pool, 0.01)
    assert profit_wall == 0.0

    # An open door: everything scored as safe -> the attacker's value gets through.
    door = EvolutionaryAttacker(pop_size=8, generations=3, seed=1)
    _, _, _, profit_door = door.step(lambda X: np.zeros(len(X)), tau, pool, 0.01)
    assert profit_door > 0.0
    assert door.n_probes > 0


if __name__ == "__main__":
    test_attack_rows_shape_and_finite()
    print("PASS test_attack_rows_shape_and_finite")
    test_evolution_responds_to_defense()
    print("PASS test_evolution_responds_to_defense")
