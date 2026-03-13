#!/usr/bin/env python3
"""
mind_model.py
================

A simple cognitive simulation inspired by the predictive global neuronal
workspace (GNW) hypothesis.  The global workspace theory proposes that
a *functional hub* transiently binds and broadcasts information from many
specialised agents, enabling adaptive, context–dependent decision making
and conscious access【181609601806680†L159-L170】.  Recent work
in computational neuroscience extends this idea to artificial agents,
showing that architectures which recruit a global workspace can perform
robustly under constrained working–memory budgets while learning
meaningful attentional patterns【801959480511349†L61-L77】.

This module implements a lightweight predictive–processing agent with
16 branches, each representing a category of cognition (e.g. work,
education, relationships or hobbies).  At each time step the agent
computes a score for each branch based on multiple drivers (urgency,
novelty, reward and cost).  The scores are normalised via a softmax
with a temperature parameter to yield attention weights.  Branch
activities are combined via a non‑linear function and a coupling matrix
Γ (Gamma) to allow cross‑branch influence.  A global mind state
``M`` is produced by broadcasting the attention‑weighted activities;
prediction errors update internal state variables, and a resource
constraint prunes deep exploration when cognitive depth exceeds a
budget.

This program can be run as a script to simulate the agent over a
number of steps and prints summary statistics.  It is intended for
exploratory research and educational purposes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

import numpy as np


@dataclass
class MindModel:
    """A simple cognitive agent with predictive processing and a global workspace.

    Attributes
    ----------
    branches : List[str]
        Human‑readable labels for each cognitive branch.
    tau : float
        Temperature parameter for the softmax; lower values lead to more
        focused attention.  A temperature parameter is often used to
        model stochastic choice in Luce's choice rule or Boltzmann
        rationality.
    alpha, beta, gamma_a, delta, kappa : float
        Coefficients for computing the raw score for each branch.
    rho, eta : float
        Recursion update parameters for the ``R`` state.
    seed : int
        Random seed for reproducible simulations.
    """

    branches: List[str]
    tau: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0
    gamma_a: float = 0.5  # avoid clash with gamma matrix
    delta: float = 1.0
    kappa: float = 0.5
    rho: float = 0.9
    eta: float = 0.1
    seed: int = 0

    # Internal state arrays will be initialised in __post_init__
    N: int = field(init=False)
    L: np.ndarray = field(init=False)  # logic/leaf state
    P: np.ndarray = field(init=False)  # perception state
    E: np.ndarray = field(init=False)  # experience state
    R: np.ndarray = field(init=False)  # recursion/meta state
    W_L: np.ndarray = field(init=False)
    W_P: np.ndarray = field(init=False)
    W_E: np.ndarray = field(init=False)
    W_R: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    V: np.ndarray = field(init=False)  # goals vector
    C: float = field(init=False)  # resource capacity
    Gamma: np.ndarray = field(init=False)  # cross‑branch coupling

    # Running progress for a stage (e.g. Work)
    progress_work: float = field(default=0.0, init=False)
    stage_closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.N = len(self.branches)
        # Initialise internal states with small positive random values
        self.L = rng.random(self.N)
        self.P = rng.random(self.N)
        self.E = rng.random(self.N)
        self.R = rng.random(self.N)
        self.W_L = rng.random(self.N)
        self.W_P = rng.random(self.N)
        self.W_E = rng.random(self.N)
        self.W_R = rng.random(self.N)
        self.b = rng.random(self.N)
        self.V = rng.random(self.N)
        # Start with generous cognitive capacity
        self.C = 10.0
        # Initialise Γ with structured couplings
        self.Gamma = rng.random((self.N, self.N)) * 0.05
        # Health (index 9) influences all branches
        if self.N > 9:
            self.Gamma[:, 9] += 0.1
        # Relationships (index 7) boosts Work (index 2)
        if self.N > 7 and self.N > 2:
            self.Gamma[2, 7] += 0.2
        # Faith (index 13) dampens errors across branches
        if self.N > 13:
            self.Gamma[:, 13] -= 0.05
        # Zero the diagonal to avoid self‑loops
        np.fill_diagonal(self.Gamma, 0.0)
        # Set random seed for Python's random module as well
        random.seed(self.seed)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute a temperature‑scaled softmax of vector x."""
        # subtract max for numerical stability
        y = (x - np.max(x)) / max(self.tau, 1e-6)
        e = np.exp(y)
        return e / e.sum()

    def D_max(self) -> float:
        """Compute the maximum allowable depth given current capacity C.

        A simple linear scaling is used; other monotonic functions could
        be substituted.  Capacity ``C`` decays over time to model
        fatigue but recovers partially when health and cats signals are
        high (see ``step``).
        """
        return 0.5 * self.C

    def compute_score(self, align: float, urgency: np.ndarray, novelty: np.ndarray,
                      reward: np.ndarray, cost: np.ndarray) -> np.ndarray:
        """Compute raw score S_i for each branch."""
        # Avoid dividing by zero if capacity has been exhausted
        inv_C = 1.0 / max(self.C, 1e-6)
        return (
            self.alpha * align
            + self.beta * urgency
            + self.gamma_a * novelty
            + self.delta * reward
            - self.kappa * cost * inv_C
        )

    def step(self) -> Tuple[float, float, np.ndarray]:
        """Advance the model by one time step.

        Returns
        -------
        M : float
            The global mind state at this step.
        epsilon : float
            Prediction error between the predicted and actual next mind state.
        a : np.ndarray
            The attention distribution over branches for this step.
        """
        # Generate synthetic inputs (in a practical system these would be
        # derived from sensory and motivational signals)
        rng = np.random.default_rng()
        urgency = rng.random(self.N) * 2.0
        novelty = rng.random(self.N)
        reward = rng.random(self.N) * 3.0 - 1.0
        cost = rng.random(self.N) * 1.5
        depth = rng.random(self.N) * 5.0 + 1.0

        # Alignment score: simple dot of goals and a random direction
        align = float(np.dot(self.V, rng.random(self.N)))
        S = self.compute_score(align, urgency, novelty, reward, cost)
        a = self.softmax(S)

        # Branch fusion: compute leaf activations y_i
        y = np.tanh(
            self.W_L * self.L + self.W_P * self.P + self.W_E * self.E + self.W_R * self.R + self.b
        )
        # Coupling: incorporate cross‑branch influence
        y_tilde = y + self.Gamma.dot(y)

        # Global mind state is a broadcast (attention‑weighted sum)
        M = float(np.dot(a, y_tilde))

        # Predict next mind state with a simple linear predictor
        o_hat = 0.82 * M
        # Simulate the actual next mind state with some noise
        M_next_true = M + rng.normal(loc=0.0, scale=0.3)
        epsilon = float(M_next_true - o_hat)

        # Update experience E: error is assigned proportionally to attention
        events = epsilon * a
        self.E += events
        # Update recursion/meta state R using absolute error and branch activity
        meta = np.abs(epsilon) * y_tilde
        self.R = self.rho * self.R + self.eta * meta

        # Resource‑based pruning: ensure weighted depth does not exceed D_max
        total_depth = float(np.dot(a, depth))
        if total_depth > self.D_max():
            # Iteratively shrink attention on branches with lowest marginal utility
            a = self._apply_pruning(a, y_tilde, depth, total_depth)

        # Stage/task integration: if Work (index 2) receives sustained high
        # attention, mark the stage as completed
        if not self.stage_closed and self.N > 2:
            # accumulate progress proportional to attention on Work
            self.progress_work += a[2]
            if self.progress_work > 2.0:
                self.stage_closed = True
        # Decay capacity with a slight recovery when health (9) and cats (0) are strong
        health_signal = y[9] if self.N > 9 else 0.0
        cat_signal = y[0] if self.N > 0 else 0.0
        recovery_factor = 1.0 + 0.02 * (health_signal + cat_signal)
        self.C = self.C * 0.95 * recovery_factor

        return M, epsilon, a

    def _apply_pruning(self, a: np.ndarray, y_tilde: np.ndarray, depth: np.ndarray, current_depth: float) -> np.ndarray:
        """Shrink attention weights to satisfy the depth budget.

        A greedy algorithm reduces attention on the branch with the lowest
        marginal utility until the weighted depth is within budget.  To
        prevent infinite loops, a maximum number of iterations is
        enforced.  If the budget cannot be met within the limit, the
        current allocation is returned unchanged.
        """
        a = a.copy()
        max_depth = self.D_max()
        iterations = 0
        # Only perform pruning when the budget is exceeded
        while current_depth > max_depth and iterations < 50:
            iterations += 1
            # marginal utility per depth unit; add small constant to avoid div/0
            utility = (a * y_tilde) / (depth + 1e-8)
            idx = int(np.argmin(utility))
            # shrink attention on the least useful branch by 10%
            a[idx] *= 0.9
            # renormalise to preserve probabilistic interpretation
            s = a.sum()
            if s > 1e-12:
                a /= s
            current_depth = float(np.dot(a, depth))
        return a

    def simulate(self, T: int) -> Tuple[List[float], List[float]]:
        """Run the simulation for ``T`` time steps.

        Parameters
        ----------
        T : int
            Number of steps to simulate.

        Returns
        -------
        M_history : List[float]
            Global mind state values for each step.
        epsilon_history : List[float]
            Prediction error values for each step.
        """
        M_history: List[float] = []
        epsilon_history: List[float] = []
        for _ in range(T):
            M, eps, _ = self.step()
            M_history.append(M)
            epsilon_history.append(eps)
        return M_history, epsilon_history


def main() -> None:
    """Run the mind model simulation and print results."""
    branches = [
        "Cats", "Cars", "Work", "Education/Career", "AI/AR/Tech Projects",
        "Astronomy/Astrophotography", "Music/Creative", "Relationships",
        "Family/Genealogy", "Health/Mental Health", "Addiction/Recovery",
        "Finance/Workflows", "Magic: The Gathering", "Faith/Bible Study",
        "Utilities/How‑to", "Retail/Oils Mapping"
    ]
    # Instantiate the model with a fixed seed for reproducibility
    model = MindModel(branches=branches, tau=0.8, seed=42)
    steps = 100
    Ms, eps = model.simulate(steps)
    print(f"Simulated {steps} steps.")
    print(f"Final mind state M: {Ms[-1]:.4f}")
    print(f"Final prediction error ε: {eps[-1]:.4f}")
    print(f"Work stage closed: {model.stage_closed}")


if __name__ == "__main__":
    main()
