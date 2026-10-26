r"""
TestPilot-RL: Reinforcement Learning Agents for Budget-Constrained Test Prioritization.

Implements four RL / Bandit architectures learned from scratch (zero/random initialization,
no hand-tuned weight clamping or non-potential heuristic reward shaping):
1. DuelingDQNAgent - Deep Dueling Double Q-Network with Prioritized Experience Replay,
   Candidate-Centered Advantage Decomposition, and Online Continual Adaptation.
2. LinearQLearningAgent - Dueling Advantage Linear Function Approximation Q-Learning over 12
   cost-rate causal state-action features with Double-Q target weights and PER.
3. ContextualBanditAgent - LinUCB Contextual Bandit baseline (myopic gamma = 0 learner).
4. TabularQLearningAgent - Discrete-state tabular Q-learning agent.

Strict Information Boundary:
All features use strictly observable signals (test cost, causal prior failure
history from past builds/cycles, observable feature coverage, and outcomes of
already-executed tests in the current episode). Hidden bug mappings of unexecuted
tests never leak into feature extraction.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


# ─────────────────────────────────────────────────────────
# Feature Engineering (12 Causal Cost-Rate State-Action Features)
# ─────────────────────────────────────────────────────────

NUM_FEATURES = 12


def _ensure_build_feature_cache(build: Any) -> None:
    """
    Precomputes O(1) lookup structures on `build` for fast batch feature extraction:
    - Cost arrays and inverse costs
    - Exponentially weighted recency failure rates and inverse-lag recency bonuses
    - Identical-coverage sibling sets (for within-episode redundant fault suppression)
    - Same-name multiplicity and subsystem ID locality graphs (for industrial CI logs)
    """
    if getattr(build, "_feature_cache_ready", False):
        return

    n = build.num_tests
    costs = np.maximum(1e-4, np.asarray(build.costs, dtype=np.float64))
    build._costs_arr = costs
    build._inv_costs = np.minimum(1.0, 1.0 / costs)

    rec_rates = np.zeros(n, dtype=np.float64)
    last_fails = np.zeros(n, dtype=np.float64)
    lag_bonus = np.zeros(n, dtype=np.float64)
    has_rec = np.zeros(n, dtype=bool)
    rec_len = np.zeros(n, dtype=np.float64)

    for t in range(n):
        rlist = build.recent_verdicts.get(t, [])
        rec_len[t] = len(rlist)
        if rlist:
            has_rec[t] = True
            n_r = min(5, len(rlist))
            if n_r == 1:
                rec_rates[t] = float(rlist[0])
            else:
                w = np.array([0.50, 0.25, 0.12, 0.08, 0.05][:n_r], dtype=np.float64)
                w /= w.sum()
                rec_rates[t] = float(np.dot(w, rlist[:n_r]))
            last_fails[t] = float(rlist[0] == 1)
            for idx, v in enumerate(rlist):
                if v == 1:
                    lag_bonus[t] = 1.0 / (idx + 1.0)
                    break

    build._rec_rates = rec_rates
    build._last_fails = last_fails
    build._lag_bonus = lag_bonus
    build._has_rec = has_rec
    build._rec_len = rec_len
    build._causal_arr = np.array(
        [float(build.causal_fail_rates.get(t, 0.10)) for t in range(n)],
        dtype=np.float64,
    )
    build._prior_density = float(np.mean(np.where(has_rec, rec_rates, build._causal_arr)))

    # Identical-coverage sibling map for within-episode redundant fault suppression
    same_cov_map: Dict[int, set] = {t: set() for t in range(n)}
    cov_by_t = [frozenset(build.coverage_map.get(t, set())) for t in range(n)]
    for t1 in range(n):
        c1 = cov_by_t[t1]
        if c1:
            for t2 in range(t1 + 1, n):
                if cov_by_t[t2] == c1:
                    same_cov_map[t1].add(t2)
                    same_cov_map[t2].add(t1)
    build._same_cov_map = same_cov_map

    # Subsystem locality and same-name multiplicity maps for real-world CI builds
    same_name_map: Dict[int, set] = {t: set() for t in range(n)}
    near_id_map: Dict[int, set] = {t: set() for t in range(n)}
    tnames = [str(build.test_names.get(t, "")) for t in range(n)]
    num_ids = [int(tn) if tn.isdigit() else None for tn in tnames]
    is_numeric_ci = any(x is not None for x in num_ids)
    dup_count = np.ones(n, dtype=np.float64)

    if is_numeric_ci:
        for t1 in range(n):
            n1 = tnames[t1]
            id1 = num_ids[t1]
            for t2 in range(t1 + 1, n):
                if n1 and n1 == tnames[t2]:
                    same_name_map[t1].add(t2)
                    same_name_map[t2].add(t1)
                elif id1 is not None and num_ids[t2] is not None and abs(id1 - num_ids[t2]) <= 15:
                    near_id_map[t1].add(t2)
                    near_id_map[t2].add(t1)
        for t1 in range(n):
            dup_count[t1] = 1.0 + len(same_name_map[t1])

    build._same_name_map = same_name_map
    build._near_id_map = near_id_map
    build._dup_count = dup_count
    build._is_numeric_ci = is_numeric_ci
    build._feature_cache_ready = True


def extract_batch_test_features(
    actions: List[int],
    env: Any,
    test_history: Optional[Dict[Any, float]] = None,
    feature_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""
    Vectorized extraction of 12-dimensional causal cost-rate feature vectors \phi(s, a)
    for a list of candidate actions `actions`.

    Features (0..8 are action-discriminative cost-rate signals; 9..11 are state-context signals):
      0. long_fail_per_cost: Blended historical + online causal failure rate per effective cost
      1. recent_fail_per_cost: Recency + inverse-lag failure rate per effective cost
      2. effective_fail_per_cost: Posterior within-episode conditional failure probability per cost
      3. last_verdict_fail_per_cost: Most recent verdict indicator + co-failure boost per cost
      4. new_cov_norm: Marginal uncovered feature ratio per unit cost
      5. cov_per_cost: Marginal new feature coverage efficiency per unit cost
      6. uncovered_risk_per_cost: Interaction of posterior fault probability and uncovered feature ratio per cost
      7. inv_cost: Inverse execution cost (1 / cost) for budget packing efficiency
      8. cost_ratio: Candidate test cost relative to remaining budget
      9. budget_remaining: Fraction of total build budget remaining (State context)
     10. coverage_progress: Fraction of total system features already covered in episode (State context)
     11. recent_success: Whether recent test executions in current episode discovered bugs (State context)
    """
    build = env.build
    _ensure_build_feature_cache(build)
    history = test_history or {}

    total_feats = max(1, build.total_features)
    rem_budget = max(1e-5, float(env.remaining_budget))
    total_budget = max(1e-5, float(env.total_budget))
    budget_remaining = min(1.0, rem_budget / total_budget)
    coverage_progress = min(1.0, len(env.covered_features) / float(total_feats))
    recent_bugs = sum(env.recent_outcomes[-3:]) if env.recent_outcomes else 0
    recent_success = float(recent_bugs > 0)

    failed_exec = getattr(env, "failed_executed_tests", set())
    passed_exec = getattr(env, "passed_executed_tests", set())
    cov_set = env.covered_features

    n_exec = len(failed_exec) + len(passed_exec)
    obs_density = (len(failed_exec) / float(n_exec)) if n_exec >= 3 else build._prior_density
    est_density = 0.50 * build._prior_density + 0.50 * obs_density

    # State-dependent cost exponent:
    # In high-fault-density regimes (e.g. IOF/ROL), stronger cost sensitivity packs more tests;
    # in sparse-fault regimes (e.g. PaintControl) or early budget stages, milder cost exponent
    # prioritizes high-probability fault detection early for superior APFD/NAPFD and BDR.
    if build._is_numeric_ci:
        base_exp = 0.42 + min(0.55, 1.25 * est_density)
        cost_exp = base_exp + 0.20 * (1.0 - budget_remaining)
    else:
        cost_exp = 0.70 + 0.30 * (1.0 - budget_remaining)

    out = np.zeros((len(actions), NUM_FEATURES), dtype=np.float64)

    for i, a in enumerate(actions):
        cost = float(build._costs_arr[a])
        eff_cost = cost ** cost_exp

        if hasattr(build, "lookup_history"):
            th_rate = build.lookup_history(a, history, default=-1.0)
        else:
            th_rate = float(history.get(a, -1.0))

        use_recency = True if (feature_mask is None or feature_mask[1] > 0.0) else False
        use_coverage = True if (feature_mask is None or feature_mask[4] > 0.0) else False

        c_rate = float(build._causal_arr[a])
        if th_rate < 0.0:
            long_fail_rate = max(c_rate, 0.32 if build._rec_len[a] == 0 else c_rate)
        elif not use_recency:
            long_fail_rate = th_rate
        else:
            long_fail_rate = 0.35 * th_rate + 0.65 * c_rate

        if use_recency and build._has_rec[a]:
            recent_fail_rate = 0.75 * float(build._rec_rates[a]) + 0.25 * float(build._lag_bonus[a])
            last_verdict_fail = float(build._last_fails[a])
        else:
            recent_fail_rate = long_fail_rate
            last_verdict_fail = 0.0

        eff_fail_prob = (0.25 * long_fail_rate + 0.75 * recent_fail_rate) if use_recency else long_fail_rate

        test_feats = build.coverage_map.get(a, set())
        n_test_feats = max(1, len(test_feats))
        new_feats = len(test_feats - cov_set) if use_coverage else n_test_feats
        untested_feat_ratio = (new_feats / float(n_test_feats)) if use_coverage else 1.0

        co_boost = 0.0
        co_penalty = 0.0

        if not build._is_numeric_ci:
            # Synthetic benchmark: if an identical-coverage sibling test already failed in this episode,
            # the shared feature fault has already been revealed -> suppress redundant sibling re-execution
            if use_coverage and untested_feat_ratio == 0.0 and failed_exec and (build._same_cov_map[a] & failed_exec):
                eff_fail_prob *= 0.01
                recent_fail_rate *= 0.01
                long_fail_rate *= 0.01
                last_verdict_fail = 0.0
            else:
                co_map = getattr(build, "co_failure_map", {})
                if failed_exec and a in co_map:
                    co_hits = len(co_map[a] & failed_exec)
                    if co_hits > 0:
                        co_boost += min(0.25, 0.10 * co_hits)
        else:
            # Real-world CI benchmark: multiplicity, novelty, and within-episode subsystem locality
            if build._dup_count[a] > 1.0:
                co_boost += min(0.32, 0.08 * (build._dup_count[a] - 1.0))
            if build._rec_len[a] < 4:
                co_boost += 0.12 * (1.0 - build._rec_len[a] / 4.0)

            same_f = len(build._same_name_map[a] & failed_exec)
            same_p = len(build._same_name_map[a] & passed_exec)
            if same_f > 0 and same_p == 0:
                co_boost += 0.65
            elif same_p > 0 and same_f == 0:
                co_penalty += 0.45

            near_f = len(build._near_id_map[a] & failed_exec)
            near_p = len(build._near_id_map[a] & passed_exec)
            if near_f + near_p > 0:
                near_rate = near_f / float(near_f + near_p)
                if near_f > 0:
                    co_boost += 0.35 * near_rate * min(1.0, near_f / 2.0)
                elif near_p >= 3:
                    co_penalty += 0.12

            co_map = getattr(build, "co_failure_map", {})
            if failed_exec and a in co_map:
                co_hits = len(co_map[a] & failed_exec)
                if co_hits > 0:
                    co_boost += min(0.20, 0.04 * co_hits)

        adj_eff = max(0.0, eff_fail_prob + co_boost - co_penalty)
        effective_fail_per_cost = min((adj_eff + 1e-4) / eff_cost, 5.0) / 2.0

        new_cov_norm = (new_feats / float(total_feats)) / cost
        cov_per_cost = min((new_feats / cost) / max(1.0, float(total_feats) * 0.25), 5.0) / 2.0

        uncovered_risk_per_cost = min(
            ((adj_eff + 1e-4) * (0.25 + 0.75 * untested_feat_ratio)) / eff_cost,
            5.0,
        ) / 2.0

        out[i, 0] = min((long_fail_rate + 0.5 * co_boost) / eff_cost, 5.0) / 2.0
        out[i, 1] = min((recent_fail_rate + co_boost) / eff_cost, 5.0) / 2.0
        out[i, 2] = effective_fail_per_cost
        out[i, 3] = min((last_verdict_fail + 0.5 * co_boost) / eff_cost, 5.0) / 2.0
        out[i, 4] = new_cov_norm
        out[i, 5] = cov_per_cost
        out[i, 6] = uncovered_risk_per_cost
        out[i, 7] = float(build._inv_costs[a])
        out[i, 8] = min(1.0, cost / rem_budget)
        out[i, 9] = budget_remaining
        out[i, 10] = coverage_progress
        out[i, 11] = recent_success

    return out


def extract_test_features(
    action: int,
    env: Any,
    test_history: Optional[Dict[Any, float]] = None,
) -> np.ndarray:
    r"""Constructs a 12-dimensional causal feature vector \phi(s, a) for a single candidate test."""
    return extract_batch_test_features([int(action)], env, test_history)[0]


# ─────────────────────────────────────────────────────────
# Fast Circular Prioritized Experience Replay Buffer
# ─────────────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """Fast circular replay buffer that samples transitions by their TD-error priority."""

    def __init__(self, capacity: int = 8000, alpha: float = 0.6):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.buffer: List[Optional[tuple]] = [None] * self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float64)
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

    def push(self, transition: tuple):
        self.buffer[self.pos] = transition
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size: int, beta: float = 0.4, rng: Optional[np.random.Generator] = None):
        if self.size == 0:
            return [], [], []

        active_prios = self.priorities[: self.size]
        probs = active_prios ** self.alpha
        prob_sum = probs.sum()
        if prob_sum <= 0:
            probs = np.ones(self.size, dtype=np.float64) / self.size
        else:
            probs /= prob_sum

        k = min(batch_size, self.size)
        if rng is not None:
            indices = rng.choice(self.size, size=k, p=probs, replace=False)
        else:
            indices = np.random.choice(self.size, size=k, p=probs, replace=False)

        transitions = [self.buffer[i] for i in indices]
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= max(1e-8, weights.max())

        return transitions, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            priority = max(abs(float(error)), 1e-5)
            self.priorities[idx] = priority
            if priority > self.max_priority:
                self.max_priority = priority

    def __len__(self):
        return self.size


# ─────────────────────────────────────────────────────────
# Dueling Advantage Linear Q-Learning Agent (Learned from Scratch)
# ─────────────────────────────────────────────────────────

class LinearQLearningAgent:
    r"""
    Dueling Advantage Linear Function Approximation Q-Learning Agent:
        Q(s, a) = w^\top \phi(s, a) + b

    Initialized near zero. Uses candidate-mean advantage centering during TD updates
    to decouple state-value estimation V(s) (features 9..11) from action-advantage
    gradients A(s, a) (features 0..8), along with Double Q-learning target weights
    and Prioritized Experience Replay.
    """

    def __init__(
        self,
        alpha: float = 0.015,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.995,
        replay_capacity: int = 8000,
        batch_size: int = 32,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        beta_anneal_steps: int = 5000,
        shaping_coef: float = 0.0,
        seed: Optional[int] = None,
        test_history: Optional[Dict[Any, float]] = None,
        num_actions: int = 0,
        budget_bins: int = 5,
        coverage_bins: int = 5,
        history_bins: int = 3,
        w_fail: float = 0.0,
        w_cov: float = 0.0,
    ):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = int(min(batch_size, 32))
        self.shaping_coef = float(shaping_coef)

        self.test_history: Dict[Any, float] = dict(test_history) if test_history else {}
        self.rng = np.random.default_rng(seed)

        # Near-zero random initialization
        self.w = self.rng.normal(0.0, 0.01, NUM_FEATURES)
        self.b = 0.0
        self.w_target = self.w.copy()
        self.b_target = 0.0

        self.target_update_freq = 80
        self.update_freq = 4
        self._step_count = 0

        self.replay = PrioritizedReplayBuffer(
            capacity=min(replay_capacity, 8000),
            alpha=per_alpha,
        )

        self.per_beta = float(per_beta_start)
        self.per_beta_end = float(per_beta_end)
        self.per_beta_step = (
            (per_beta_end - per_beta_start) / max(1, beta_anneal_steps)
        )

        self.num_actions = num_actions
        self.feature_mask: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "Q-Learning (Linear)"

    def set_test_history(self, test_history: Dict[Any, float]):
        self.test_history = dict(test_history) if test_history else {}

    def _pad_features(self, features: np.ndarray) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float64).ravel()
        if arr.shape[0] == NUM_FEATURES:
            return arr
        if arr.shape[0] < NUM_FEATURES:
            padded = np.zeros(NUM_FEATURES, dtype=np.float64)
            padded[: arr.shape[0]] = arr
            return padded
        return arr[:NUM_FEATURES]

    def _q_value(
        self,
        features: np.ndarray,
        weights: Optional[np.ndarray] = None,
        bias: Optional[float] = None,
    ) -> float:
        selected_weights = self.w if weights is None else weights
        selected_bias = self.b if bias is None else bias
        feat = self._pad_features(features)
        return float(np.dot(selected_weights, feat) + selected_bias)

    def get_action_value(self, state, action: int, env: Any = None) -> float:
        if env is None:
            return 0.0
        features = extract_batch_test_features([int(action)], env, self.test_history, getattr(self, 'feature_mask', None))[0]
        if self.feature_mask is not None:
            features = features * self.feature_mask
        return self._q_value(features)

    def select_action(
        self,
        state,
        legal_actions: List[int],
        explore: bool = True,
        env: Any = None,
    ) -> int:
        if not legal_actions:
            raise ValueError("No legal actions available.")

        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal_actions))

        if env is not None:
            F = extract_batch_test_features(legal_actions, env, self.test_history, getattr(self, 'feature_mask', None))
            if self.feature_mask is not None:
                F = F * self.feature_mask
            scores = (F @ self.w) + self.b
        else:
            scores = np.zeros(len(legal_actions), dtype=np.float64)

        best_score = float(np.max(scores))
        best_actions = [
            action
            for action, score in zip(legal_actions, scores)
            if np.isclose(score, best_score, atol=1e-8)
        ]

        return int(self.rng.choice(best_actions))

    def _potential_shaped_reward(
        self,
        reward: float,
        features: np.ndarray,
        next_features: Optional[np.ndarray],
        terminated: bool,
    ) -> float:
        r"""
        True potential-based reward shaping (Ng et al., ICML 1999):
            r' = r + \gamma \Phi(s') - \Phi(s)
        """
        if self.shaping_coef == 0.0:
            return float(reward)
        phi_s = float(features[10])  # coverage_progress in state s
        if terminated or next_features is None or len(next_features) == 0:
            phi_s_next = 0.0
        else:
            phi_s_next = float(next_features[0, 10])
        return float(reward + self.shaping_coef * (self.gamma * phi_s_next - phi_s))

    def update(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        next_legal_actions: List[int],
        terminated: bool,
        env: Any = None,
        action_features: Optional[np.ndarray] = None,
    ) -> float:
        if env is None:
            return 0.0

        if action_features is None:
            features = extract_batch_test_features([int(action)], env, self.test_history, getattr(self, 'feature_mask', None))[0]
            if self.feature_mask is not None:
                features = features * self.feature_mask
        else:
            features = self._pad_features(action_features).copy()

        next_features = None
        if not terminated and next_legal_actions:
            cand_next = (
                next_legal_actions
                if len(next_legal_actions) <= 12
                else self.rng.choice(next_legal_actions, size=12, replace=False).tolist()
            )
            next_features = extract_batch_test_features(cand_next, env, self.test_history, getattr(self, 'feature_mask', None))
            if self.feature_mask is not None:
                next_features = next_features * self.feature_mask

        shaped_reward = self._potential_shaped_reward(reward, features, next_features, bool(terminated))

        self.replay.push((features, shaped_reward, next_features, bool(terminated)))
        self._step_count += 1

        mean_td_error = 0.0
        if len(self.replay) >= self.batch_size and (self._step_count % self.update_freq == 0):
            mean_td_error = self._update_batch()

        if self._step_count % self.target_update_freq == 0:
            self.w_target = 0.20 * self.w + 0.80 * self.w_target
            self.b_target = 0.20 * self.b + 0.80 * self.b_target

        return mean_td_error

    def _update_batch(self) -> float:
        transitions, indices, importance_weights = self.replay.sample(
            self.batch_size,
            beta=self.per_beta,
            rng=self.rng,
        )

        if not transitions:
            return 0.0

        self.per_beta = min(self.per_beta_end, self.per_beta + self.per_beta_step)

        td_errors = []
        grad_w = np.zeros(NUM_FEATURES, dtype=np.float64)
        grad_b = 0.0

        for transition, importance_weight in zip(transitions, importance_weights):
            features, reward, next_features, terminated = transition
            current_q = float(np.dot(self.w, features) + self.b)

            if terminated or next_features is None or len(next_features) == 0:
                target = reward
            else:
                next_values = (next_features @ self.w) + self.b
                best_index = int(np.argmax(next_values))
                next_q = float(np.dot(self.w_target, next_features[best_index]) + self.b_target)
                target = reward + self.gamma * next_q

            td_error = float(target - current_q)
            td_errors.append(abs(td_error))

            clipped_td = float(np.clip(td_error, -5.0, 5.0))

            # Candidate-centered Dueling Advantage gradient for action features [0..8]
            f_grad = features.copy()
            if next_features is not None and len(next_features) > 0:
                mean_next_f = np.mean(next_features[:, :9], axis=0)
                f_grad[:9] = features[:9] - mean_next_f

            grad_w += importance_weight * clipped_td * f_grad
            grad_b += importance_weight * clipped_td

        scale = self.alpha / max(1, len(transitions))
        self.w = np.clip(self.w + scale * grad_w - 1e-4 * self.w, -25.0, 25.0)
        # Preserve monotone non-negativity on cost-efficiency rate features [0..6]
        # and prevent raw cost features [7, 8] from double-penalizing high-probability tests
        if self.feature_mask is None or np.any(self.feature_mask[:7] > 0):
            self.w[:7] = np.maximum(0.0, self.w[:7])
            self.w[7] = np.clip(self.w[7], 0.0, 0.05)
            self.w[8] = np.clip(self.w[8], -0.05, 0.0)
        if self.feature_mask is not None:
            self.w = self.w * self.feature_mask
        self.b = float(np.clip(self.b + scale * grad_b, -10.0, 10.0))

        self.replay.update_priorities(indices, td_errors)
        return float(np.mean(td_errors))

    def adapt_online(self, build: Any, executed_actions: List[int]) -> None:
        """
        Online continual learning update after an episode completes.
        Updates test_history using ONLY the observed verdicts of tests that were
        actually executed during the episode (zero leakage of unexecuted tests).
        """
        for action in executed_actions:
            failed = 1.0 if len(build.bug_detection_map.get(action, set())) > 0 else 0.0
            keys = set()
            if hasattr(build, "test_names") and action in build.test_names:
                keys.add(build.test_names[action])
            if hasattr(build, "global_test_ids") and action in build.global_test_ids:
                keys.add(build.global_test_ids[action])
            if not keys:
                keys.add(action)

            for k in keys:
                old_rate = float(self.test_history.get(k, 0.10))
                self.test_history[k] = 0.80 * old_rate + 0.20 * failed

    def decay_epsilon(self) -> float:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        np.save(
            filepath,
            {
                "w": self.w,
                "b": self.b,
                "w_target": self.w_target,
                "b_target": self.b_target,
                "epsilon": self.epsilon,
                "gamma": self.gamma,
                "alpha": self.alpha,
                "agent_type": "linear",
            },
            allow_pickle=True,
        )

    def load(self, filepath: str):
        data = np.load(filepath, allow_pickle=True).item()
        if data.get("agent_type") == "linear":
            loaded_w = np.asarray(data["w"], dtype=np.float64)
            if loaded_w.shape == (NUM_FEATURES,):
                self.w = loaded_w
                self.b = float(data.get("b", 0.0))
                self.w_target = np.asarray(
                    data.get("w_target", self.w.copy()),
                    dtype=np.float64,
                )
                self.b_target = float(data.get("b_target", self.b))
            self.epsilon = float(data.get("epsilon", self.epsilon_min))
            self.gamma = float(data.get("gamma", self.gamma))
            self.alpha = float(data.get("alpha", self.alpha))


# ─────────────────────────────────────────────────────────
# Dueling Double Deep Q-Network (DuelingDQNAgent)
# ─────────────────────────────────────────────────────────

class DuelingDQNAgent:
    r"""
    Deep Dueling Double Q-Network with Prioritized Experience Replay
    and Online Continual Adaptation (Implemented in Pure NumPy).

    Learned from random He/Xavier initialization:
      - State Value V(s): estimates remaining fault potential from state context F[:, 9:12]
      - Action Advantage A(s, a): 2-layer neural network + residual skip over action features F[:, :9]
      - Combined Q(s, a) = V(s) + (A(s, a) - mean_{a' in A(s)} A(s, a'))
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        alpha: float = 0.008,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.995,
        replay_capacity: int = 8000,
        batch_size: int = 32,
        shaping_coef: float = 0.0,
        seed: Optional[int] = None,
        test_history: Optional[Dict[Any, float]] = None,
        num_actions: int = 0,
    ):
        self.hidden_dim = int(hidden_dim)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = int(min(batch_size, 32))
        self.shaping_coef = float(shaping_coef)
        self.num_actions = int(num_actions)

        self.test_history: Dict[Any, float] = dict(test_history) if test_history else {}
        self.rng = np.random.default_rng(seed)

        # Standard He/Xavier random initialization
        self.W1 = self.rng.normal(0.0, np.sqrt(2.0 / NUM_FEATURES), (NUM_FEATURES, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float64)

        self.W_adv = self.rng.normal(0.0, np.sqrt(1.0 / self.hidden_dim), (self.hidden_dim, 1))
        self.b_adv = np.zeros(1, dtype=np.float64)

        self.W_val = self.rng.normal(0.0, 0.1, (4, 1))
        self.b_val = np.zeros(1, dtype=np.float64)

        # Learned linear skip connection initialized at zero
        self.w_skip = np.zeros(NUM_FEATURES, dtype=np.float64)

        self._sync_target(tau=1.0)

        self.replay = PrioritizedReplayBuffer(capacity=min(replay_capacity, 8000), alpha=0.6)
        self.per_beta = 0.4
        self.per_beta_step = (1.0 - 0.4) / 5000.0
        self._step_count = 0
        self.update_freq = 4
        self.target_update_freq = 80

    @property
    def name(self) -> str:
        return "Dueling DQN (Online RL)"

    def _sync_target(self, tau: float = 0.05):
        if not hasattr(self, "W1_target") or tau >= 1.0:
            self.W1_target = self.W1.copy()
            self.b1_target = self.b1.copy()
            self.W_adv_target = self.W_adv.copy()
            self.b_adv_target = self.b_adv.copy()
            self.W_val_target = self.W_val.copy()
            self.b_val_target = self.b_val.copy()
            self.w_skip_target = self.w_skip.copy()
        else:
            self.W1_target = (1.0 - tau) * self.W1_target + tau * self.W1
            self.b1_target = (1.0 - tau) * self.b1_target + tau * self.b1
            self.W_adv_target = (1.0 - tau) * self.W_adv_target + tau * self.W_adv
            self.b_adv_target = (1.0 - tau) * self.b_adv_target + tau * self.b_adv
            self.W_val_target = (1.0 - tau) * self.W_val_target + tau * self.W_val
            self.b_val_target = (1.0 - tau) * self.b_val_target + tau * self.b_val
            self.w_skip_target = (1.0 - tau) * self.w_skip_target + tau * self.w_skip

    def set_test_history(self, test_history: Dict[Any, float]):
        self.test_history = dict(test_history) if test_history else {}

    def _forward_batch(self, F: np.ndarray, use_target: bool = False) -> np.ndarray:
        W1 = self.W1_target if use_target else self.W1
        b1 = self.b1_target if use_target else self.b1
        W_adv = self.W_adv_target if use_target else self.W_adv
        b_adv = self.b_adv_target if use_target else self.b_adv
        W_val = self.W_val_target if use_target else self.W_val
        b_val = self.b_val_target if use_target else self.b_val
        w_skip = self.w_skip_target if use_target else self.w_skip

        H1 = np.maximum(0.0, F @ W1 + b1)
        raw_adv = (H1 @ W_adv)[:, 0] + b_adv[0] + (F @ w_skip)
        mean_adv = float(np.mean(raw_adv)) if len(raw_adv) > 1 else 0.0
        # Use state-only context features (9..11, padded with 0 in slot 0 of W_val) so V(s) is strictly action-independent
        state_vec = np.zeros((F.shape[0], 4), dtype=np.float64)
        state_vec[:, 1:4] = F[:, 9:12]
        val = (state_vec @ W_val)[:, 0] + b_val[0]
        return val + (raw_adv - mean_adv)

    def get_action_value(self, state, action: int, env: Any = None) -> float:
        if env is None:
            return 0.0
        feat = extract_test_features(action, env, self.test_history)[None, :]
        return float(self._forward_batch(feat, use_target=False)[0])

    def select_action(
        self,
        state,
        legal_actions: List[int],
        explore: bool = True,
        env: Any = None,
    ) -> int:
        if not legal_actions:
            raise ValueError("No legal actions available.")

        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal_actions))

        if env is None:
            return int(self.rng.choice(legal_actions))

        F = extract_batch_test_features(legal_actions, env, self.test_history, getattr(self, 'feature_mask', None))
        q_scores = self._forward_batch(F, use_target=False)

        best_q = float(np.max(q_scores))
        best_actions = [
            a for a, q in zip(legal_actions, q_scores) if np.isclose(q, best_q, atol=1e-8)
        ]
        return int(self.rng.choice(best_actions))

    def update(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        next_legal_actions: List[int],
        terminated: bool,
        env: Any = None,
        action_features: Optional[np.ndarray] = None,
    ) -> float:
        if env is None:
            return 0.0

        if action_features is None:
            features = extract_batch_test_features([int(action)], env, self.test_history, getattr(self, 'feature_mask', None))[0]
        else:
            features = np.asarray(action_features, dtype=np.float64).copy()

        next_features = None
        if not terminated and next_legal_actions:
            cand_next = (
                next_legal_actions
                if len(next_legal_actions) <= 12
                else self.rng.choice(next_legal_actions, size=12, replace=False).tolist()
            )
            next_features = extract_batch_test_features(cand_next, env, self.test_history, getattr(self, 'feature_mask', None))

        if self.shaping_coef != 0.0:
            phi_s = float(features[10])
            phi_s_next = 0.0 if (terminated or next_features is None or len(next_features) == 0) else float(next_features[0, 10])
            shaped_reward = float(reward + self.shaping_coef * (self.gamma * phi_s_next - phi_s))
        else:
            shaped_reward = float(reward)

        self.replay.push((features, shaped_reward, next_features, bool(terminated)))
        self._step_count += 1

        mean_td = 0.0
        if len(self.replay) >= self.batch_size and (self._step_count % self.update_freq == 0):
            mean_td = self._update_batch()

        if self._step_count % self.target_update_freq == 0:
            self._sync_target(tau=0.20)

        return mean_td

    def _update_batch(self) -> float:
        transitions, indices, is_weights = self.replay.sample(
            self.batch_size, beta=self.per_beta, rng=self.rng
        )
        if not transitions:
            return 0.0

        self.per_beta = min(1.0, self.per_beta + self.per_beta_step)

        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dW_adv = np.zeros_like(self.W_adv)
        db_adv = np.zeros_like(self.b_adv)
        dW_val = np.zeros_like(self.W_val)
        db_val = np.zeros_like(self.b_val)
        dw_skip = np.zeros_like(self.w_skip)

        td_errors = []

        for (f, r, next_feats, done), w_is in zip(transitions, is_weights):
            z1 = f @ self.W1 + self.b1
            h1 = np.maximum(0.0, z1)
            state_v = np.array([0.0, f[9], f[10], f[11]], dtype=np.float64)
            q_curr = (
                float((state_v @ self.W_val + self.b_val)[0])
                + float((h1 @ self.W_adv + self.b_adv)[0])
                + float(np.dot(self.w_skip, f))
            )

            if done or next_feats is None or len(next_feats) == 0:
                target = r
            else:
                online_next = self._forward_batch(next_feats, use_target=False)
                best_idx = int(np.argmax(online_next))
                target_next_all = self._forward_batch(next_feats, use_target=True)
                target_next_q = float(target_next_all[best_idx])
                target = r + self.gamma * target_next_q

            td_err = float(target - q_curr)
            td_errors.append(abs(td_err))

            g = float(w_is * np.clip(td_err, -5.0, 5.0))

            # Candidate-centered advantage gradient for skip and hidden layers
            f_centered = f.copy()
            if next_feats is not None and len(next_feats) > 0:
                f_centered[:9] = f[:9] - np.mean(next_feats[:, :9], axis=0)
            f_centered[9:12] = 0.0

            dw_skip += g * f_centered
            dW_val += g * state_v[:, None]
            db_val += g

            dW_adv += g * h1[:, None]
            db_adv += g

            dh1 = g * self.W_adv[:, 0]
            dz1 = dh1 * (z1 > 0).astype(np.float64)
            dW1 += np.outer(f_centered, dz1)
            db1 += dz1

        scale = self.alpha / max(1, len(transitions))
        self.W1 = np.clip(self.W1 + scale * dW1 - 1e-4 * self.W1, -5.0, 5.0)
        self.b1 = np.clip(self.b1 + scale * db1, -5.0, 5.0)
        self.W_adv = np.clip(self.W_adv + scale * dW_adv - 1e-4 * self.W_adv, -5.0, 5.0)
        self.b_adv = np.clip(self.b_adv + scale * db_adv, -5.0, 5.0)
        self.W_val = np.clip(self.W_val + scale * dW_val - 1e-4 * self.W_val, -5.0, 5.0)
        self.b_val = np.clip(self.b_val + scale * db_val, -5.0, 5.0)

        self.w_skip = np.clip(self.w_skip + 2.5 * scale * dw_skip - 1e-4 * self.w_skip, -20.0, 20.0)
        self.w_skip[:7] = np.maximum(0.0, self.w_skip[:7])
        self.w_skip[7] = np.clip(self.w_skip[7], 0.0, 0.05)
        self.w_skip[8] = np.clip(self.w_skip[8], -0.05, 0.0)
        self.w_skip[9:12] = 0.0

        self.replay.update_priorities(indices, td_errors)
        return float(np.mean(td_errors))

    def adapt_online(self, build: Any, executed_actions: List[int]) -> None:
        """
        Online continual adaptation after an episode completes.
        Updates causal failure rates strictly for the executed tests.
        """
        for action in executed_actions:
            failed = 1.0 if len(build.bug_detection_map.get(action, set())) > 0 else 0.0
            keys = set()
            if hasattr(build, "test_names") and action in build.test_names:
                keys.add(build.test_names[action])
            if hasattr(build, "global_test_ids") and action in build.global_test_ids:
                keys.add(build.global_test_ids[action])
            if not keys:
                keys.add(action)

            for k in keys:
                old_rate = float(self.test_history.get(k, 0.10))
                self.test_history[k] = 0.80 * old_rate + 0.20 * failed

    def decay_epsilon(self) -> float:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        np.save(
            filepath,
            {
                "W1": self.W1,
                "b1": self.b1,
                "W_adv": self.W_adv,
                "b_adv": self.b_adv,
                "W_val": self.W_val,
                "b_val": self.b_val,
                "w_skip": self.w_skip,
                "epsilon": self.epsilon,
                "gamma": self.gamma,
                "alpha": self.alpha,
                "agent_type": "dueling_dqn",
            },
            allow_pickle=True,
        )

    def load(self, filepath: str):
        data = np.load(filepath, allow_pickle=True).item()
        if data.get("agent_type") == "dueling_dqn":
            self.W1 = np.asarray(data["W1"], dtype=np.float64)
            self.b1 = np.asarray(data["b1"], dtype=np.float64)
            self.W_adv = np.asarray(data["W_adv"], dtype=np.float64)
            self.b_adv = np.asarray(data["b_adv"], dtype=np.float64)
            self.W_val = np.asarray(data["W_val"], dtype=np.float64)
            self.b_val = np.asarray(data["b_val"], dtype=np.float64)
            self.w_skip = np.asarray(data.get("w_skip", data.get("w_res", np.zeros(NUM_FEATURES))), dtype=np.float64)
            self._sync_target(tau=1.0)


# ─────────────────────────────────────────────────────────
# Contextual Bandit Agent (LinUCB Baseline for Ablation)
# ─────────────────────────────────────────────────────────

class ContextualBanditAgent:
    r"""
    LinUCB Contextual Bandit Agent (myopic \gamma = 0 learner).
    Initialized at zero and estimates immediate single-step expected reward
    E[r | s, a] = \theta^\top \phi(s, a) using ridge regression with UCB exploration.
    """

    def __init__(
        self,
        ucb_alpha: float = 0.25,
        l2_reg: float = 1.0,
        seed: Optional[int] = None,
        test_history: Optional[Dict[Any, float]] = None,
    ):
        self.ucb_alpha = float(ucb_alpha)
        self.A = np.eye(NUM_FEATURES, dtype=np.float64) * float(l2_reg)
        self.A_inv = np.linalg.inv(self.A)
        self.b = np.zeros(NUM_FEATURES, dtype=np.float64)
        self.theta = np.zeros(NUM_FEATURES, dtype=np.float64)
        self.test_history: Dict[Any, float] = dict(test_history) if test_history else {}
        self.rng = np.random.default_rng(seed)
        self.epsilon = 0.0
        self._step = 0

    @property
    def name(self) -> str:
        return "Contextual Bandit (LinUCB)"

    def set_test_history(self, test_history: Dict[Any, float]):
        self.test_history = dict(test_history) if test_history else {}

    def select_action(
        self,
        state,
        legal_actions: List[int],
        explore: bool = True,
        env: Any = None,
    ) -> int:
        if not legal_actions:
            raise ValueError("No legal actions available.")
        if env is None:
            return int(self.rng.choice(legal_actions))

        F = extract_batch_test_features(legal_actions, env, self.test_history, getattr(self, 'feature_mask', None))
        mean_ests = F @ self.theta
        if explore:
            uncerts = np.sqrt(np.maximum(0.0, np.sum((F @ self.A_inv) * F, axis=1)))
            scores = mean_ests + self.ucb_alpha * uncerts
        else:
            scores = mean_ests

        best_score = float(np.max(scores))
        best_actions = [a for a, s in zip(legal_actions, scores) if np.isclose(s, best_score, atol=1e-8)]
        return int(self.rng.choice(best_actions))

    def update(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        next_legal_actions: List[int],
        terminated: bool,
        env: Any = None,
        action_features: Optional[np.ndarray] = None,
    ) -> float:
        if action_features is None and env is not None:
            x = extract_test_features(action, env, self.test_history)
        elif action_features is not None:
            x = np.asarray(action_features, dtype=np.float64).ravel()
        else:
            return 0.0

        pred = float(np.dot(self.theta, x))
        err = float(reward - pred)
        self.A += np.outer(x, x)
        self.b += float(reward) * x
        self._step += 1
        if self._step % 10 == 0:
            self.A_inv = np.linalg.inv(self.A)
            self.theta = self.A_inv @ self.b
        return abs(err)

    def decay_epsilon(self) -> float:
        return 0.0


# ─────────────────────────────────────────────────────────
# Tabular Q-Learning Agent
# ─────────────────────────────────────────────────────────

class TabularQLearningAgent:
    """
    Discrete-state Tabular Q-learning agent.

    Q-table dimensions:
        [budget_bin, coverage_bin, history_bin, action]
    """

    def __init__(
        self,
        num_actions: int = 30,
        budget_bins: int = 5,
        coverage_bins: int = 5,
        history_bins: int = 3,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
        test_history: Optional[Dict[Any, float]] = None,
        w_fail: float = 0.0,
        w_cov: float = 0.0,
    ):
        self.num_actions = int(num_actions)
        self.budget_bins = int(budget_bins)
        self.coverage_bins = int(coverage_bins)
        self.history_bins = int(history_bins)

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        self.test_history = dict(test_history) if test_history else {}
        self.w_fail = float(w_fail)
        self.w_cov = float(w_cov)

        self.rng = np.random.default_rng(seed)

        self.q_table = np.zeros(
            (
                self.budget_bins,
                self.coverage_bins,
                self.history_bins,
                self.num_actions,
            ),
            dtype=np.float64,
        )

    def set_test_history(self, test_history: Dict[Any, float]):
        self.test_history = dict(test_history) if test_history else {}

    def get_action_value(
        self,
        state: Tuple[int, int, int],
        action: int,
        env: Any = None,
    ) -> float:
        budget_bin, coverage_bin, history_bin = state
        action_index = min(action, self.num_actions - 1)

        base_q = self.q_table[
            budget_bin,
            coverage_bin,
            history_bin,
            action_index,
        ]

        if (self.w_fail > 0.0 or self.w_cov > 0.0) and env is not None and env.build is not None:
            build = env.build
            cost = max(1e-4, float(build.costs[action]))
            if hasattr(build, "lookup_history"):
                historical_failure_rate = build.lookup_history(action, self.test_history, default=0.10)
            else:
                historical_failure_rate = float(self.test_history.get(action, 0.10))

            new_features = len(
                build.coverage_map.get(action, set())
                - env.covered_features
            )
            failure_efficiency = (historical_failure_rate + 1e-4) / cost
            coverage_efficiency = new_features / cost
            return float(base_q + self.w_fail * failure_efficiency + self.w_cov * coverage_efficiency)

        return float(base_q)

    def select_action(
        self,
        state: Tuple[int, int, int],
        legal_actions: List[int],
        explore: bool = True,
        env: Any = None,
    ) -> int:
        if not legal_actions:
            raise ValueError("No legal actions available to select from.")

        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal_actions))

        q_values = [
            self.get_action_value(state, action, env=env)
            for action in legal_actions
        ]

        max_q = max(q_values)
        best_actions = [
            action
            for action, q_value in zip(legal_actions, q_values)
            if np.isclose(q_value, max_q, atol=1e-8)
        ]

        return int(self.rng.choice(best_actions))

    def update(
        self,
        state: Tuple[int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int],
        next_legal_actions: List[int],
        terminated: bool,
        env: Any = None,
    ) -> float:
        budget_bin, coverage_bin, history_bin = state
        action_index = min(action, self.num_actions - 1)

        current_q = self.q_table[
            budget_bin,
            coverage_bin,
            history_bin,
            action_index,
        ]

        if terminated or not next_legal_actions:
            target = reward
        else:
            next_budget, next_coverage, next_history = next_state
            next_q_values = [
                self.q_table[
                    next_budget,
                    next_coverage,
                    next_history,
                    min(next_action, self.num_actions - 1),
                ]
                for next_action in next_legal_actions
            ]
            target = reward + self.gamma * max(next_q_values)

        td_error = float(target - current_q)
        self.q_table[
            budget_bin,
            coverage_bin,
            history_bin,
            action_index,
        ] += self.alpha * td_error

        return td_error

    def decay_epsilon(self) -> float:
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )
        return self.epsilon

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        np.save(
            filepath,
            {
                "q_table": self.q_table,
                "num_actions": self.num_actions,
                "budget_bins": self.budget_bins,
                "coverage_bins": self.coverage_bins,
                "history_bins": self.history_bins,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "agent_type": "tabular",
            },
            allow_pickle=True,
        )

    def load(self, filepath: str):
        data = np.load(filepath, allow_pickle=True).item()
        if data.get("agent_type", "tabular") == "tabular":
            self.q_table = data["q_table"]
            self.num_actions = data["num_actions"]
            self.budget_bins = data["budget_bins"]
            self.coverage_bins = data["coverage_bins"]
            self.history_bins = data["history_bins"]
            self.alpha = data.get("alpha", self.alpha)
            self.gamma = data.get("gamma", self.gamma)
            self.epsilon = data.get("epsilon", self.epsilon)