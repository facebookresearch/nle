# This file taken from
#     https://github.com/deepmind/scalable_agent/blob/
#         d24bd74bd53d454b7222b7f0bea57a358e4ca33e/vtrace_test.py
# and modified.

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for V-trace.

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.
"""

import unittest

import numpy as np
import torch
import vtrace


def _shaped_arange(*shape):
    """Runs np.arange, converts to float and reshapes."""
    return np.arange(np.prod(shape), dtype=np.float32).reshape(*shape)


def _softmax(logits):
    """Applies softmax non-linearity on inputs."""
    return np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)


def _ground_truth_calculation(
    discounts,
    log_rhos,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold,
    clip_pg_rho_threshold,
):
    """Calculates the ground truth for V-trace in Python/Numpy."""
    vs = []
    seq_len = len(discounts)
    rhos = np.exp(log_rhos)
    cs = np.minimum(rhos, 1.0)
    clipped_rhos = rhos
    if clip_rho_threshold:
        clipped_rhos = np.minimum(rhos, clip_rho_threshold)
    clipped_pg_rhos = rhos
    if clip_pg_rho_threshold:
        clipped_pg_rhos = np.minimum(rhos, clip_pg_rho_threshold)

    # This is a very inefficient way to calculate the V-trace ground truth.
    # We calculate it this way because it is close to the mathematical notation
    # of V-trace.
    # v_s = V(x_s)
    #             + \sum^{T-1}_{t=s} \gamma^{t-s}
    #                 * \prod_{i=s}^{t-1} c_i
    #                 * \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))
    # Note that when we take the product over c_i, we write `s:t` as the
    # notation of the paper is inclusive of the `t-1`, but Python is exclusive.
    # Also note that np.prod([]) == 1.
    values_t_plus_1 = np.concatenate([values, bootstrap_value[None, :]], axis=0)
    for s in range(seq_len):
        v_s = np.copy(values[s])  # Very important copy.
        for t in range(s, seq_len):
            v_s += (
                np.prod(discounts[s:t], axis=0)
                * np.prod(cs[s:t], axis=0)
                * clipped_rhos[t]
                * (rewards[t] + discounts[t] * values_t_plus_1[t + 1] - values[t])
            )
        vs.append(v_s)
    vs = np.stack(vs, axis=0)
    pg_advantages = clipped_pg_rhos * (
        rewards
        + discounts * np.concatenate([vs[1:], bootstrap_value[None, :]], axis=0)
        - values
    )

    return vtrace.VTraceReturns(vs=vs, pg_advantages=pg_advantages)


def assert_allclose(actual, desired):
    return np.testing.assert_allclose(actual, desired, rtol=1e-06, atol=1e-05)


class ActionLogProbsTest(unittest.TestCase):
    def test_action_log_probs(self, batch_size=2):
        seq_len = 7
        num_actions = 3

        policy_logits = _shaped_arange(seq_len, batch_size, num_actions) + 10
        actions = np.random.randint(
            0, num_actions, size=(seq_len, batch_size), dtype=np.int64
        )

        action_log_probs_tensor = vtrace.action_log_probs(
            torch.from_numpy(policy_logits), torch.from_numpy(actions)
        )

        # Ground Truth
        # Using broadcasting to create a mask that indexes action logits
        action_index_mask = actions[..., None] == np.arange(num_actions)

        def index_with_mask(array, mask):
            return array[mask].reshape(*array.shape[:-1])

        # Note: Normally log(softmax) is not a good idea because it's not
        # numerically stable. However, in this test we have well-behaved values.
        ground_truth_v = index_with_mask(
            np.log(_softmax(policy_logits)), action_index_mask
        )

        assert_allclose(ground_truth_v, action_log_probs_tensor)

    def test_action_log_probs_batch_1(self):
        self.test_action_log_probs(1)


class VtraceTest(unittest.TestCase):
    def test_vtrace(self, batch_size=5):
        """Tests V-trace against ground truth data calculated in python."""
        seq_len = 5

        # Create log_rhos such that rho will span from near-zero to above the
        # clipping thresholds. In particular, calculate log_rhos in [-2.5, 2.5),
        # so that rho is in approx [0.08, 12.2).
        log_rhos = _shaped_arange(seq_len, batch_size) / (batch_size * seq_len)
        log_rhos = 5 * (log_rhos - 0.5)  # [0.0, 1.0) -> [-2.5, 2.5).
        values = {
            "log_rhos": log_rhos,
            # T, B where B_i: [0.9 / (i+1)] * T
            "discounts": np.array(
                [[0.9 / (b + 1) for b in range(batch_size)] for _ in range(seq_len)],
                dtype=np.float32,
            ),
            "rewards": _shaped_arange(seq_len, batch_size),
            "values": _shaped_arange(seq_len, batch_size) / batch_size,
            "bootstrap_value": _shaped_arange(batch_size) + 1.0,
            "clip_rho_threshold": 3.7,
            "clip_pg_rho_threshold": 2.2,
        }

        ground_truth = _ground_truth_calculation(**values)

        values = {key: torch.tensor(value) for key, value in values.items()}
        output = vtrace.from_importance_weights(**values)

        for a, b in zip(ground_truth, output):
            assert_allclose(a, b)

    def test_vtrace_batch_1(self):
        self.test_vtrace(1)

    def test_vtrace_from_logits(self, batch_size=2):
        """Tests V-trace calculated from logits."""
        seq_len = 5
        num_actions = 3
        clip_rho_threshold = None  # No clipping.
        clip_pg_rho_threshold = None  # No clipping.

        values = {
            "behavior_policy_logits": _shaped_arange(seq_len, batch_size, num_actions),
            "target_policy_logits": _shaped_arange(seq_len, batch_size, num_actions),
            "actions": np.random.randint(
                0, num_actions - 1, size=(seq_len, batch_size)
            ),
            "discounts": np.array(  # T, B where B_i: [0.9 / (i+1)] * T
                [[0.9 / (b + 1) for b in range(batch_size)] for _ in range(seq_len)],
                dtype=np.float32,
            ),
            "rewards": _shaped_arange(seq_len, batch_size),
            "values": _shaped_arange(seq_len, batch_size) / batch_size,
            "bootstrap_value": _shaped_arange(batch_size) + 1.0,  # B
        }
        values = {k: torch.from_numpy(v) for k, v in values.items()}

        from_logits_output = vtrace.from_logits(
            clip_rho_threshold=clip_rho_threshold,
            clip_pg_rho_threshold=clip_pg_rho_threshold,
            **values,
        )

        target_log_probs = vtrace.action_log_probs(
            values["target_policy_logits"], values["actions"]
        )
        behavior_log_probs = vtrace.action_log_probs(
            values["behavior_policy_logits"], values["actions"]
        )
        log_rhos = target_log_probs - behavior_log_probs

        # Calculate V-trace using the ground truth logits.
        from_iw = vtrace.from_importance_weights(
            log_rhos=log_rhos,
            discounts=values["discounts"],
            rewards=values["rewards"],
            values=values["values"],
            bootstrap_value=values["bootstrap_value"],
            clip_rho_threshold=clip_rho_threshold,
            clip_pg_rho_threshold=clip_pg_rho_threshold,
        )

        assert_allclose(from_iw.vs, from_logits_output.vs)
        assert_allclose(from_iw.pg_advantages, from_logits_output.pg_advantages)
        assert_allclose(
            behavior_log_probs, from_logits_output.behavior_action_log_probs
        )
        assert_allclose(target_log_probs, from_logits_output.target_action_log_probs)
        assert_allclose(log_rhos, from_logits_output.log_rhos)

    def test_vtrace_from_logits_batch_1(self):
        self.test_vtrace_from_logits(1)

    def test_higher_rank_inputs_for_importance_weights(self):
        """Checks support for additional dimensions in inputs."""
        T = 3  # pylint: disable=invalid-name
        B = 2  # pylint: disable=invalid-name
        values = {
            "log_rhos": torch.zeros(T, B, 1),
            "discounts": torch.zeros(T, B, 1),
            "rewards": torch.zeros(T, B, 42),
            "values": torch.zeros(T, B, 42),
            "bootstrap_value": torch.zeros(B, 42),
        }
        output = vtrace.from_importance_weights(**values)
        self.assertSequenceEqual(output.vs.shape, (T, B, 42))

    def test_inconsistent_rank_inputs_for_importance_weights(self):
        """Test one of many possible errors in shape of inputs."""
        T = 3  # pylint: disable=invalid-name
        B = 2  # pylint: disable=invalid-name

        values = {
            "log_rhos": torch.zeros(T, B, 1),
            "discounts": torch.zeros(T, B, 1),
            "rewards": torch.zeros(T, B, 42),
            "values": torch.zeros(T, B, 42),
            # Should be [B, 42].
            "bootstrap_value": torch.zeros(B),
        }

        with self.assertRaisesRegex(
            RuntimeError, "same number of dimensions: got 3 and 2"
        ):
            vtrace.from_importance_weights(**values)


if __name__ == "__main__":
    unittest.main()
