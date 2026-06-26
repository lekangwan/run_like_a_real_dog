"""Shared high-level reward metric formulas.

This module is the canonical score table for the high-level gait wrapper and
for offline consistency checks.  Keep formulas here small and tensor-only so
online and offline callers can compare the same trajectory without drift.
"""

import torch


CANONICAL_REWARD_NAMES = (
    "progress",
    "yaw_tracking",
    "tracking_gate",
    "tracking_gate_strict",
    "orientation",
    "pitch_rate",
    "roll_rate",
    "yaw_rate",
    "lateral_drift",
    "vertical_bounce",
    "slip",
    "contact_slip",
    "energy",
    "power_efficiency",
    "transport_efficiency",
    "impact",
    "scuffing",
    "clearance",
    "gait_stability",
    "action_smoothness",
    "action_magnitude",
    "action_boundary_margin",
    "survival",
    "safety_lateral_drift",
    "safety_contact_slip",
    "safety_impact",
    "safety_scuffing",
    "gated_orientation",
    "gated_lateral_drift",
    "gated_contact_slip",
    "gated_power_efficiency",
    "gated_transport_efficiency",
    "gated_impact",
    "gated_scuffing",
    "gated_action_smoothness",
    "strict_gated_power_efficiency",
)


UNIFIED_REWARD_PROFILES = {
    # Historical incomplete live proxies. Keep them available for reproducing
    # old diagnostics, but do not train with them as mainline rewards.
    "unified_efficiency": {
        "progress": 2.0,
        "yaw_tracking": 0.3,
        "orientation": 0.8,
        "lateral_drift": 0.4,
        "slip": 0.7,
        "energy": 2.0,
        "clearance": 0.2,
        "action_smoothness": 0.8,
        "action_boundary_margin": 0.5,
        "survival": 1.5,
    },
    "unified_balanced": {
        "progress": 1.5,
        "yaw_tracking": 0.3,
        "orientation": 1.0,
        "lateral_drift": 0.8,
        "slip": 1.0,
        "energy": 1.0,
        "clearance": 0.5,
        "action_smoothness": 0.5,
        "action_boundary_margin": 0.4,
        "survival": 2.0,
    },
    # Canonical candidates use explicit impact/scuffing scores from this shared
    # high-level reward metric table. They still require validation before PPO.
    "canonical_efficiency_candidate": {
        "progress": 2.0,
        "yaw_tracking": 0.3,
        "orientation": 0.8,
        "lateral_drift": 0.4,
        "slip": 0.7,
        "energy": 2.0,
        "impact": 1.5,
        "scuffing": 0.5,
        "clearance": 0.2,
        "action_smoothness": 0.8,
        "action_boundary_margin": 0.5,
        "survival": 1.5,
    },
    "canonical_balanced_candidate": {
        "progress": 1.5,
        "yaw_tracking": 0.3,
        "orientation": 1.0,
        "lateral_drift": 0.8,
        "slip": 1.0,
        "energy": 1.0,
        "impact": 1.0,
        "scuffing": 0.8,
        "clearance": 0.5,
        "action_smoothness": 0.5,
        "action_boundary_margin": 0.4,
        "survival": 2.0,
    },
    # v2 is the first metric-repair candidate after the 37-config sanity audit.
    # It keeps tracking/survival as base task terms and gates secondary quality
    # and efficiency terms by tracking quality with a nonzero floor.
    "canonical_efficiency_v2_candidate": {
        "progress": 3.0,
        "yaw_tracking": 0.4,
        "survival": 2.0,
        "gated_orientation": 1.0,
        "gated_lateral_drift": 0.6,
        "gated_contact_slip": 0.8,
        "gated_impact": 0.8,
        "gated_scuffing": 0.4,
        "gated_power_efficiency": 0.5,
        "gated_transport_efficiency": 0.5,
        "gated_action_smoothness": 0.4,
        "action_boundary_margin": 0.3,
    },
    # v3 physical is for fair gait audits.  It intentionally excludes action
    # regularizers such as smoothness, magnitude, boundary margin, and gait
    # switch stability from the ranking score.  Those terms may be logged or
    # later used as tiny PPO regularizers, but they should not decide gait-family
    # physical quality.
    "canonical_efficiency_v3_physical": {
        "progress": 3.0,
        "yaw_tracking": 0.4,
        "survival": 2.0,
        "gated_orientation": 1.0,
        "gated_lateral_drift": 0.6,
        "gated_contact_slip": 0.8,
        "gated_impact": 0.8,
        "gated_scuffing": 0.4,
        "gated_power_efficiency": 0.7,
    },
    # v4 keeps the reward unified but changes the structure from linear
    # compensation to tracking-first physical scoring.  Safety terms are
    # thresholded so small differences inside an acceptable band cannot
    # overpower command tracking; power efficiency is only active once tracking
    # is reasonably good.  Action regularizers remain excluded from fair gait
    # ranking.
    "canonical_efficiency_v4_physical": {
        "progress": 5.0,
        "yaw_tracking": 0.2,
        "survival": 2.0,
        "orientation": 0.8,
        "safety_lateral_drift": 0.25,
        "safety_contact_slip": 0.4,
        "safety_impact": 0.4,
        "safety_scuffing": 0.25,
        "strict_gated_power_efficiency": 0.8,
    },
}


def zero_like(reference):
    return torch.zeros_like(reference)


def threshold_score(score, low, high):
    return torch.clamp((score - low) / max(high - low, 1e-6), 0.0, 1.0)


def compute_metric_score_dict(
    *,
    velocity_reward,
    yaw_reward,
    orientation_penalty,
    pitch_rate_penalty,
    roll_rate_penalty,
    yaw_rate_penalty,
    lateral_velocity_penalty,
    lateral_position_penalty,
    vertical_velocity_penalty,
    slip_penalty,
    contact_slip_penalty=None,
    torque_penalty,
    mechanical_power_abs=None,
    transport_cost_proxy=None,
    clearance_reward,
    gait_switch_penalty,
    action_delta_penalty,
    continuous_action_penalty,
    action_boundary_penalty,
    fall_penalty,
    impact_velocity_rms=None,
    scuffing_ratio=None,
):
    """Return per-env normalized metric scores keyed by canonical names."""

    if impact_velocity_rms is None:
        impact_velocity_rms = zero_like(velocity_reward)
    if scuffing_ratio is None:
        scuffing_ratio = zero_like(velocity_reward)
    if contact_slip_penalty is None:
        contact_slip_penalty = slip_penalty
    if mechanical_power_abs is None:
        mechanical_power_abs = torque_penalty
    if transport_cost_proxy is None:
        transport_cost_proxy = mechanical_power_abs

    tracking_gate = torch.clamp(0.25 + 0.75 * velocity_reward, 0.25, 1.0)
    tracking_gate_strict = torch.clamp((velocity_reward - 0.75) / 0.25, 0.0, 1.0)
    orientation_score = torch.exp(-orientation_penalty / 0.05)
    lateral_drift_score = torch.exp(
        -lateral_velocity_penalty / 0.05 - lateral_position_penalty / 1.00
    )
    contact_slip_score = torch.exp(-contact_slip_penalty / 0.05)
    power_efficiency_score = torch.exp(-mechanical_power_abs / 250.0)
    transport_efficiency_score = torch.exp(-transport_cost_proxy / 250.0)
    impact_score = torch.exp(-(impact_velocity_rms**2) / 4.0)
    scuffing_score = torch.exp(-scuffing_ratio / 0.20)
    action_smoothness_score = torch.exp(-action_delta_penalty / 0.05)
    safety_lateral_drift_score = threshold_score(lateral_drift_score, 0.25, 0.50)
    safety_contact_slip_score = threshold_score(contact_slip_score, 0.60, 0.85)
    safety_impact_score = threshold_score(impact_score, 0.80, 0.92)
    safety_scuffing_score = threshold_score(scuffing_score, 0.70, 0.90)

    return {
        "progress": velocity_reward,
        "yaw_tracking": yaw_reward,
        "tracking_gate": tracking_gate,
        "tracking_gate_strict": tracking_gate_strict,
        "orientation": orientation_score,
        "pitch_rate": torch.exp(-pitch_rate_penalty / 0.25),
        "roll_rate": torch.exp(-roll_rate_penalty / 0.25),
        "yaw_rate": torch.exp(-yaw_rate_penalty / 0.25),
        "lateral_drift": lateral_drift_score,
        "vertical_bounce": torch.exp(-vertical_velocity_penalty / 0.05),
        "slip": torch.exp(-slip_penalty / 0.05),
        "contact_slip": contact_slip_score,
        "energy": torch.exp(-torque_penalty / 0.50),
        "power_efficiency": power_efficiency_score,
        "transport_efficiency": transport_efficiency_score,
        "impact": impact_score,
        "scuffing": scuffing_score,
        "clearance": clearance_reward,
        "gait_stability": torch.exp(-gait_switch_penalty / 0.25),
        "action_smoothness": action_smoothness_score,
        "action_magnitude": torch.exp(-continuous_action_penalty / 0.25),
        "action_boundary_margin": torch.exp(-action_boundary_penalty / 0.25),
        "survival": 1.0 - fall_penalty,
        "safety_lateral_drift": safety_lateral_drift_score,
        "safety_contact_slip": safety_contact_slip_score,
        "safety_impact": safety_impact_score,
        "safety_scuffing": safety_scuffing_score,
        "gated_orientation": tracking_gate * orientation_score,
        "gated_lateral_drift": tracking_gate * lateral_drift_score,
        "gated_contact_slip": tracking_gate * contact_slip_score,
        "gated_power_efficiency": tracking_gate * power_efficiency_score,
        "gated_transport_efficiency": tracking_gate * transport_efficiency_score,
        "gated_impact": tracking_gate * impact_score,
        "gated_scuffing": tracking_gate * scuffing_score,
        "gated_action_smoothness": tracking_gate * action_smoothness_score,
        "strict_gated_power_efficiency": tracking_gate_strict * power_efficiency_score,
    }


def stack_metric_scores(score_dict, reward_names=CANONICAL_REWARD_NAMES):
    return torch.stack(tuple(score_dict[name] for name in reward_names), dim=1)


def compute_weighted_metric_reward(scores, weights=None):
    if weights is None:
        return torch.mean(scores, dim=1)
    weight_sum = torch.sum(weights, dim=1).clamp(min=1e-6)
    return torch.sum(weights * scores, dim=1) / weight_sum
