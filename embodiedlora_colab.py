"""Calibrated hardware-free EmbodiedLoRa Expected SARSA experiment.

Google Colab:
    %run embodiedlora_colab.py

The experiment separates:

1. Active energy per transmission cycle: the primary metric.
2. Total energy per hour: the lifetime metric.
3. PDR: constrained to a configurable minimum.

Energy is calculated from current, voltage, and duration:

    mAh = current_mA * duration_seconds / 3600

The code does not claim hardware accuracy. Replace the radio-current and
timing constants with measurements from the target RN2483/device for physical
results.
"""

import random
from collections import Counter, deque
from dataclasses import dataclass
from statistics import mean, stdev

import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


SF_VALUES = (7, 8, 9, 10, 11, 12)
POWER_VALUES = (1, 2, 3, 4, 5)
INTERVAL_VALUES = (30, 60, 120, 300)
POWER_DBM = {1: 14, 2: 12, 3: 10, 4: 8, 5: 6}
REQUIRED_SNR = {
    7: -7.5,
    8: -10.0,
    9: -12.5,
    10: -15.0,
    11: -17.5,
    12: -20.0,
}

# Approximate electrical model. These are replaceable assumptions.
VOLTAGE_V = 3.3
SENSOR_CURRENT_MA = 5.0
SENSOR_SECONDS = 0.20
PROCESSING_CURRENT_MA = 8.0
PROCESSING_SECONDS = 0.10
RX_CURRENT_MA = 12.0
RX_SECONDS = 0.20
SLEEP_CURRENT_MA = 0.005  # 5 uA deep sleep
INITIAL_BATTERY_MAH = 1000.0

# Approximate radio timing. Higher SF takes longer on air.
TX_CURRENT_MA = {1: 38.0, 2: 34.0, 3: 30.0, 4: 26.0, 5: 22.0}
TX_SECONDS = {
    sf: 0.08 * (2 ** (sf - 7))
    for sf in SF_VALUES
}

PDR_WINDOW = 8
PDR_TARGET = 0.80
ACTIVE_ENERGY_WEIGHT = 0.55
PDR_WEIGHT = 0.30
LIFETIME_ENERGY_WEIGHT = 0.20
FAILURE_PENALTY = 1.20
RELIABILITY_PENALTY = 4.00
CONGESTION_RELIEF_WEIGHT = 0.12

TRAINING_EPISODES = 1200
TRAINING_STEPS = 60
TRAINING_SEEDS = (7, 17, 27)
EVALUATION_SEEDS = tuple(range(1000, 1050))
EVALUATION_STEPS = 500
VALIDATION_SEEDS = tuple(range(900, 910))
VALIDATION_STEPS = 200


def energy_mwh(current_ma, seconds):
    """Return electrical energy in mWh."""
    return VOLTAGE_V * current_ma * seconds / 3600.0


def energy_mah(current_ma, seconds):
    """Return battery charge consumption in mAh."""
    return current_ma * seconds / 3600.0


def mean_ci(values):
    """Return mean, standard deviation, and approximate 95% CI."""
    average = mean(values)
    deviation = stdev(values) if len(values) > 1 else 0.0
    margin = 1.96 * deviation / (len(values) ** 0.5)
    return average, deviation, margin


@dataclass(frozen=True)
class Observation:
    battery: float
    rssi: float
    snr: float
    pdr: float
    cbr: float
    recent_loss: float
    elapsed_hours: float


@dataclass(frozen=True)
class Result:
    delivered: bool
    rssi: float
    snr: float
    cbr: float
    own_channel_load: float
    active_energy_mah: float
    sleep_energy_mah: float
    active_energy_mwh: float

    @property
    def total_energy_mah(self):
        return self.active_energy_mah + self.sleep_energy_mah


class EmbodiedLoRaEnvironment:
    """Stochastic channel and battery model without physical hardware."""

    def __init__(self, seed=7):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.battery_mah = INITIAL_BATTERY_MAH
        self.elapsed_seconds = 0.0
        self.base_snr = -8.0
        self.cbr = 0.20
        self.step_number = 0
        self.deliveries = deque(maxlen=PDR_WINDOW)
        self.rssis = deque(maxlen=PDR_WINDOW)
        self.snrs = deque(maxlen=PDR_WINDOW)
        return self.observe()

    def observe(self):
        return Observation(
            battery=self.battery_mah / INITIAL_BATTERY_MAH,
            rssi=mean(self.rssis) if self.rssis else -100.0,
            snr=mean(self.snrs) if self.snrs else self.base_snr,
            pdr=mean(self.deliveries) if self.deliveries else 1.0,
            cbr=self.cbr,
            recent_loss=(
                1.0 - mean(self.deliveries)
                if self.deliveries
                else 0.0
            ),
            elapsed_hours=self.elapsed_seconds / 3600.0,
        )

    def transmit(self, action):
        sf, power, interval = action
        if sf not in SF_VALUES:
            raise ValueError(f"Unsupported SF: {sf}")
        if power not in POWER_VALUES:
            raise ValueError(f"Unsupported power index: {power}")
        if interval not in INTERVAL_VALUES:
            raise ValueError(f"Unsupported interval: {interval}")

        self.step_number += 1
        if self.step_number % 25 == 0:
            self.base_snr = self.rng.choice((-15.0, -11.0, -7.0))

        external_cbr = self.rng.uniform(0.10, 0.75)
        own_channel_load = min(0.25, 0.12 * 60.0 / interval)
        self.cbr = max(
            0.05,
            min(
                0.95,
                0.35 * self.cbr
                + 0.65 * external_cbr
                + own_channel_load,
            ),
        )

        snr = (
            self.base_snr
            + POWER_DBM[power]
            - 14.0
            - 3.0 * self.cbr
            + self.rng.uniform(-2.0, 2.0)
        )
        rssi = -100.0 + snr + self.rng.uniform(-3.0, 3.0)
        delivered = snr >= REQUIRED_SNR[sf]

        active_components = (
            energy_mah(SENSOR_CURRENT_MA, SENSOR_SECONDS),
            energy_mah(PROCESSING_CURRENT_MA, PROCESSING_SECONDS),
            energy_mah(TX_CURRENT_MA[power], TX_SECONDS[sf]),
            energy_mah(RX_CURRENT_MA, RX_SECONDS if delivered else 0.05),
            energy_mah(3.0 + 8.0 * self.cbr, 0.15),
        )
        active_energy_mah = sum(active_components)
        sleep_energy_mah = energy_mah(SLEEP_CURRENT_MA, interval)
        active_energy_mwh = active_energy_mah * VOLTAGE_V

        self.battery_mah = max(
            0.0,
            self.battery_mah - active_energy_mah - sleep_energy_mah,
        )
        self.elapsed_seconds += interval
        self.deliveries.append(int(delivered))
        self.rssis.append(rssi)
        self.snrs.append(snr)

        return Result(
            delivered=delivered,
            rssi=rssi,
            snr=snr,
            cbr=self.cbr,
            own_channel_load=own_channel_load,
            active_energy_mah=active_energy_mah,
            sleep_energy_mah=sleep_energy_mah,
            active_energy_mwh=active_energy_mwh,
        )


class ExpectedSARSA:
    """Tabular Expected SARSA with a compact state representation."""

    def __init__(
        self,
        seed=7,
        learning_rate=0.20,
        discount=0.90,
        epsilon=0.25,
        minimum_epsilon=0.03,
        epsilon_decay=0.997,
    ):
        self.rng = random.Random(seed)
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.minimum_epsilon = minimum_epsilon
        self.epsilon_decay = epsilon_decay
        self.actions = [
            (sf, power, interval)
            for sf in SF_VALUES
            for power in POWER_VALUES
            for interval in INTERVAL_VALUES
        ]
        self.q = {}
        self.update_count = 0

    def state(self, observation):
        """Use fewer bins and recent loss instead of redundant action fields."""
        return (
            min(4, max(0, int(observation.battery * 5))),
            min(3, max(0, int((observation.rssi + 120) / 15))),
            min(3, max(0, int((observation.snr + 24) / 6))),
            min(4, max(0, int(observation.pdr * 5))),
            min(4, max(0, int(observation.cbr * 5))),
            min(4, max(0, int(observation.recent_loss * 5))),
            int(observation.elapsed_hours) % 4,
        )

    def values(self, state):
        return self.q.setdefault(
            state,
            {action: 0.0 for action in self.actions},
        )

    def probabilities(self, state):
        values = self.values(state)
        best = max(values.values())
        best_actions = [
            action for action, value in values.items()
            if value == best
        ]
        probabilities = {
            action: self.epsilon / len(self.actions)
            for action in self.actions
        }
        exploitation = (1.0 - self.epsilon) / len(best_actions)
        for action in best_actions:
            probabilities[action] += exploitation
        return probabilities

    def choose(self, state, explore=True):
        values = self.values(state)
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        best = max(values.values())
        choices = [
            action for action, value in values.items()
            if value == best
        ]
        return self.rng.choice(choices)

    def reward(self, observation, result, interval):
        """Reliability-constrained reward with interval-aware lifetime cost."""
        active_scale = max(
            energy_mah(50.0, 1.0),
            energy_mah(50.0, 2.0),
        )
        active_score = min(
            1.0,
            result.active_energy_mah / active_scale,
        )
        total_per_hour = (
            result.total_energy_mah * 3600.0 / interval
        )
        lifetime_score = min(
            1.0,
            total_per_hour / energy_mah(2.0, 3600.0),
        )
        reliability_deficit = max(0.0, PDR_TARGET - observation.pdr)
        congestion_relief = max(
            0.0,
            0.12 - result.own_channel_load,
        )
        return (
            ACTIVE_ENERGY_WEIGHT * (1.0 - active_score)
            + LIFETIME_ENERGY_WEIGHT * (1.0 - lifetime_score)
            + PDR_WEIGHT * observation.pdr
            + CONGESTION_RELIEF_WEIGHT * congestion_relief
            + (0.50 if result.delivered else -FAILURE_PENALTY)
            - RELIABILITY_PENALTY * reliability_deficit
        )

    def update(self, state, action, reward, next_state):
        current = self.values(state)[action]
        next_values = self.values(next_state)
        probabilities = self.probabilities(next_state)
        expected_next = sum(
            probabilities[action] * value
            for action, value in next_values.items()
        )
        target = reward + self.gamma * expected_next
        self.values(state)[action] = current + self.alpha * (
            target - current
        )
        self.update_count += 1

    def end_episode(self):
        self.epsilon = max(
            self.minimum_epsilon,
            self.epsilon * self.epsilon_decay,
        )

    def q_statistics(self):
        values = [
            value
            for action_values in self.q.values()
            for value in action_values.values()
        ]
        if not values:
            return 0.0, 0.0
        return mean(abs(value) for value in values), max(
            abs(value) for value in values
        )


def run_episode(agent, environment, steps, explore=True, collect_trace=False):
    observation = environment.reset()
    total_reward = 0.0
    delivered = 0
    active = 0.0
    sleep = 0.0
    actions = []
    trace = []

    for step in range(1, steps + 1):
        previous_observation = observation
        state = agent.state(observation)
        action = agent.choose(state, explore=explore)
        result = environment.transmit(action)
        next_observation = environment.observe()
        next_state = agent.state(next_observation)
        reward = agent.reward(next_observation, result, action[2])

        if explore:
            agent.update(state, action, reward, next_state)

        observation = next_observation
        total_reward += reward
        delivered += int(result.delivered)
        active += result.active_energy_mah
        sleep += result.sleep_energy_mah
        actions.append(action)
        if collect_trace:
            trace.append(
                {
                    "step": step,
                    "time_hours": environment.elapsed_seconds / 3600.0,
                    "observation": previous_observation,
                    "next_observation": observation,
                    "action": action,
                    "result": result,
                }
            )

    output = {
        "reward": total_reward,
        "pdr": delivered / steps,
        "active_energy_mah": active / steps,
        "sleep_energy_mah": sleep / steps,
        "total_energy_mah": (active + sleep) / steps,
        "battery": observation.battery,
        "actions": actions,
        "total_energy_per_hour": (
            active + sleep
        ) / max(environment.elapsed_seconds / 3600.0, 1e-12),
    }
    if collect_trace:
        output["trace"] = trace
    return output


def train(episodes=TRAINING_EPISODES, steps=TRAINING_STEPS, seed=7):
    environment = EmbodiedLoRaEnvironment(seed)
    agent = ExpectedSARSA(seed)
    history = []

    for episode in range(episodes):
        result = run_episode(agent, environment, steps, explore=True)
        agent.end_episode()
        mean_abs_q, max_abs_q = agent.q_statistics()
        result["episode"] = episode + 1
        result["epsilon"] = agent.epsilon
        result["states"] = len(agent.q)
        result["mean_abs_q"] = mean_abs_q
        result["max_abs_q"] = max_abs_q
        history.append(result)

    return agent, history


def evaluate_policy(
    agent,
    seeds,
    steps=EVALUATION_STEPS,
    representative_seed=None,
):
    results = []
    for seed in seeds:
        environment = EmbodiedLoRaEnvironment(seed)
        result = run_episode(
            agent,
            environment,
            steps,
            explore=False,
            collect_trace=seed == representative_seed,
        )
        results.append(result)
    return results


def fixed_policy(action, seeds, steps):
    results = []
    for seed in seeds:
        environment = EmbodiedLoRaEnvironment(seed)
        result = run_fixed_episode(environment, action, steps)
        results.append(result)
    return results


def random_policy(seed):
    rng = random.Random(seed)
    return rng.choice(
        [
            (sf, power, interval)
            for sf in SF_VALUES
            for power in POWER_VALUES
            for interval in INTERVAL_VALUES
        ]
    )


def run_fixed_episode(environment, action, steps):
    observation = environment.reset()
    delivered = 0
    active = 0.0
    sleep = 0.0

    for _ in range(steps):
        result = environment.transmit(action)
        observation = environment.observe()
        delivered += int(result.delivered)
        active += result.active_energy_mah
        sleep += result.sleep_energy_mah

    return {
        "pdr": delivered / steps,
        "active_energy_mah": active / steps,
        "sleep_energy_mah": sleep / steps,
        "total_energy_mah": (active + sleep) / steps,
        "battery": observation.battery,
        "actions": [action] * steps,
        "total_energy_per_hour": (
            active + sleep
        ) / max(environment.elapsed_seconds / 3600.0, 1e-12),
    }


def aggregate(results):
    names = (
        "pdr",
        "active_energy_mah",
        "sleep_energy_mah",
        "total_energy_mah",
        "battery",
        "total_energy_per_hour",
    )
    output = {}
    for name in names:
        values = [row[name] for row in results]
        output[name] = mean_ci(values)
    return output


def select_best_fixed_config():
    """Select the best fixed action using validation data only."""
    candidates = [
        (sf, power, interval)
        for sf in SF_VALUES
        for power in POWER_VALUES
        for interval in INTERVAL_VALUES
    ]
    scored = []
    for action in candidates:
        results = fixed_policy(
            action,
            VALIDATION_SEEDS,
            VALIDATION_STEPS,
        )
        summary = aggregate(results)
        pdr = summary["pdr"][0]
        energy = summary["active_energy_mah"][0]
        if pdr >= PDR_TARGET:
            scored.append((energy, -pdr, action))

    if scored:
        return min(scored)[2]

    # If no candidate meets the target, choose the most reliable one.
    fallback = []
    for action in candidates:
        results = fixed_policy(
            action,
            VALIDATION_SEEDS,
            VALIDATION_STEPS,
        )
        summary = aggregate(results)
        fallback.append((
            -summary["pdr"][0],
            summary["active_energy_mah"][0],
            action,
        ))
    return min(fallback)[2]


def greedy_low_energy_policy(seeds, steps):
    action = min(
        (
            (sf, power, interval)
            for sf in SF_VALUES
            for power in POWER_VALUES
            for interval in INTERVAL_VALUES
        ),
        key=lambda item: TX_CURRENT_MA[item[1]] * TX_SECONDS[item[0]],
    )
    return action, fixed_policy(action, seeds, steps)


def random_policy_results(seeds, steps):
    results = []
    for seed in seeds:
        rng = random.Random(seed)
        environment = EmbodiedLoRaEnvironment(seed)
        observation = environment.reset()
        delivered = 0
        active = 0.0
        sleep = 0.0
        actions = []

        for _ in range(steps):
            action = rng.choice(
                [
                    (sf, power, interval)
                    for sf in SF_VALUES
                    for power in POWER_VALUES
                    for interval in INTERVAL_VALUES
                ]
            )
            result = environment.transmit(action)
            observation = environment.observe()
            delivered += int(result.delivered)
            active += result.active_energy_mah
            sleep += result.sleep_energy_mah
            actions.append(action)

        results.append(
            {
                "pdr": delivered / steps,
                "active_energy_mah": active / steps,
                "sleep_energy_mah": sleep / steps,
                "total_energy_mah": (active + sleep) / steps,
                "battery": observation.battery,
                "actions": actions,
                "total_energy_per_hour": (
                    active + sleep
                ) / max(environment.elapsed_seconds / 3600.0, 1e-12),
            }
        )
    return results


def reliability_policy(seeds, steps):
    """Highest-reliability fixed configuration selected on validation data."""
    candidates = [
        (sf, power, interval)
        for sf in SF_VALUES
        for power in POWER_VALUES
        for interval in INTERVAL_VALUES
    ]
    best = None
    best_key = None
    for action in candidates:
        summary = aggregate(
            fixed_policy(action, VALIDATION_SEEDS, VALIDATION_STEPS)
        )
        key = (
            -summary["pdr"][0],
            summary["active_energy_mah"][0],
        )
        if best_key is None or key < best_key:
            best_key = key
            best = action
    return best, fixed_policy(best, seeds, steps)


def evaluate_continuous_lifetime(agent, duration_hours=24, seed=2007):
    environment = EmbodiedLoRaEnvironment(seed)
    observation = environment.reset()
    horizon = duration_hours * 3600.0
    trace = []
    delivered = 0
    active = 0.0
    sleep = 0.0

    while environment.elapsed_seconds < horizon:
        state = agent.state(observation)
        action = agent.choose(state, explore=False)
        if action[2] > horizon - environment.elapsed_seconds:
            break
        result = environment.transmit(action)
        observation = environment.observe()
        delivered += int(result.delivered)
        active += result.active_energy_mah
        sleep += result.sleep_energy_mah
        trace.append(
            {
                "time_hours": environment.elapsed_seconds / 3600.0,
                "observation": observation,
                "action": action,
                "result": result,
            }
        )
        if environment.battery_mah <= 0.0:
            break

    elapsed = environment.elapsed_seconds / 3600.0
    transmissions = len(trace)
    total = active + sleep
    return {
        "duration_hours": elapsed,
        "transmissions": transmissions,
        "pdr": delivered / max(1, transmissions),
        "active_energy_mah": active,
        "sleep_energy_mah": sleep,
        "total_energy_mah": total,
        "total_energy_per_hour": total / max(elapsed, 1e-12),
        "battery": environment.battery_mah,
        "trace": trace,
    }


def action_frequencies(actions):
    counts = Counter(actions)
    total = max(1, len(actions))
    return {
        action: count / total
        for action, count in counts.items()
    }


def print_summary(name, summary):
    pdr, pdr_sd, pdr_ci = summary["pdr"]
    active, active_sd, active_ci = summary["active_energy_mah"]
    total, total_sd, total_ci = summary["total_energy_mah"]
    hourly, hourly_sd, hourly_ci = summary["total_energy_per_hour"]
    print(
        f"{name}: "
        f"PDR={pdr:.3f} +/- {pdr_ci:.3f} "
        f"(SD {pdr_sd:.3f}), "
        f"active={active:.6f} mAh +/- {active_ci:.6f}, "
        f"total/cycle={total:.6f} mAh +/- {total_ci:.6f}, "
        f"total/hour={hourly:.6f} mAh +/- {hourly_ci:.6f}"
    )


def confidence_table(baseline_results):
    """Return policy-level 95% confidence intervals across evaluation seeds."""
    rows = []
    for name, results in baseline_results.items():
        pdr_mean, pdr_sd, pdr_ci = mean_ci(
            [row["pdr"] for row in results]
        )
        energy_mean, energy_sd, energy_ci = mean_ci(
            [row["active_energy_mah"] for row in results]
        )
        rows.append(
            {
                "Policy": name,
                "Seeds": len(results),
                "PDR mean": pdr_mean,
                "PDR 95% CI": pdr_ci,
                "PDR CI lower": pdr_mean - pdr_ci,
                "PDR CI upper": pdr_mean + pdr_ci,
                "Active energy mean (mAh/cycle)": energy_mean,
                "Energy 95% CI": energy_ci,
                "Energy CI lower": energy_mean - energy_ci,
                "Energy CI upper": energy_mean + energy_ci,
            }
        )
    return rows


def display_confidence_table(baseline_results):
    """Display the requested PDR and energy confidence table."""
    rows = confidence_table(baseline_results)
    columns = (
        "Policy",
        "Seeds",
        "PDR mean",
        "PDR 95% CI",
        "Active energy mean (mAh/cycle)",
        "Energy 95% CI",
    )
    try:
        import pandas as pd
        from IPython.display import display

        frame = pd.DataFrame(rows)[list(columns)].copy()
        for column in columns[2:]:
            frame[column] = frame[column].map(
                lambda value: f"{value:.6f}"
            )
        display(frame)
    except ImportError:
        print("\n95% confidence intervals across evaluation seeds")
        print("-" * 110)
        print(
            f"{'Policy':35} {'Seeds':>5} {'PDR':>10} "
            f"{'PDR 95% CI':>19} {'Energy mAh/cycle':>18} "
            f"{'Energy 95% CI':>19}"
        )
        for row in rows:
            print(
                f"{row['Policy'][:35]:35} "
                f"{row['Seeds']:5d} "
                f"{row['PDR mean']:10.4f} "
                f"[{row['PDR CI lower']:.4f}, "
                f"{row['PDR CI upper']:.4f}] "
                f"{row['Active energy mean (mAh/cycle)']:18.7f} "
                f"[{row['Energy CI lower']:.7f}, "
                f"{row['Energy CI upper']:.7f}]"
            )


def print_report(
    agent,
    histories,
    learned_results,
    baseline_results,
    lifetime,
):
    print("\nCalibrated EmbodiedLoRa Expected SARSA report")
    print("==============================================")
    print(f"Training runs: {len(histories)}")
    print(f"Evaluation seeds: {len(EVALUATION_SEEDS)}")
    print(f"Evaluation transmissions/seed: {EVALUATION_STEPS}")
    print(f"Learned states, final run: {len(agent.q)}")
    print(f"PDR target: {PDR_TARGET:.2f}")

    final_rewards = []
    final_pdr = []
    final_energy = []
    for history in histories:
        window = history[-100:]
        final_rewards.extend(row["reward"] for row in window)
        final_pdr.extend(row["pdr"] for row in window)
        final_energy.extend(row["active_energy_mah"] for row in window)

    reward_mean, reward_sd, reward_ci = mean_ci(final_rewards)
    pdr_mean, pdr_sd, pdr_ci = mean_ci(final_pdr)
    energy_mean, energy_sd, energy_ci = mean_ci(final_energy)
    print(
        f"Final-100 reward: {reward_mean:.3f} +/- {reward_ci:.3f}"
    )
    print(f"Final-100 PDR: {pdr_mean:.3f} +/- {pdr_ci:.3f}")
    print(
        f"Final-100 active energy/cycle: "
        f"{energy_mean:.6f} +/- {energy_ci:.6f} mAh"
    )

    print("\nIndependent evaluation:")
    for name, results in baseline_results.items():
        print_summary(name, aggregate(results))

    learned_summary = aggregate(learned_results)
    best_name = next(
        name for name in baseline_results
        if name.startswith("Best validated fixed")
    )
    best_summary = aggregate(baseline_results[best_name])
    learned_pdr = learned_summary["pdr"]
    learned_active = learned_summary["active_energy_mah"]
    learned_hourly = learned_summary["total_energy_per_hour"]
    best_active = best_summary["active_energy_mah"]
    best_hourly = best_summary["total_energy_per_hour"]
    pdr_pass = learned_pdr[0] >= PDR_TARGET
    print("\nIndependent-test checks:")
    print(
        f"  PDR target >= {PDR_TARGET:.2f}: "
        f"{'PASS' if pdr_pass else 'FAIL'} "
        f"(mean={learned_pdr[0]:.3f}, "
        f"95% CI=[{learned_pdr[0] - learned_pdr[2]:.3f}, "
        f"{learned_pdr[0] + learned_pdr[2]:.3f}])"
    )
    print(
        f"  Active energy/cycle: {learned_active[0]:.6f} mAh "
        f"(95% CI +/- {learned_active[2]:.6f})"
    )
    print(
        f"  Total energy/hour: {learned_hourly[0]:.6f} mAh "
        f"(95% CI +/- {learned_hourly[2]:.6f})"
    )
    print(
        f"  Compared with best validated fixed: "
        f"active energy {((learned_active[0] / best_active[0]) - 1) * 100:+.1f}%, "
        f"total/hour {((learned_hourly[0] / best_hourly[0]) - 1) * 100:+.1f}%"
    )

    print("\nContinuous lifetime:")
    print(f"Duration: {lifetime['duration_hours']:.2f} hours")
    print(f"Transmissions: {lifetime['transmissions']}")
    print(f"PDR: {lifetime['pdr']:.3f}")
    print(
        f"Total energy/hour: "
        f"{lifetime['total_energy_per_hour']:.6f} mAh/hour"
    )
    print(f"Remaining battery: {lifetime['battery']:.2f} mAh")


def plot_baseline_comparison(baseline_results):
    """Compare PDR and active energy for all policies with 95% CI bars."""
    rows = confidence_table(baseline_results)
    labels = [row["Policy"] for row in rows]
    pdr = [row["PDR mean"] for row in rows]
    pdr_ci = [row["PDR 95% CI"] for row in rows]
    energy = [
        row["Active energy mean (mAh/cycle)"]
        for row in rows
    ]
    energy_ci = [row["Energy 95% CI"] for row in rows]

    if PLOTLY_AVAILABLE:
        figure = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "PDR across evaluation seeds",
                "Active energy per cycle",
            ),
        )
        figure.add_trace(
            go.Bar(
                x=labels,
                y=pdr,
                error_y=dict(type="data", array=pdr_ci),
                customdata=[
                    [row["PDR CI lower"], row["PDR CI upper"]]
                    for row in rows
                ],
                hovertemplate=(
                    "%{x}<br>Mean PDR: %{y:.4f}"
                    "<br>95% CI: [%{customdata[0]:.4f}, "
                    "%{customdata[1]:.4f}]<extra></extra>"
                ),
                name="PDR",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Bar(
                x=labels,
                y=energy,
                error_y=dict(type="data", array=energy_ci),
                customdata=[
                    [row["Energy CI lower"], row["Energy CI upper"]]
                    for row in rows
                ],
                hovertemplate=(
                    "%{x}<br>Mean energy: %{y:.7f} mAh/cycle"
                    "<br>95% CI: [%{customdata[0]:.7f}, "
                    "%{customdata[1]:.7f}]<extra></extra>"
                ),
                name="Active energy",
            ),
            row=1,
            col=2,
        )
        figure.add_hline(
            y=PDR_TARGET,
            line_dash="dash",
            annotation_text=f"PDR target {PDR_TARGET:.2f}",
            row=1,
            col=1,
        )
        figure.update_layout(
            title="Baseline comparison with 95% confidence intervals",
            template="plotly_white",
            height=600,
            showlegend=False,
        )
        figure.update_xaxes(tickangle=-35)
        figure.update_yaxes(title_text="PDR", range=[0, 1.05], col=1)
        figure.update_yaxes(
            title_text="Active energy (mAh/cycle)",
            col=2,
        )
        figure.show()
        return

    positions = range(len(labels))
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.bar(positions, pdr, yerr=pdr_ci, capsize=5)
    plt.axhline(PDR_TARGET, linestyle="--", color="red")
    plt.xticks(positions, labels, rotation=35, ha="right")
    plt.ylabel("PDR")
    plt.ylim(0, 1.05)
    plt.title("PDR comparison")
    plt.grid(axis="y")
    plt.subplot(1, 2, 2)
    plt.bar(positions, energy, yerr=energy_ci, capsize=5)
    plt.xticks(positions, labels, rotation=35, ha="right")
    plt.ylabel("Active energy (mAh/cycle)")
    plt.title("Energy comparison")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


def plot_training(histories):
    plt.figure(figsize=(16, 10))
    labels = (
        ("reward", "Training reward"),
        ("pdr", "Training PDR"),
        ("active_energy_mah", "Active energy per cycle (mAh)"),
    )

    for index, (key, title) in enumerate(labels, start=1):
        plt.subplot(2, 2, index)
        for run, history in enumerate(histories, start=1):
            values = [row[key] for row in history]
            window = 100
            if len(values) >= window:
                average = [
                    mean(values[i - window + 1:i + 1])
                    for i in range(window - 1, len(values))
                ]
                plt.plot(
                    range(window, len(values) + 1),
                    average,
                    label=f"Run {run}",
                )
            else:
                plt.plot(values, label=f"Run {run}")
        plt.title(title)
        plt.xlabel("Episode")
        plt.legend()
        plt.grid()

    plt.subplot(2, 2, 4)
    for run, history in enumerate(histories, start=1):
        q_values = [row["mean_abs_q"] for row in history]
        plt.plot(q_values, label=f"Run {run}")
    plt.title("Mean absolute Q-value")
    plt.xlabel("Episode")
    plt.ylabel("|Q|")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_action_frequencies(agent, results):
    actions = []
    for result in results:
        actions.extend(result["actions"])
    frequencies = action_frequencies(actions)
    top = sorted(
        frequencies.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:20]
    labels = [
        f"SF{action[0]}/P{POWER_DBM[action[1]]}/{action[2]}s"
        for action, _ in top
    ]
    values = [frequency for _, frequency in top]

    plt.figure(figsize=(12, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=70)
    plt.ylabel("Selection frequency")
    plt.title("Most frequently selected learned actions")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


def plot_evaluation_analysis(evaluation):
    """Plot the six requested diagnostics for one held-out trace."""
    trace = evaluation.get("trace", [])
    if not trace:
        print("No representative evaluation trace is available.")
        return

    steps = [row["step"] for row in trace]
    energy = [row["result"].active_energy_mah for row in trace]
    snr = [row["result"].snr for row in trace]
    sf = [row["action"][0] for row in trace]
    power = [POWER_DBM[row["action"][1]] for row in trace]
    intervals = [row["action"][2] for row in trace]
    delivered = [int(row["result"].delivered) for row in trace]

    def average(values, window=25):
        if len(values) < window:
            return values
        return [
            mean(values[index - window + 1:index + 1])
            for index in range(window - 1, len(values))
        ]

    energy_average = average(energy)
    offset = 24 if len(energy) >= 25 else 0

    plt.figure(figsize=(16, 11))

    plt.subplot(3, 2, 1)
    plt.plot(steps, energy, alpha=0.35, label="Per-cycle energy")
    plt.plot(
        steps[offset:],
        energy_average,
        linewidth=2,
        label="25-cycle average",
    )
    plt.title("Average energy per cycle")
    plt.xlabel("Transmission")
    plt.ylabel("Active energy (mAh)")
    plt.legend()
    plt.grid()

    plt.subplot(3, 2, 2)
    plt.plot(steps, snr)
    plt.title("Evaluation SNR")
    plt.xlabel("Transmission")
    plt.ylabel("SNR (dB)")
    plt.grid()

    plt.subplot(3, 2, 3)
    plt.step(steps, sf, where="mid")
    plt.title("Evaluation spreading factor")
    plt.xlabel("Transmission")
    plt.ylabel("SF")
    plt.yticks(SF_VALUES)
    plt.grid()

    plt.subplot(3, 2, 4)
    plt.step(steps, power, where="mid")
    plt.title("Evaluation transmission power")
    plt.xlabel("Transmission")
    plt.ylabel("Power (dBm)")
    plt.yticks(sorted(POWER_DBM.values()))
    plt.grid()

    plt.subplot(3, 2, 5)
    plt.step(steps, intervals, where="mid")
    plt.title("Evaluation transmission interval")
    plt.xlabel("Transmission")
    plt.ylabel("Interval (seconds)")
    plt.yticks(INTERVAL_VALUES)
    plt.grid()

    plt.subplot(3, 2, 6)
    plt.bar(steps, delivered, width=1.0)
    plt.title("Evaluation delivery")
    plt.xlabel("Transmission")
    plt.ylabel("Delivered")
    plt.yticks([0, 1])
    plt.grid()

    plt.tight_layout()
    plt.show()


def plot_interactive_evaluation(evaluation):
    """Interactive six-panel dashboard for one held-out evaluation seed."""
    trace = evaluation.get("trace", [])
    if not trace:
        print("No representative evaluation trace is available.")
        return

    steps = [row["step"] for row in trace]
    energy = [row["result"].active_energy_mah for row in trace]
    snr = [row["result"].snr for row in trace]
    sf = [row["action"][0] for row in trace]
    power = [POWER_DBM[row["action"][1]] for row in trace]
    intervals = [row["action"][2] for row in trace]
    delivered = [int(row["result"].delivered) for row in trace]
    cbr = [row["result"].cbr for row in trace]

    if not PLOTLY_AVAILABLE:
        print(
            "Plotly is not installed; displaying the static dashboard "
            "instead."
        )
        plot_evaluation_analysis(evaluation)
        return

    figure = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Active energy per cycle",
            "Evaluation SNR and CBR",
            "Spreading factor",
            "Transmission power",
            "Transmission interval",
            "Packet delivery",
        ),
        vertical_spacing=0.10,
    )
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=energy,
            mode="lines",
            name="Active energy",
            hovertemplate="Step %{x}<br>%{y:.7f} mAh<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=snr,
            mode="lines+markers",
            name="SNR (dB)",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=cbr,
            mode="lines",
            name="CBR",
            yaxis="y2",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=sf,
            mode="lines",
            name="SF",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=power,
            mode="lines",
            name="Power (dBm)",
        ),
        row=2,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=intervals,
            mode="lines",
            name="Interval (s)",
        ),
        row=3,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=steps,
            y=delivered,
            name="Delivered",
            hovertemplate="Step %{x}<br>%{y}<extra></extra>",
        ),
        row=3,
        col=2,
    )
    figure.update_yaxes(title_text="mAh/cycle", row=1, col=1)
    figure.update_yaxes(title_text="dB", row=1, col=2)
    figure.update_yaxes(title_text="SF", tickvals=SF_VALUES, row=2, col=1)
    figure.update_yaxes(
        title_text="dBm",
        tickvals=sorted(POWER_DBM.values()),
        row=2,
        col=2,
    )
    figure.update_yaxes(
        title_text="Seconds",
        tickvals=INTERVAL_VALUES,
        row=3,
        col=1,
    )
    figure.update_yaxes(title_text="Delivered", range=[0, 1.05], row=3, col=2)
    figure.update_xaxes(title_text="Transmission")
    figure.update_layout(
        title="Interactive held-out evaluation dashboard",
        template="plotly_white",
        height=950,
        hovermode="x unified",
    )
    figure.show()


def plot_lifetime(lifetime):
    if not lifetime["trace"]:
        return
    times = [row["time_hours"] for row in lifetime["trace"]]
    battery = [
        row["observation"].battery * INITIAL_BATTERY_MAH
        for row in lifetime["trace"]
    ]
    cumulative = []
    total = 0.0
    for row in lifetime["trace"]:
        total += row["result"].total_energy_mah
        cumulative.append(total)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.step(times, battery, where="post")
    plt.title("Continuous battery")
    plt.xlabel("Time (hours)")
    plt.ylabel("Battery (mAh)")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.step(times, cumulative, where="post")
    plt.title("Cumulative lifetime energy")
    plt.xlabel("Time (hours)")
    plt.ylabel("Energy (mAh)")
    plt.grid()
    plt.tight_layout()
    plt.show()


def main():
    histories = []
    agents = []
    for seed in TRAINING_SEEDS:
        agent, history = train(seed=seed)
        agents.append(agent)
        histories.append(history)

    agent = agents[-1]
    learned_results = evaluate_policy(
        agent,
        EVALUATION_SEEDS,
        EVALUATION_STEPS,
        representative_seed=EVALUATION_SEEDS[0],
    )

    best_fixed = select_best_fixed_config()
    reliable_action, reliable_results = reliability_policy(
        EVALUATION_SEEDS,
        EVALUATION_STEPS,
    )
    greedy_action, greedy_results = greedy_low_energy_policy(
        EVALUATION_SEEDS,
        EVALUATION_STEPS,
    )
    random_results = random_policy_results(
        EVALUATION_SEEDS,
        EVALUATION_STEPS,
    )
    best_fixed_results = fixed_policy(
        best_fixed,
        EVALUATION_SEEDS,
        EVALUATION_STEPS,
    )

    baseline_results = {
        "Learned Expected SARSA": learned_results,
        f"Best validated fixed {best_fixed}": best_fixed_results,
        f"Reliability fixed {reliable_action}": reliable_results,
        f"Greedy lowest-energy {greedy_action}": greedy_results,
        "Random policy": random_results,
    }

    lifetime = evaluate_continuous_lifetime(
        agent,
        duration_hours=24,
        seed=2007,
    )

    print_report(
        agent,
        histories,
        learned_results,
        baseline_results,
        lifetime,
    )
    display_confidence_table(baseline_results)

    print("\nValidation-selected configurations:")
    print(f"Best fixed constrained action: {best_fixed}")
    print(f"Reliability-oriented action: {reliable_action}")
    print(f"Greedy lowest-energy action: {greedy_action}")

    plot_training(histories)
    plot_action_frequencies(agent, learned_results)
    plot_baseline_comparison(baseline_results)
    plot_evaluation_analysis(learned_results[0])
    plot_interactive_evaluation(learned_results[0])
    plot_lifetime(lifetime)


if __name__ == "__main__":
    main()
