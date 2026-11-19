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
import os
import math
import csv
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

PLOT_DPI = 300
PLOT_DIRECTORY = "embodiedlora_plots"
os.makedirs(PLOT_DIRECTORY, exist_ok=True)


def save_matplotlib_figure(filename):
    """Save and display a high-resolution static figure."""
    path = os.path.join(PLOT_DIRECTORY, filename)
    plt.gcf().savefig(
        path,
        dpi=PLOT_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"Saved plot: {path}")


def save_plotly_figure(figure, filename):
    """Save an interactive HTML plot and, if supported, a 300-DPI PNG."""
    html_path = os.path.join(PLOT_DIRECTORY, filename + ".html")
    figure.write_html(html_path, include_plotlyjs="cdn")
    print(f"Saved interactive plot: {html_path}")
    try:
        png_path = os.path.join(PLOT_DIRECTORY, filename + ".png")
        figure.write_image(png_path, scale=300 / 96)
        print(f"Saved plot at 300-DPI-equivalent resolution: {png_path}")
    except Exception as error:
        print(
            "PNG export requires Plotly image export support "
            "(for example, Kaleido). Interactive HTML was saved."
        )


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
ADR_INTERVAL = 60
ADR_MARGIN_DB = 10.0
NO_ADR_ACTION = (12, 1, 60)
PAYLOAD_BITS = 64
LIFETIME_SEEDS = tuple(range(3000, 3010))
EXTENDED_STUDY_SEEDS = tuple(range(7, 17))
EXTENDED_EVALUATION_SEEDS = tuple(range(1100, 1150))
EXTENDED_EPISODES = 600
EXTENDED_STEPS = 60
RUN_EXTENDED_STUDIES = True
MEASUREMENT_CSV = os.environ.get("LORAWAN_MEASUREMENTS_CSV")
PAYLOAD_BYTES = PAYLOAD_BITS // 8
MAX_RETRANSMISSIONS = 2
DUTY_CYCLE_LIMIT = 0.01  # 1% sub-band duty-cycle constraint
PATH_LOSS_DISTANCE_M = 250.0
PATH_LOSS_EXPONENT = 2.7
REFERENCE_DISTANCE_M = 100.0
REFERENCE_SNR_DB = -8.0
INTERFERENCE_STD_DB = 2.0
RADIO_CALIBRATION = {
    "source": (
        "Microchip RN2483 Data Sheet DS50002346B, Table 2-1; "
        "replace assumptions with bench measurements"
    ),
    "url": (
        "https://www.microchip.com/content/dam/mchp/documents/"
        "OTH/ProductDocuments/DataSheets/RN2483-Data-Sheet-DS50002346E.pdf"
    ),
    "voltage_v": VOLTAGE_V,
    "tx_current_ma": TX_CURRENT_MA,
    "rx_current_ma": RX_CURRENT_MA,
    "sleep_current_ma": SLEEP_CURRENT_MA,
}


def energy_mwh(current_ma, seconds):
    """Return electrical energy in mWh."""
    return VOLTAGE_V * current_ma * seconds / 3600.0


def energy_mah(current_ma, seconds):
    """Return battery charge consumption in mAh."""
    return current_ma * seconds / 3600.0


def summarize_episode_samples(
    delivered,
    snrs,
    cbrs,
    active,
    sleep,
    airtime,
    timestamps,
    elapsed_seconds,
    retransmissions=None,
    duty_cycle_limited=None,
):
    """Create comparable per-seed metrics for every policy."""
    retransmissions = retransmissions or [0] * len(delivered)
    duty_cycle_limited = duty_cycle_limited or [False] * len(delivered)
    delivered_times = [
        elapsed
        for elapsed, was_delivered in zip(timestamps, delivered)
        if was_delivered
    ]
    snr_regions = {"weak": [], "medium": [], "strong": []}
    cbr_regions = {"low": [], "medium": [], "high": []}
    for was_delivered, snr, cbr in zip(delivered, snrs, cbrs):
        snr_key = (
            "weak" if snr < -18.0
            else "medium" if snr < -12.0
            else "strong"
        )
        cbr_key = (
            "low" if cbr < 0.33
            else "medium" if cbr < 0.66
            else "high"
        )
        snr_regions[snr_key].append(int(was_delivered))
        cbr_regions[cbr_key].append(int(was_delivered))
    delivery_gaps = [
        later - earlier
        for earlier, later in zip(
            delivered_times,
            delivered_times[1:],
        )
    ]
    return {
        "pdr": sum(delivered) / max(1, len(delivered)),
        "loss_rate": 1.0 - (
            sum(delivered) / max(1, len(delivered))
        ),
        "loss_count": len(delivered) - sum(delivered),
        "mean_snr": mean(snrs),
        "mean_cbr": mean(cbrs),
        "active_energy_mah": sum(active) / max(1, len(active)),
        "sleep_energy_mah": sum(sleep) / max(1, len(sleep)),
        "total_energy_mah": (
            sum(active) + sum(sleep)
        ) / max(1, len(active)),
        "total_energy_per_hour": (
            sum(active) + sum(sleep)
        ) / max(elapsed_seconds / 3600.0, 1e-12),
        "airtime_seconds_per_cycle": mean(airtime),
        "airtime_fraction": sum(airtime) / max(elapsed_seconds, 1e-12),
        "throughput_bps": (
            sum(delivered) * PAYLOAD_BITS
            / max(elapsed_seconds, 1e-12)
        ),
        "packet_jitter_seconds": (
            stdev(delivery_gaps)
            if len(delivery_gaps) > 1
            else 0.0
        ),
        "retransmissions": sum(retransmissions),
        "retransmission_rate": (
            sum(retransmissions) / max(1, len(retransmissions))
        ),
        "duty_cycle_limited_fraction": (
            sum(duty_cycle_limited)
            / max(1, len(duty_cycle_limited))
        ),
        "pdr_by_snr_region": {
            key: sum(values) / len(values) if values else float("nan")
            for key, values in snr_regions.items()
        },
        "pdr_by_cbr_region": {
            key: sum(values) / len(values) if values else float("nan")
            for key, values in cbr_regions.items()
        },
    }


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
    airtime_seconds: float
    retransmissions: int
    duty_cycle_limited: bool
    path_loss_db: float

    @property
    def total_energy_mah(self):
        return self.active_energy_mah + self.sleep_energy_mah


class EmbodiedLoRaEnvironment:
    """Stochastic channel and battery model without physical hardware."""

    def __init__(
        self,
        seed=7,
        distance_m=PATH_LOSS_DISTANCE_M,
        path_loss_exponent=PATH_LOSS_EXPONENT,
        max_retransmissions=MAX_RETRANSMISSIONS,
        duty_cycle_limit=DUTY_CYCLE_LIMIT,
        payload_bytes=PAYLOAD_BYTES,
        interference_std_db=INTERFERENCE_STD_DB,
        external_cbr_range=(0.10, 0.75),
        initial_battery_mah=INITIAL_BATTERY_MAH,
    ):
        self.rng = random.Random(seed)
        self.distance_m = distance_m
        self.path_loss_exponent = path_loss_exponent
        self.max_retransmissions = max_retransmissions
        self.duty_cycle_limit = duty_cycle_limit
        self.payload_bytes = payload_bytes
        self.interference_std_db = interference_std_db
        self.external_cbr_range = external_cbr_range
        self.initial_battery_mah = initial_battery_mah
        self.reset()

    def reset(self):
        self.battery_mah = self.initial_battery_mah
        self.elapsed_seconds = 0.0
        distance_loss = 10.0 * self.path_loss_exponent * math.log10(
            max(self.distance_m, 1.0) / REFERENCE_DISTANCE_M
        )
        self.base_snr = REFERENCE_SNR_DB - distance_loss
        self.cbr = 0.20
        self.step_number = 0
        self.deliveries = deque(maxlen=PDR_WINDOW)
        self.rssis = deque(maxlen=PDR_WINDOW)
        self.snrs = deque(maxlen=PDR_WINDOW)
        return self.observe()

    def observe(self):
        return Observation(
            battery=self.battery_mah / self.initial_battery_mah,
            rssi=mean(self.rssis) if self.rssis else -100.0,
            snr=mean(self.snrs) if self.snrs else self.base_snr,
            pdr=mean(self.deliveries) if self.deliveries else 1.0,
            cbr=self.cbr,
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

        external_cbr = self.rng.uniform(*self.external_cbr_range)
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
            + self.rng.gauss(0.0, self.interference_std_db)
        )
        rssi = -100.0 + snr + self.rng.uniform(-3.0, 3.0)
        airtime_seconds = (
            TX_SECONDS[sf] * self.payload_bytes / PAYLOAD_BYTES
        )
        effective_interval = max(
            interval,
            airtime_seconds / max(self.duty_cycle_limit, 1e-9),
        )
        duty_cycle_limited = effective_interval > interval
        retransmissions = 0
        delivered = False
        for attempt in range(self.max_retransmissions + 1):
            attempt_snr = snr + self.rng.gauss(0.0, 0.5)
            if attempt_snr >= REQUIRED_SNR[sf]:
                delivered = True
                retransmissions = attempt
                break
            retransmissions = attempt + 1

        active_components = (
            energy_mah(SENSOR_CURRENT_MA, SENSOR_SECONDS),
            energy_mah(PROCESSING_CURRENT_MA, PROCESSING_SECONDS),
            energy_mah(
                TX_CURRENT_MA[power],
                airtime_seconds * (retransmissions + 1),
            ),
            energy_mah(RX_CURRENT_MA, RX_SECONDS if delivered else 0.05),
            energy_mah(3.0 + 8.0 * self.cbr, 0.15),
        )
        active_energy_mah = sum(active_components)
        sleep_energy_mah = energy_mah(
            SLEEP_CURRENT_MA,
            effective_interval,
        )
        active_energy_mwh = active_energy_mah * VOLTAGE_V

        self.battery_mah = max(
            0.0,
            self.battery_mah - active_energy_mah - sleep_energy_mah,
        )
        self.elapsed_seconds += effective_interval
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
            airtime_seconds=airtime_seconds * (retransmissions + 1),
            retransmissions=retransmissions,
            duty_cycle_limited=duty_cycle_limited,
            path_loss_db=10.0 * self.path_loss_exponent * math.log10(
                max(self.distance_m, 1.0) / REFERENCE_DISTANCE_M
            ),
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
        state_features=("battery", "rssi", "snr", "pdr", "cbr"),
        actions=None,
        reward_weights=None,
    ):
        self.rng = random.Random(seed)
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.minimum_epsilon = minimum_epsilon
        self.epsilon_decay = epsilon_decay
        self.state_features = tuple(state_features)
        self.actions = actions or [
            (sf, power, interval)
            for sf in SF_VALUES
            for power in POWER_VALUES
            for interval in INTERVAL_VALUES
        ]
        self.reward_weights = reward_weights or {
            "active": ACTIVE_ENERGY_WEIGHT,
            "lifetime": LIFETIME_ENERGY_WEIGHT,
            "pdr": PDR_WEIGHT,
            "congestion": CONGESTION_RELIEF_WEIGHT,
        }
        self.q = {}
        self.update_count = 0

    def state(self, observation):
        """Discretize battery, RSSI, SNR, PDR, and CBR."""
        bins = {
            "battery": min(4, max(0, int(observation.battery * 5))),
            "rssi": min(3, max(0, int((observation.rssi + 120) / 15))),
            "snr": min(3, max(0, int((observation.snr + 24) / 6))),
            "pdr": min(4, max(0, int(observation.pdr * 5))),
            "cbr": min(4, max(0, int(observation.cbr * 5))),
        }
        return tuple(bins[name] for name in self.state_features)

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
            self.reward_weights["active"] * (1.0 - active_score)
            + self.reward_weights["lifetime"] * (1.0 - lifetime_score)
            + self.reward_weights["pdr"] * observation.pdr
            + self.reward_weights["congestion"] * congestion_relief
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
    delivered_samples = []
    snrs = []
    cbrs = []
    active_samples = []
    sleep_samples = []
    airtime = []
    timestamps = []
    retransmissions = []
    duty_cycle_limited = []
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
        delivered_samples.append(int(result.delivered))
        snrs.append(result.snr)
        cbrs.append(result.cbr)
        active_samples.append(result.active_energy_mah)
        sleep_samples.append(result.sleep_energy_mah)
        airtime.append(result.airtime_seconds)
        timestamps.append(environment.elapsed_seconds)
        retransmissions.append(result.retransmissions)
        duty_cycle_limited.append(result.duty_cycle_limited)
        retransmissions.append(result.retransmissions)
        duty_cycle_limited.append(result.duty_cycle_limited)
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
    output.update(
        summarize_episode_samples(
            delivered_samples,
            snrs,
            cbrs,
            active_samples,
            sleep_samples,
            airtime,
            timestamps,
            environment.elapsed_seconds,
            retransmissions,
            duty_cycle_limited,
        )
    )
    if collect_trace:
        output["trace"] = trace
    return output


def train(
    episodes=TRAINING_EPISODES,
    steps=TRAINING_STEPS,
    seed=7,
    environment_kwargs=None,
    agent_kwargs=None,
):
    environment = EmbodiedLoRaEnvironment(
        seed,
        **(environment_kwargs or {}),
    )
    agent = ExpectedSARSA(seed, **(agent_kwargs or {}))
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
    environment_kwargs=None,
):
    results = []
    for seed in seeds:
        environment = EmbodiedLoRaEnvironment(
            seed,
            **(environment_kwargs or {}),
        )
        result = run_episode(
            agent,
            environment,
            steps,
            explore=False,
            collect_trace=seed == representative_seed,
        )
        results.append(result)
    return results


def run_ablation_study():
    """Measure the contribution of state, reward, and action components."""
    configurations = {
        "Full model": {},
        "No battery state": {
            "agent_kwargs": {
                "state_features": ("rssi", "snr", "pdr", "cbr"),
            },
        },
        "No CBR state": {
            "agent_kwargs": {
                "state_features": ("battery", "rssi", "snr", "pdr"),
            },
        },
        "No lifetime reward": {
            "agent_kwargs": {
                "reward_weights": {
                    "active": ACTIVE_ENERGY_WEIGHT,
                    "lifetime": 0.0,
                    "pdr": PDR_WEIGHT,
                    "congestion": CONGESTION_RELIEF_WEIGHT,
                },
            },
        },
        "No reliability reward": {
            "agent_kwargs": {
                "reward_weights": {
                    "active": ACTIVE_ENERGY_WEIGHT,
                    "lifetime": LIFETIME_ENERGY_WEIGHT,
                    "pdr": 0.0,
                    "congestion": CONGESTION_RELIEF_WEIGHT,
                },
            },
        },
        "SF-only actions": {
            "agent_kwargs": {
                "actions": [
                    (sf, 1, 60)
                    for sf in SF_VALUES
                ],
            },
        },
        "Fixed SF12 actions": {
            "agent_kwargs": {
                "actions": [(12, power, 60) for power in POWER_VALUES],
            },
        },
    }
    output = {}
    for name, options in configurations.items():
        agents = [
            train(
                episodes=EXTENDED_EPISODES,
                steps=EXTENDED_STEPS,
                seed=seed,
                agent_kwargs=options.get("agent_kwargs"),
            )[0]
            for seed in EXTENDED_STUDY_SEEDS
        ]
        results = [
            evaluate_policy(
                agent,
                EXTENDED_EVALUATION_SEEDS,
                steps=500,
            )
            for agent in agents
        ]
        per_agent = []
        for agent_results in results:
            summary = aggregate(agent_results)
            per_agent.append({
                "pdr": summary["pdr"][0],
                "active_energy_mah": summary["active_energy_mah"][0],
            })
        pdr_values = [row["pdr"] for row in per_agent]
        energy_values = [row["active_energy_mah"] for row in per_agent]
        pdr_ci = mean_ci(pdr_values)
        energy_ci = mean_ci(energy_values)
        output[name] = {
            "pdr": pdr_ci,
            "active_energy_mah": energy_ci,
        }
    return output


def run_sensitivity_study():
    """Evaluate traffic, battery, payload, channel, target, and seed effects."""
    settings = {
        "low traffic": {
            "environment_kwargs": {
                "external_cbr_range": (0.05, 0.25),
                "interference_std_db": 1.0,
            },
        },
        "high traffic": {
            "environment_kwargs": {
                "external_cbr_range": (0.60, 0.95),
                "interference_std_db": 3.5,
            },
        },
        "small battery": {
            "environment_kwargs": {"initial_battery_mah": 250.0},
        },
        "large payload": {
            "environment_kwargs": {"payload_bytes": 24},
        },
        "strong channel": {
            "environment_kwargs": {"distance_m": 120.0},
        },
        "weak channel": {
            "environment_kwargs": {"distance_m": 450.0},
        },
    }
    output = {}
    original_payload = PAYLOAD_BITS
    original_target = PDR_TARGET
    try:
        for name, options in settings.items():
            if name == "large payload":
                globals()["PAYLOAD_BITS"] = 192
            else:
                globals()["PAYLOAD_BITS"] = original_payload
            agents = [
                train(
                    episodes=EXTENDED_EPISODES,
                    steps=EXTENDED_STEPS,
                    seed=seed,
                    environment_kwargs=options["environment_kwargs"],
                )[0]
                for seed in EXTENDED_STUDY_SEEDS
            ]
            per_agent = []
            for agent in agents:
                rows = evaluate_policy(
                    agent,
                    EXTENDED_EVALUATION_SEEDS,
                    steps=500,
                    environment_kwargs=options["environment_kwargs"],
                )
                summary = aggregate(rows)
                per_agent.append({
                    "pdr": summary["pdr"][0],
                    "active_energy_mah": summary["active_energy_mah"][0],
                })
            output[name] = {
                "pdr": mean_ci([row["pdr"] for row in per_agent]),
                "active_energy_mah": mean_ci([
                    row["active_energy_mah"] for row in per_agent
                ]),
            }
        for target in (0.70, 0.80, 0.90):
            globals()["PDR_TARGET"] = target
            per_agent = []
            for seed in EXTENDED_STUDY_SEEDS:
                agent = train(
                    episodes=EXTENDED_EPISODES,
                    steps=EXTENDED_STEPS,
                    seed=seed,
                )[0]
                summary = aggregate(
                    evaluate_policy(
                        agent,
                        EXTENDED_EVALUATION_SEEDS,
                        steps=500,
                    )
                )
                per_agent.append({
                    "pdr": summary["pdr"][0],
                    "active_energy_mah": summary["active_energy_mah"][0],
                })
            output[f"PDR target {target:.2f}"] = {
                "pdr": mean_ci([row["pdr"] for row in per_agent]),
                "active_energy_mah": mean_ci([
                    row["active_energy_mah"] for row in per_agent
                ]),
            }
    finally:
        globals()["PAYLOAD_BITS"] = original_payload
        globals()["PDR_TARGET"] = original_target
    return output


def print_extended_study(name, results):
    print(f"\n{name}:")
    for label, summary in results.items():
        print(
            f"{label}: PDR={summary['pdr'][0]:.3f} "
            f"[+/- {summary['pdr'][2]:.3f}], "
            f"energy={summary['active_energy_mah'][0]:.6f} mAh/cycle"
        )


def compare_external_measurements(path, simulation_summary):
    """Compare simulation means with a user-supplied measurement CSV.

    CSV columns: policy,pdr,active_energy_mah,total_energy_per_hour.
    """
    if not path or not os.path.exists(path):
        print(
            "\nExternal measurement comparison: not run. "
            "Provide a CSV with policy,pdr,active_energy_mah,"
            "total_energy_per_hour."
        )
        return
    print("\nExternal measurement comparison:")
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            policy = row["policy"]
            if policy not in simulation_summary:
                continue
            summary = simulation_summary[policy]
            print(
                f"{policy}: measured PDR={row['pdr']}, "
                f"simulated PDR={summary['pdr'][0]:.3f}; "
                f"measured energy={row['active_energy_mah']}, "
                f"simulated energy="
                f"{summary['active_energy_mah'][0]:.6f} mAh/cycle"
            )


def save_seed_results(baseline_results, filename="evaluation_seed_results.csv"):
    """Persist raw per-seed results for reproducible statistical analysis."""
    path = os.path.join(PLOT_DIRECTORY, filename)
    fields = [
        "policy", "seed_index", "pdr", "loss_rate", "mean_snr", "mean_cbr",
        "active_energy_mah", "total_energy_per_hour", "throughput_bps",
        "airtime_seconds_per_cycle", "packet_jitter_seconds",
        "retransmission_rate", "duty_cycle_limited_fraction",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for policy, rows in baseline_results.items():
            for seed_index, row in enumerate(rows):
                writer.writerow({
                    field: row.get(field, "")
                    for field in fields
                    if field not in ("policy", "seed_index")
                } | {
                    "policy": policy,
                    "seed_index": seed_index,
                })
    print(f"Saved raw seed results: {path}")


def print_validation_protocol():
    print("\nValidation protocol:")
    print(f"Radio calibration source: {RADIO_CALIBRATION['source']}")
    print(f"Datasheet URL: {RADIO_CALIBRATION['url']}")
    print(
        "The model includes distance-based path loss, stochastic "
        "interference, automatic confirmed-message retransmissions, "
        f"and a {DUTY_CYCLE_LIMIT:.1%} duty-cycle constraint."
    )
    print(
        "Publication results should replace assumptions with measured "
        "current, voltage, airtime, and packet-trace data."
    )


def fixed_policy(action, seeds, steps):
    results = []
    for seed in seeds:
        environment = EmbodiedLoRaEnvironment(seed)
        result = run_fixed_episode(environment, action, steps)
        results.append(result)
    return results


def choose_adr_action(observation, previous_power=1):
    """Choose a conservative ADR-style SF and power at a fixed interval.

    This is a simulator baseline, not a complete LoRaWAN network-server ADR
    implementation. It uses the recent measured SNR, a 10 dB link margin,
    and keeps the transmission interval fixed because LoRaWAN ADR controls
    data rate and TX power rather than application reporting frequency.
    """
    if observation.snr <= -90.0:
        return 12, 1, ADR_INTERVAL

    selected_sf = 12
    for sf in SF_VALUES:
        if observation.snr >= REQUIRED_SNR[sf] + ADR_MARGIN_DB:
            selected_sf = sf
            break

    selected_power = 1
    for power in reversed(POWER_VALUES):
        estimated_snr = (
            observation.snr
            + POWER_DBM[power]
            - POWER_DBM[previous_power]
        )
        if estimated_snr >= REQUIRED_SNR[selected_sf] + ADR_MARGIN_DB:
            selected_power = power
            break

    return selected_sf, selected_power, ADR_INTERVAL


def run_adr_episode(environment, steps, collect_trace=False):
    """Evaluate the ADR-style baseline without learning or exploration."""
    observation = environment.reset()
    previous_power = 1
    delivered = 0
    active = 0.0
    sleep = 0.0
    delivered_samples = []
    snrs = []
    cbrs = []
    active_samples = []
    sleep_samples = []
    airtime = []
    timestamps = []
    retransmissions = []
    duty_cycle_limited = []
    actions = []
    trace = []

    for step in range(1, steps + 1):
        previous_observation = observation
        action = choose_adr_action(observation, previous_power)
        result = environment.transmit(action)
        observation = environment.observe()
        previous_power = action[1]
        delivered += int(result.delivered)
        active += result.active_energy_mah
        sleep += result.sleep_energy_mah
        delivered_samples.append(int(result.delivered))
        snrs.append(result.snr)
        cbrs.append(result.cbr)
        active_samples.append(result.active_energy_mah)
        sleep_samples.append(result.sleep_energy_mah)
        airtime.append(result.airtime_seconds)
        timestamps.append(environment.elapsed_seconds)
        retransmissions.append(result.retransmissions)
        duty_cycle_limited.append(result.duty_cycle_limited)
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
    output.update(
        summarize_episode_samples(
            delivered_samples,
            snrs,
            cbrs,
            active_samples,
            sleep_samples,
            airtime,
            timestamps,
            environment.elapsed_seconds,
            retransmissions,
            duty_cycle_limited,
        )
    )
    if collect_trace:
        output["trace"] = trace
    return output


def adr_policy(seeds, steps, representative_seed=None):
    results = []
    for seed in seeds:
        environment = EmbodiedLoRaEnvironment(seed)
        results.append(
            run_adr_episode(
                environment,
                steps,
                collect_trace=seed == representative_seed,
            )
        )
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
    delivered_samples = []
    snrs = []
    cbrs = []
    active_samples = []
    sleep_samples = []
    airtime = []
    timestamps = []
    retransmissions = []
    duty_cycle_limited = []

    for _ in range(steps):
        result = environment.transmit(action)
        observation = environment.observe()
        delivered += int(result.delivered)
        active += result.active_energy_mah
        sleep += result.sleep_energy_mah
        delivered_samples.append(int(result.delivered))
        snrs.append(result.snr)
        cbrs.append(result.cbr)
        active_samples.append(result.active_energy_mah)
        sleep_samples.append(result.sleep_energy_mah)
        airtime.append(result.airtime_seconds)
        timestamps.append(environment.elapsed_seconds)
        retransmissions.append(result.retransmissions)
        duty_cycle_limited.append(result.duty_cycle_limited)

    output = {
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
    output.update(
        summarize_episode_samples(
            delivered_samples,
            snrs,
            cbrs,
            active_samples,
            sleep_samples,
            airtime,
            timestamps,
            environment.elapsed_seconds,
            retransmissions,
            duty_cycle_limited,
        )
    )
    return output


def aggregate(results):
    names = (
        "pdr",
        "loss_rate",
        "loss_count",
        "mean_snr",
        "mean_cbr",
        "active_energy_mah",
        "sleep_energy_mah",
        "total_energy_mah",
        "battery",
        "total_energy_per_hour",
        "airtime_seconds_per_cycle",
        "airtime_fraction",
        "throughput_bps",
        "packet_jitter_seconds",
        "retransmission_rate",
        "duty_cycle_limited_fraction",
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


def evaluate_policy_lifetime(
    policy_name,
    agent=None,
    fixed_action=NO_ADR_ACTION,
    duration_hours=24,
    seed=2007,
):
    """Run a continuous battery lifetime test for any compared policy."""
    environment = EmbodiedLoRaEnvironment(seed)
    observation = environment.reset()
    horizon = duration_hours * 3600.0
    delivered = []
    snrs = []
    cbrs = []
    active = []
    sleep = []
    airtime = []
    timestamps = []
    retransmissions = []
    duty_cycle_limited = []
    previous_power = 1

    while environment.elapsed_seconds < horizon:
        if policy_name == "Learned Expected SARSA":
            action = agent.choose(agent.state(observation), explore=False)
        elif policy_name == "Standard LoRaWAN ADR-style":
            action = choose_adr_action(observation, previous_power)
        else:
            action = fixed_action
        if action[2] > horizon - environment.elapsed_seconds:
            break
        result = environment.transmit(action)
        observation = environment.observe()
        previous_power = action[1]
        delivered.append(int(result.delivered))
        snrs.append(result.snr)
        cbrs.append(result.cbr)
        active.append(result.active_energy_mah)
        sleep.append(result.sleep_energy_mah)
        airtime.append(result.airtime_seconds)
        timestamps.append(environment.elapsed_seconds)
        retransmissions.append(result.retransmissions)
        duty_cycle_limited.append(result.duty_cycle_limited)
        if environment.battery_mah <= 0.0:
            break

    elapsed_seconds = environment.elapsed_seconds
    metrics = summarize_episode_samples(
        delivered,
        snrs,
        cbrs,
        active,
        sleep,
        airtime,
        timestamps,
        elapsed_seconds,
        retransmissions,
        duty_cycle_limited,
    )
    return {
        "duration_hours": elapsed_seconds / 3600.0,
        "transmissions": len(delivered),
        "pdr": metrics["pdr"],
        "active_energy_mah": sum(active),
        "sleep_energy_mah": sum(sleep),
        "total_energy_mah": sum(active) + sum(sleep),
        "total_energy_per_hour": (
            sum(active) + sum(sleep)
        ) / max(elapsed_seconds / 3600.0, 1e-12),
        "battery": environment.battery_mah,
        "mean_snr": metrics["mean_snr"],
        "mean_cbr": metrics["mean_cbr"],
        "packet_jitter_seconds": metrics["packet_jitter_seconds"],
        "retransmissions": 0,
    }


def evaluate_lifetime_policies(
    agents,
    duration_hours=24,
    seeds=LIFETIME_SEEDS,
):
    """Evaluate all three policies over independent lifetime seeds."""
    output = {}
    for policy_name in (
        "Learned Expected SARSA",
        "Standard LoRaWAN ADR-style",
        "LoRaWAN without ADR",
    ):
        agent = agents[-1] if policy_name == "Learned Expected SARSA" else None
        output[policy_name] = [
            evaluate_policy_lifetime(
                policy_name,
                agent=agent,
                duration_hours=duration_hours,
                seed=seed,
            )
            for seed in seeds
        ]
    return output


def print_lifetime_summary(lifetime_results):
    print("\nMulti-seed continuous lifetime (95% CI):")
    for name, results in lifetime_results.items():
        duration = mean_ci([row["duration_hours"] for row in results])
        battery = mean_ci([row["battery"] for row in results])
        hourly = mean_ci(
            [row["total_energy_per_hour"] for row in results]
        )
        print(
            f"{name}: duration={duration[0]:.2f} +/- {duration[2]:.2f} h, "
            f"remaining battery={battery[0]:.2f} +/- {battery[2]:.2f} mAh, "
            f"energy/hour={hourly[0]:.6f} +/- {hourly[2]:.6f} mAh"
        )


def paired_significance(results_a, results_b, seed=991):
    """Paired permutation p-value, effect size, and bootstrap CI."""
    differences = [
        a - b for a, b in zip(results_a, results_b)
    ]
    if len(differences) < 2:
        return {
            "difference": differences[0] if differences else 0.0,
            "p_value": 1.0,
            "effect_size": 0.0,
            "ci": 0.0,
        }
    difference_mean = mean(differences)
    difference_sd = stdev(differences)
    observed = abs(difference_mean)
    rng = random.Random(seed)
    exceed = 0
    permutations = 20000
    for _ in range(permutations):
        signed = [
            value if rng.random() < 0.5 else -value
            for value in differences
        ]
        if abs(mean(signed)) >= observed:
            exceed += 1
    p_value = (exceed + 1) / (permutations + 1)
    effect_size = (
        difference_mean / difference_sd
        if difference_sd else 0.0
    )
    bootstrap = []
    for _ in range(5000):
        sample = [
            rng.choice(differences)
            for _ in differences
        ]
        bootstrap.append(mean(sample))
    bootstrap.sort()
    lower = bootstrap[int(0.025 * len(bootstrap))]
    upper = bootstrap[int(0.975 * len(bootstrap))]
    return {
        "difference": difference_mean,
        "p_value": p_value,
        "effect_size": effect_size,
        "ci": (lower, upper),
    }


def print_significance_tests(baseline_results):
    learned = baseline_results["Learned Expected SARSA"]
    for metric in ("pdr", "active_energy_mah"):
        learned_values = [row[metric] for row in learned]
        print(f"\nPaired significance test ({metric}, paired permutation):")
        for comparison in (
            "Standard LoRaWAN ADR-style",
            "LoRaWAN without ADR",
        ):
            comparison_values = [
                row[metric] for row in baseline_results[comparison]
            ]
            result = paired_significance(
                learned_values,
                comparison_values,
            )
            print(
                f"Learned - {comparison}: "
                f"mean difference={result['difference']:.6f}, "
                f"95% bootstrap CI=[{result['ci'][0]:.6f}, "
                f"{result['ci'][1]:.6f}], "
                f"permutation p={result['p_value']:.4g}, "
                f"Cohen dz={result['effect_size']:.3f}"
            )


def print_operating_region_metrics(baseline_results):
    print("\nPDR by operating region:")
    for name, results in baseline_results.items():
        def region_mean(key, region):
            values = [
                row[key][region]
                for row in results
                if not math.isnan(row[key][region])
            ]
            return mean(values) if values else float("nan")

        snr_regions = {
            region: region_mean("pdr_by_snr_region", region)
            for region in ("weak", "medium", "strong")
        }
        cbr_regions = {
            region: region_mean("pdr_by_cbr_region", region)
            for region in ("low", "medium", "high")
        }
        print(
            f"{name}: SNR weak/medium/strong="
            f"{snr_regions['weak']:.3f}/"
            f"{snr_regions['medium']:.3f}/"
            f"{snr_regions['strong']:.3f}; "
            f"CBR low/medium/high="
            f"{cbr_regions['low']:.3f}/"
            f"{cbr_regions['medium']:.3f}/"
            f"{cbr_regions['high']:.3f}"
        )


def print_learned_agent_results(agent_results):
    print("\nIndependent evaluation of trained Expected SARSA agents:")
    for index, results in enumerate(agent_results, start=1):
        summary = aggregate(results)
        print_summary(f"Learned agent {index}", summary)


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
    print(
        f"  SNR={summary['mean_snr'][0]:.2f} dB, "
        f"CBR={summary['mean_cbr'][0]:.3f}, "
        f"loss rate={summary['loss_rate'][0]:.3f}, "
        f"throughput={summary['throughput_bps'][0]:.3f} bit/s, "
        f"airtime/cycle={summary['airtime_seconds_per_cycle'][0]:.3f} s, "
        f"jitter={summary['packet_jitter_seconds'][0]:.3f} s"
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
    adr_summary = aggregate(
        baseline_results["Standard LoRaWAN ADR-style"]
    )
    no_adr_summary = aggregate(
        baseline_results["LoRaWAN without ADR"]
    )
    learned_pdr = learned_summary["pdr"]
    learned_active = learned_summary["active_energy_mah"]
    learned_hourly = learned_summary["total_energy_per_hour"]
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
        f"  Compared with Standard LoRaWAN ADR-style: "
        f"PDR {learned_pdr[0] - adr_summary['pdr'][0]:+.3f}, "
        f"active energy "
        f"{((learned_active[0] / adr_summary['active_energy_mah'][0]) - 1) * 100:+.1f}%, "
        f"total/hour "
        f"{((learned_hourly[0] / adr_summary['total_energy_per_hour'][0]) - 1) * 100:+.1f}%"
    )
    print(
        f"  Compared with LoRaWAN without ADR: "
        f"PDR {learned_pdr[0] - no_adr_summary['pdr'][0]:+.3f}, "
        f"active energy "
        f"{((learned_active[0] / no_adr_summary['active_energy_mah'][0]) - 1) * 100:+.1f}%, "
        f"total/hour "
        f"{((learned_hourly[0] / no_adr_summary['total_energy_per_hour'][0]) - 1) * 100:+.1f}%"
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
        save_plotly_figure(
            figure,
            "baseline_comparison_95pct_ci",
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
    save_matplotlib_figure("baseline_comparison_95pct_ci.png")
    plt.show()


def plot_pdr_energy_tradeoff(baseline_results):
    """Plot the PDR-energy Pareto trade-off with 95% confidence bars."""
    names = list(baseline_results)
    summaries = [aggregate(baseline_results[name]) for name in names]
    x = [summary["active_energy_mah"][0] for summary in summaries]
    y = [summary["pdr"][0] for summary in summaries]
    xerr = [summary["active_energy_mah"][2] for summary in summaries]
    yerr = [summary["pdr"][2] for summary in summaries]
    plt.figure(figsize=(8, 6))
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=4)
    for name, x_value, y_value in zip(names, x, y):
        plt.annotate(name, (x_value, y_value), xytext=(6, 6),
                     textcoords="offset points")
    plt.axhline(PDR_TARGET, color="red", linestyle="--",
                label=f"PDR target ({PDR_TARGET:.2f})")
    plt.xlabel("Active energy (mAh/cycle)")
    plt.ylabel("PDR")
    plt.title("PDR-energy trade-off")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    save_matplotlib_figure("pdr_energy_tradeoff_300dpi.png")
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
    save_matplotlib_figure("training_convergence_300dpi.png")
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
    save_matplotlib_figure("action_frequencies_300dpi.png")
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
    save_matplotlib_figure("evaluation_diagnostics_300dpi.png")
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
    save_plotly_figure(
        figure,
        "interactive_evaluation_dashboard",
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
    save_matplotlib_figure("continuous_lifetime_300dpi.png")
    plt.show()


def main():
    histories = []
    agents = []
    for seed in TRAINING_SEEDS:
        agent, history = train(seed=seed)
        agents.append(agent)
        histories.append(history)

    agent = agents[-1]
    learned_agent_results = [
        evaluate_policy(
            trained_agent,
            EVALUATION_SEEDS,
            EVALUATION_STEPS,
        )
        for trained_agent in agents
    ]
    learned_results = evaluate_policy(
        agent,
        EVALUATION_SEEDS,
        EVALUATION_STEPS,
        representative_seed=EVALUATION_SEEDS[0],
    )
    adr_results = adr_policy(
        EVALUATION_SEEDS,
        EVALUATION_STEPS,
        representative_seed=EVALUATION_SEEDS[0],
    )

    no_adr_results = fixed_policy(
        NO_ADR_ACTION,
        EVALUATION_SEEDS,
        EVALUATION_STEPS,
    )
    fixed_sf7_results = fixed_policy(
        (7, 1, 60),
        EVALUATION_SEEDS,
        EVALUATION_STEPS,
    )
    fixed_sf10_results = fixed_policy(
        (10, 1, 60),
        EVALUATION_SEEDS,
        EVALUATION_STEPS,
    )
    reliability_action, reliability_results = reliability_policy(
        EVALUATION_SEEDS,
        EVALUATION_STEPS,
    )

    baseline_results = {
        "Learned Expected SARSA": learned_results,
        "Standard LoRaWAN ADR-style": adr_results,
        "LoRaWAN without ADR": no_adr_results,
        "Fixed SF7, 14 dBm": fixed_sf7_results,
        "Fixed SF10, 14 dBm": fixed_sf10_results,
        "Reliability-first heuristic": reliability_results,
    }

    lifetime = evaluate_continuous_lifetime(
        agent,
        duration_hours=24,
        seed=2007,
    )
    lifetime_results = evaluate_lifetime_policies(
        agents,
        duration_hours=24,
        seeds=LIFETIME_SEEDS,
    )

    print_report(
        agent,
        histories,
        learned_results,
        baseline_results,
        lifetime,
    )
    print_learned_agent_results(learned_agent_results)
    print_operating_region_metrics(baseline_results)
    print_significance_tests(baseline_results)
    print_lifetime_summary(lifetime_results)
    print_validation_protocol()
    compare_external_measurements(MEASUREMENT_CSV, {
        name: aggregate(results)
        for name, results in baseline_results.items()
    })
    if RUN_EXTENDED_STUDIES:
        print_extended_study(
            "Ablation study",
            run_ablation_study(),
        )
        print_extended_study(
            "Sensitivity study",
            run_sensitivity_study(),
        )
    print(
        f"\nReliability-first validation action: {reliability_action}"
    )
    display_confidence_table(baseline_results)
    save_seed_results(baseline_results)

    print("\nBaseline configurations:")
    print(
        "Standard LoRaWAN ADR-style: adaptive SF/power, "
        f"fixed interval={ADR_INTERVAL}s, margin={ADR_MARGIN_DB:.1f} dB"
    )
    print(
        "LoRaWAN without ADR: "
        f"fixed SF{NO_ADR_ACTION[0]}, "
        f"{POWER_DBM[NO_ADR_ACTION[1]]} dBm, "
        f"interval={NO_ADR_ACTION[2]}s"
    )

    plot_training(histories)
    plot_action_frequencies(agent, learned_results)
    plot_baseline_comparison(baseline_results)
    plot_pdr_energy_tradeoff(baseline_results)
    plot_evaluation_analysis(learned_results[0])
    plot_interactive_evaluation(learned_results[0])
    plot_lifetime(lifetime)


if __name__ == "__main__":
    main()
