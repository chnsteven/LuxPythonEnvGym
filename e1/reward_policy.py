import copy
import json
import os

import numpy as np

from luxai2021.game.actions import (
    MoveAction,
    ResearchAction,
    SpawnCartAction,
    SpawnCityAction,
    SpawnWorkerAction,
    TransferAction,
)
from luxai2021.game.constants import Constants


RESOURCE_TYPES = (
    Constants.RESOURCE_TYPES.WOOD,
    Constants.RESOURCE_TYPES.COAL,
    Constants.RESOURCE_TYPES.URANIUM,
)


def deep_update(base, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_reward_config(reward_config):
    if reward_config is None:
        return {}
    if isinstance(reward_config, dict):
        return copy.deepcopy(reward_config)
    if isinstance(reward_config, str):
        config_text = reward_config.strip()
        if os.path.exists(config_text):
            with open(config_text, "r", encoding="utf-8") as config_file:
                return json.load(config_file)
        return json.loads(config_text)
    raise TypeError("reward_config must be a dict, JSON string, JSON file path, or None")


def make_simple_reward_config(enabled_groups=()):
    groups = set(enabled_groups)
    return {
        "error_reward": -1.0,
        "phase_turns": {"early_end": 120, "mid_end": 240},
        "components": {
            # 1. Win/loss and final city tiles. Enabled by default.
            "outcome": {
                "enabled": True,
                "weight": 1.0,
                "win_loss": 1.0,
                "final_city_tile": 1.0,
                "city_tile_margin": 0.5,
                "city_tile_divisor": 30.0,
                "city_tile_margin_divisor": 30.0,
                "phase": {"early": 0.0, "mid": 0.0, "late": 1.0},
            },
            # 2. Resource/city/unit growth.
            "growth": {
                "enabled": 2 in groups,
                "weight": 0.9,
                "fuel_generated": 0.60,
                "city_fuel": 0.20,
                "cargo_fuel": 0.12,
                "city_tile": 0.90,
                "worker": 0.45,
                "cart": 0.25,
                "fuel_divisor": 12000.0,
                "city_tile_divisor": 5.0,
                "unit_divisor": 8.0,
                "phase": {"early": 1.25, "mid": 1.0, "late": 0.65},
            },
            # 3. City/unit survival.
            "survival": {
                "enabled": 3 in groups,
                "weight": 0.45,
                "safe_turns_target": 12.0,
                "city_tile_divisor": 30.0,
                "unit_divisor": 30.0,
                "city_safety": 0.70,
                "unit_survival": 0.30,
                "phase": {"early": 0.35, "mid": 0.85, "late": 1.65},
            },
            # 4. Unit action tendency.
            "unit_tendency": {
                "enabled": 4 in groups,
                "weight": 0.25,
                "progress_divisor": 12.0,
                "worker_loaded_ratio": 0.80,
                "cart_loaded_ratio": 0.50,
                "phase": {"early": 1.20, "mid": 0.90, "late": 0.35},
            },
            # 5. City action tendency.
            "city_tendency": {
                "enabled": 5 in groups,
                "weight": 0.22,
                "action_divisor": 8.0,
                "abundant_ratio": 0.25,
                "unit_when_abundant": 1.0,
                "research_when_abundant": -0.25,
                "research_when_scarce": 1.0,
                "unit_when_scarce": -0.40,
                "cart_factor": 0.55,
                "phase": {"early": 1.20, "mid": 0.85, "late": 0.35},
            },
            # 6. Phase tendency: early accepted movement/exploration, late survival.
            "phase_tendency": {
                "enabled": 6 in groups,
                "weight": 0.25,
                "move_divisor": 20.0,
                "survival_scale": 0.60,
                "phase": {"early": 1.00, "mid": 0.60, "late": 1.20},
            },
            # 7. Penalties. Enabled by default.
            "penalty": {
                "enabled": True,
                "weight": 0.35,
                "invalid_action": 1.0,
                "no_progress": 0.6,
                "divisor": 8.0,
                "phase": {"early": 1.0, "mid": 1.0, "late": 0.8},
            },
        },
    }


REWARD_PRESETS = {
    "simple_base": make_simple_reward_config(()),
    "simple_g2": make_simple_reward_config((2,)),
    "simple_g3": make_simple_reward_config((3,)),
    "simple_g245": make_simple_reward_config((2, 4, 5)),
    "simple_g345": make_simple_reward_config((3, 4, 5)),
    "simple_g2345": make_simple_reward_config((2, 3, 4, 5)),
    "simple_g236": make_simple_reward_config((2, 3, 6)),
    "simple_g23456": make_simple_reward_config((2, 3, 4, 5, 6)),
}

# Compatibility aliases. They keep older notebooks/scripts importable; new
# experiments should use simple_* presets above.
for _alias in [
    "balanced",
    "minimal",
    "expansion",
    "logistics",
    "pressure",
    "survival",
    "phased_competitive",
    "early_expand_expansion",
    "early_expand_logistics",
    "early_expand_pressure",
    "early_expand_phased_competitive",
]:
    REWARD_PRESETS.setdefault(_alias, copy.deepcopy(REWARD_PRESETS["simple_base"]))


class ConfigurableRewardPolicy:
    REWARD_PRESET_NAMES = tuple(REWARD_PRESETS.keys())

    def __init__(self, reward_preset="balanced", reward_config=None):
        self.reward_preset = reward_preset
        self.config = self.build_reward_config(reward_preset, reward_config)
        self.reset_runtime_state()

    @staticmethod
    def build_reward_config(reward_preset, reward_config):
        if reward_preset not in REWARD_PRESETS:
            raise ValueError(
                "Unknown reward_preset '%s'. Available presets: %s"
                % (reward_preset, ", ".join(sorted(REWARD_PRESETS.keys())))
            )
        config = copy.deepcopy(REWARD_PRESETS[reward_preset])
        deep_update(config, load_reward_config(reward_config))
        return config

    def reset_runtime_state(self):
        self.team = None
        self.last_snapshot = None
        self.initial_resource_amount = {resource_type: 0 for resource_type in RESOURCE_TYPES}
        self.pending_invalid_actions = []
        self.pending_unit_actions = []
        self.pending_city_tendency = []
        self.pending_accepted_moves = 0
        self.pending_fallback_actions = 0
        self.last_fallback_scale = 1.0
        self.last_reward_breakdown = {}
        # 满资源滞留惩罚：{unit_id: int} 记录每个 worker 连续满资源的回合数
        self.unit_full_cargo_turns = {}

    def game_start(self, game, team):
        self.reset_runtime_state()
        self.team = team
        self.initial_resource_amount = self._map_resource_amount(game, mineable_only=False)
        self.last_snapshot = self._build_reward_snapshot(game)

    def record_action(self, game, team, action, action_was_accepted):
        if self.team is None:
            self.team = team
        if action is None or getattr(action, "team", self.team) != self.team:
            return

        self._record_unit_action_attempt(game, action)

        if not action_was_accepted:
            self.pending_invalid_actions.append(1.0)
            return

        if isinstance(action, MoveAction) and action.direction != Constants.DIRECTIONS.CENTER:
            self.pending_accepted_moves += 1
        if isinstance(action, (SpawnWorkerAction, SpawnCartAction, ResearchAction)):
            self._record_city_tendency_event(game, action)

    def record_invalid_action(self):
        self.pending_invalid_actions.append(1.0)

    def record_fallback_action(self):
        self.pending_fallback_actions += 1

    def get_reward(self, game, team, is_game_finished, is_new_turn, is_game_error):
        if is_game_error:
            self.last_reward_breakdown = {"rew/error": self.config.get("error_reward", -1.0)}
            return float(self.config.get("error_reward", -1.0))

        if not is_new_turn and not is_game_finished:
            return 0.0

        self.team = team
        current = self._build_reward_snapshot(game)
        previous = self.last_snapshot
        if previous is None:
            self.last_snapshot = current
            self._clear_pending_events()
            return 0.0

        phase = current["phase"]
        rewards = {}
        self._add_component(rewards, "growth", self._growth_raw(current, previous), phase)
        self._add_component(rewards, "survival", self._survival_raw(current), phase)
        self._add_component(rewards, "unit_tendency", self._unit_tendency_raw(current, previous), phase)
        self._add_component(rewards, "city_tendency", self._city_tendency_raw(), phase)
        self._add_component(rewards, "phase_tendency", self._phase_tendency_raw(current), phase)
        self._add_component(rewards, "penalty", self._penalty_raw(current), phase)
        full_cargo_pen = self._full_cargo_penalty_raw(game)
        if full_cargo_pen != 0.0:
            rewards["rew/full_cargo_penalty"] = full_cargo_pen
        if is_game_finished:
            self._add_component(rewards, "outcome", self._outcome_raw(game, current), phase)

        reward = float(sum(rewards.values()))
        if not np.isfinite(reward):
            reward = 0.0

        fallback_scale = 1.0
        if self.pending_fallback_actions > 0:
            fallback_scale = self.config.get("fallback_reward_scale", 0.2)
            reward *= fallback_scale
            rewards = {name: value * fallback_scale for name, value in rewards.items()}
        self.last_fallback_scale = fallback_scale

        self.last_reward_breakdown = rewards
        self.last_snapshot = current
        self._clear_pending_events()
        return reward

    def _clear_pending_events(self):
        self.pending_invalid_actions = []
        self.pending_unit_actions = []
        self.pending_city_tendency = []
        self.pending_accepted_moves = 0
        self.pending_fallback_actions = 0

    def _phase(self, turn):
        phase_turns = self.config.get("phase_turns", {})
        if turn < phase_turns.get("early_end", 120):
            return "early"
        if turn < phase_turns.get("mid_end", 240):
            return "mid"
        return "late"

    def _component_config(self, name):
        return self.config.get("components", {}).get(name, {})

    def _component_weight(self, name, phase):
        component = self._component_config(name)
        if not component.get("enabled", False):
            return 0.0
        return component.get("weight", 0.0) * component.get("phase", {}).get(phase, 1.0)

    def _add_component(self, rewards, name, raw_value, phase):
        weight = self._component_weight(name, phase)
        if weight == 0.0 or raw_value == 0:
            return
        value = float(raw_value) * weight
        if value != 0 and np.isfinite(value):
            rewards["rew/%s" % name] = value

    def _clip_unit(self, value):
        return max(-1.0, min(1.0, float(value)))

    def _resource_fuel_rate(self, game, resource_type):
        return game.configs["parameters"]["RESOURCE_TO_FUEL_RATE"][resource_type.upper()]

    def _cargo_fuel_value(self, game, cargo):
        return sum(
            cargo.get(resource_type, 0) * self._resource_fuel_rate(game, resource_type)
            for resource_type in RESOURCE_TYPES
        )

    def _map_resource_amount(self, game, mineable_only):
        amounts = {resource_type: 0 for resource_type in RESOURCE_TYPES}
        for cell in game.map.resources:
            if not cell.has_resource():
                continue
            if mineable_only and not self._resource_is_mineable(game, cell.resource.type):
                continue
            amounts[cell.resource.type] += cell.resource.amount
        return amounts

    def _resource_is_mineable(self, game, resource_type):
        if resource_type == Constants.RESOURCE_TYPES.WOOD:
            return True
        researched = game.state["teamStates"][self.team]["researched"]
        if resource_type == Constants.RESOURCE_TYPES.COAL:
            return bool(researched["coal"])
        if resource_type == Constants.RESOURCE_TYPES.URANIUM:
            return bool(researched["uranium"])
        return False

    def _mineable_resource_cells(self, game):
        return [
            cell for cell in game.map.resources
            if cell.has_resource() and self._resource_is_mineable(game, cell.resource.type)
        ]

    def _own_city_cells(self, game):
        return [
            city_cell
            for city in game.cities.values()
            if city.team == self.team
            for city_cell in city.city_cells
        ]

    def _own_cart_units(self, game):
        return [
            unit for unit in game.state["teamStates"][self.team]["units"].values()
            if unit.type == Constants.UNIT_TYPES.CART
        ]

    def _own_worker_units(self, game):
        return [
            unit for unit in game.state["teamStates"][self.team]["units"].values()
            if unit.type == Constants.UNIT_TYPES.WORKER
        ]

    def _low_fuel_city_cells(self, game):
        city_cells = []
        for city in game.cities.values():
            if city.team != self.team:
                continue
            survival_turns = city.fuel / max(city.get_light_upkeep(), 1)
            if survival_turns <= 15:
                city_cells.extend(city.city_cells)
        if city_cells:
            return city_cells
        return self._own_city_cells(game)

    def _buildable_positions(self, game):
        positions = []
        for y in range(game.map.height):
            for x in range(game.map.width):
                cell = game.map.get_cell(x, y)
                if cell.is_city_tile() or cell.has_resource() or cell.units:
                    continue
                positions.append(cell.pos)
        return positions

    def _nearest_distance(self, pos, targets):
        if not targets:
            return None
        return min(pos.distance_to(getattr(target, "pos", target)) for target in targets)

    def _unit_capacity(self, unit):
        return max(sum(unit.cargo.values()) + unit.get_cargo_space_left(), 1)

    def _unit_cargo_amount(self, unit):
        return sum(unit.cargo.values())

    def _unit_target_distance(self, game, unit, target_cache=None):
        if target_cache is None:
            target_cache = {
                "mineable": self._mineable_resource_cells(game),
                "own_city": self._own_city_cells(game),
                "carts": self._own_cart_units(game),
                "workers": self._own_worker_units(game),
                "low_fuel_city": self._low_fuel_city_cells(game),
                "buildable": self._buildable_positions(game),
            }
        cargo_ratio = self._unit_cargo_amount(unit) / self._unit_capacity(unit)
        config = self._component_config("unit_tendency")
        worker_loaded_ratio = config.get("worker_loaded_ratio", 0.80)
        cart_loaded_ratio = config.get("cart_loaded_ratio", 0.50)

        if unit.type == Constants.UNIT_TYPES.WORKER:
            if cargo_ratio < worker_loaded_ratio:
                return self._nearest_distance(unit.pos, target_cache["mineable"])
            targets = []
            targets.extend(target_cache["own_city"])
            targets.extend(target_cache["carts"])
            targets.extend(target_cache["buildable"])
            return self._nearest_distance(unit.pos, targets)

        if unit.type == Constants.UNIT_TYPES.CART:
            if cargo_ratio < cart_loaded_ratio:
                targets = []
                targets.extend(target_cache["workers"])
                targets.extend(target_cache["mineable"])
                return self._nearest_distance(unit.pos, targets)
            return self._nearest_distance(unit.pos, target_cache["low_fuel_city"])

        return None

    def _record_unit_action_attempt(self, game, action):
        unit_id = getattr(action, "unit_id", None)
        if not unit_id:
            return
        try:
            unit = game.get_unit(self.team, unit_id)
        except KeyError:
            return
        self.pending_unit_actions.append({
            "unit_id": unit.id,
            "x": unit.pos.x,
            "y": unit.pos.y,
            "cargo": self._unit_cargo_amount(unit),
        })

    def _record_city_tendency_event(self, game, action):
        config = self._component_config("city_tendency")
        if not config.get("enabled", False):
            return
        initial_total = max(sum(self.initial_resource_amount.values()), 1)
        mineable_total = sum(self._map_resource_amount(game, mineable_only=True).values())
        abundant = (mineable_total / initial_total) >= config.get("abundant_ratio", 0.25)

        if isinstance(action, ResearchAction):
            score = (
                config.get("research_when_abundant", -0.25)
                if abundant else
                config.get("research_when_scarce", 1.0)
            )
        elif isinstance(action, SpawnWorkerAction):
            score = (
                config.get("unit_when_abundant", 1.0)
                if abundant else
                config.get("unit_when_scarce", -0.40)
            )
        elif isinstance(action, SpawnCartAction):
            base = (
                config.get("unit_when_abundant", 1.0)
                if abundant else
                config.get("unit_when_scarce", -0.40)
            )
            score = base * config.get("cart_factor", 0.55)
        else:
            return
        self.pending_city_tendency.append(score)

    def _build_reward_snapshot(self, game):
        team = self.team
        opponent = (team + 1) % 2
        units = list(game.state["teamStates"][team]["units"].values())
        opponent_units = list(game.state["teamStates"][opponent]["units"].values())

        city_count = 0
        city_tile_count = 0
        opponent_city_tile_count = 0
        city_fuel = 0.0
        city_upkeep = 0.0
        survival_total = 0.0
        for city in game.cities.values():
            if city.team == team:
                city_count += 1
                city_tile_count += len(city.city_cells)
                city_fuel += city.fuel
                upkeep = city.get_light_upkeep()
                city_upkeep += upkeep
                survival_total += city.fuel / max(upkeep, 1)
            else:
                opponent_city_tile_count += len(city.city_cells)

        worker_count = sum(1 for unit in units if unit.type == Constants.UNIT_TYPES.WORKER)
        cart_count = sum(1 for unit in units if unit.type == Constants.UNIT_TYPES.CART)
        cargo_fuel = sum(self._cargo_fuel_value(game, unit.cargo) for unit in units)
        resource_fuel_collected = sum(
            game.stats["teamStats"][team]["resourcesCollected"].get(resource_type, 0)
            * self._resource_fuel_rate(game, resource_type)
            for resource_type in RESOURCE_TYPES
        )
        opponent_resource_fuel_collected = sum(
            game.stats["teamStats"][opponent]["resourcesCollected"].get(resource_type, 0)
            * self._resource_fuel_rate(game, resource_type)
            for resource_type in RESOURCE_TYPES
        )

        target_cache = {
            "mineable": self._mineable_resource_cells(game),
            "own_city": self._own_city_cells(game),
            "carts": self._own_cart_units(game),
            "workers": self._own_worker_units(game),
            "low_fuel_city": self._low_fuel_city_cells(game),
            "buildable": self._buildable_positions(game),
        }

        unit_snapshots = {}
        for unit in units:
            unit_snapshots[unit.id] = {
                "x": unit.pos.x,
                "y": unit.pos.y,
                "type": unit.type,
                "cargo": self._unit_cargo_amount(unit),
                "cargo_fuel": self._cargo_fuel_value(game, unit.cargo),
                "target_distance": self._unit_target_distance(game, unit, target_cache),
            }

        return {
            "turn": game.state["turn"],
            "phase": self._phase(game.state["turn"]),
            "is_night": game.is_night(),
            "city_count": city_count,
            "city_tile_count": city_tile_count,
            "opponent_city_tile_count": opponent_city_tile_count,
            "city_fuel": city_fuel,
            "city_upkeep": city_upkeep,
            "avg_city_survival_turns": survival_total / max(city_count, 1),
            "unit_count": len(units),
            "worker_count": worker_count,
            "cart_count": cart_count,
            "opponent_unit_count": len(opponent_units),
            "cargo_fuel": cargo_fuel,
            "fuel_generated": game.stats["teamStats"][team]["fuelGenerated"],
            "opponent_fuel_generated": game.stats["teamStats"][opponent]["fuelGenerated"],
            "resource_fuel_collected": resource_fuel_collected,
            "opponent_resource_fuel_collected": opponent_resource_fuel_collected,
            "unit_snapshots": unit_snapshots,
        }

    def _growth_raw(self, current, previous):
        config = self._component_config("growth")
        fuel_divisor = max(config.get("fuel_divisor", 12000.0), 1.0)
        city_tile_divisor = max(config.get("city_tile_divisor", 5.0), 1.0)
        unit_divisor = max(config.get("unit_divisor", 8.0), 1.0)

        fuel_delta = current["fuel_generated"] - previous["fuel_generated"]
        city_fuel_delta = current["city_fuel"] - previous["city_fuel"]
        cargo_delta = current["cargo_fuel"] - previous["cargo_fuel"]
        city_tile_delta = current["city_tile_count"] - previous["city_tile_count"]
        worker_delta = current["worker_count"] - previous["worker_count"]
        cart_delta = current["cart_count"] - previous["cart_count"]

        raw = 0.0
        raw += config.get("fuel_generated", 0.0) * self._clip_unit(fuel_delta / fuel_divisor)
        raw += config.get("city_fuel", 0.0) * self._clip_unit(city_fuel_delta / fuel_divisor)
        raw += config.get("cargo_fuel", 0.0) * self._clip_unit(cargo_delta / fuel_divisor)
        raw += config.get("city_tile", 0.0) * self._clip_unit(city_tile_delta / city_tile_divisor)
        raw += config.get("worker", 0.0) * self._clip_unit(worker_delta / unit_divisor)
        raw += config.get("cart", 0.0) * self._clip_unit(cart_delta / unit_divisor)
        return raw

    def _survival_raw(self, current):
        config = self._component_config("survival")
        safe_target = max(config.get("safe_turns_target", 12.0), 1.0)
        city_tile_divisor = max(config.get("city_tile_divisor", 30.0), 1.0)
        unit_divisor = max(config.get("unit_divisor", 30.0), 1.0)

        city_safety = min(current["avg_city_survival_turns"] / safe_target, 1.0)
        city_scale = min(current["city_tile_count"] / city_tile_divisor, 1.0)
        unit_scale = min(current["unit_count"] / unit_divisor, 1.0)
        return (
            config.get("city_safety", 0.70) * city_safety * city_scale +
            config.get("unit_survival", 0.30) * unit_scale
        )

    def _unit_tendency_raw(self, current, previous):
        config = self._component_config("unit_tendency")
        divisor = max(config.get("progress_divisor", 12.0), 1.0)
        max_distance = 64.0
        progress = 0.0
        for unit_id, now in current["unit_snapshots"].items():
            before = previous["unit_snapshots"].get(unit_id)
            if before is None:
                continue
            if before["target_distance"] is None or now["target_distance"] is None:
                continue
            progress += (before["target_distance"] - now["target_distance"]) / max_distance
        return self._clip_unit(progress / divisor)

    def _city_tendency_raw(self):
        config = self._component_config("city_tendency")
        divisor = max(config.get("action_divisor", 8.0), 1.0)
        return self._clip_unit(sum(self.pending_city_tendency) / divisor)

    def _phase_tendency_raw(self, current):
        config = self._component_config("phase_tendency")
        move_part = min(self.pending_accepted_moves / max(config.get("move_divisor", 20.0), 1.0), 1.0)
        survival_part = self._survival_raw(current) * config.get("survival_scale", 0.60)
        if current["phase"] == "early":
            return move_part
        if current["phase"] == "mid":
            return 0.5 * move_part + 0.5 * survival_part
        return survival_part

    def _penalty_raw(self, current):
        config = self._component_config("penalty")
        divisor = max(config.get("divisor", 8.0), 1.0)
        no_progress = 0
        for event in self.pending_unit_actions:
            now = current["unit_snapshots"].get(event["unit_id"])
            if now is None:
                continue
            same_position = now["x"] == event["x"] and now["y"] == event["y"]
            same_cargo = now["cargo"] == event["cargo"]
            if same_position and same_cargo:
                no_progress += 1
        raw_penalty = (
            config.get("invalid_action", 1.0) * sum(self.pending_invalid_actions) +
            config.get("no_progress", 0.6) * no_progress
        )
        return -self._clip_unit(raw_penalty / divisor)

    def _full_cargo_penalty_raw(self, game):
        """
        满资源滞留惩罚：鼓励 worker 满载后尽快卸货，而不是原地停滞。

        宽限期 = UNIT_ACTION_COOLDOWN.WORKER（默认 2 轮），期间无惩罚，
        给 worker 足够时间走向城市/建城。超出宽限期后，按
            penalty = -base * excess ^ exponent
        非线性递增（exponent=1.5，增长介于线性与平方之间）。
        一旦 cargo 不满则重置计数，惩罚立即停止。
        """
        try:
            worker_cooldown = game.configs["parameters"]["UNIT_ACTION_COOLDOWN"]["WORKER"]
        except (KeyError, AttributeError):
            worker_cooldown = 2

        penalty_base     = 0.02
        penalty_exponent = 1.5
        cargo_capacity   = game.configs["parameters"]["RESOURCE_CAPACITY"]["WORKER"]

        current_unit_ids = set(game.state["teamStates"][self.team]["units"].keys())
        # 清理已消失 unit 的计数器
        for uid in list(self.unit_full_cargo_turns.keys()):
            if uid not in current_unit_ids:
                del self.unit_full_cargo_turns[uid]

        total_penalty = 0.0
        for unit in game.state["teamStates"][self.team]["units"].values():
            if unit.type != Constants.UNIT_TYPES.WORKER:
                continue

            cargo_total = unit.cargo["wood"] + unit.cargo["coal"] + unit.cargo["uranium"]
            if cargo_total >= cargo_capacity:
                self.unit_full_cargo_turns[unit.id] = self.unit_full_cargo_turns.get(unit.id, 0) + 1
                excess = self.unit_full_cargo_turns[unit.id] - worker_cooldown
                if excess > 0:
                    total_penalty -= penalty_base * (excess ** penalty_exponent)
            else:
                self.unit_full_cargo_turns[unit.id] = 0

        return total_penalty

    def _outcome_raw(self, game, current):
        config = self._component_config("outcome")
        city_divisor = max(config.get("city_tile_divisor", 30.0), 1.0)
        margin_divisor = max(config.get("city_tile_margin_divisor", 30.0), 1.0)
        winner, draw = self._deterministic_result(game)
        win_loss = 0.0
        if not draw:
            win_loss = 1.0 if winner == self.team else -1.0
        city_margin = current["city_tile_count"] - current["opponent_city_tile_count"]
        return (
            config.get("win_loss", 1.0) * win_loss +
            config.get("final_city_tile", 1.0) * self._clip_unit(current["city_tile_count"] / city_divisor) +
            config.get("city_tile_margin", 0.5) * self._clip_unit(city_margin / margin_divisor)
        )

    def _deterministic_result(self, game):
        city_tiles = {0: 0, 1: 0}
        for city in game.cities.values():
            city_tiles[city.team] += len(city.city_cells)
        if city_tiles[0] != city_tiles[1]:
            return (0 if city_tiles[0] > city_tiles[1] else 1), False

        units = {
            0: len(game.get_teams_units(0)),
            1: len(game.get_teams_units(1)),
        }
        if units[0] != units[1]:
            return (0 if units[0] > units[1] else 1), False

        fuel = {
            0: game.stats["teamStats"][0]["fuelGenerated"],
            1: game.stats["teamStats"][1]["fuelGenerated"],
        }
        if fuel[0] != fuel[1]:
            return (0 if fuel[0] > fuel[1] else 1), False
        return None, True
