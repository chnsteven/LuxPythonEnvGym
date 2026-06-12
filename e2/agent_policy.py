import sys
import time
import math
from functools import partial  # pip install functools
import copy
import random

import numpy as np
from gym import spaces

from luxai2021.env.agent import Agent, AgentWithModel
from luxai2021.game.actions import *
from luxai2021.game.game_constants import GAME_CONSTANTS
from luxai2021.game.position import Position


# https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
def closest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist_2)


def furthest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmax(dist_2)


_DIRECTION_MAPPING = {
    Constants.DIRECTIONS.CENTER: 0,
    Constants.DIRECTIONS.NORTH: 1,
    Constants.DIRECTIONS.WEST: 2,
    Constants.DIRECTIONS.SOUTH: 3,
    Constants.DIRECTIONS.EAST: 4,
}


def fill_direction_distance(obs, idx, pos, target_pos, max_distance):
    """写入 [dir_onehot×5, distance]，返回 idx+6"""
    direction = pos.direction_to(target_pos)
    obs[idx + _DIRECTION_MAPPING[direction]] = 1.0
    obs[idx + 5] = min(pos.distance_to(target_pos) / max_distance, 1.0)
    return idx + 6


def fill_empty_direction_distance(obs, idx):
    """无目标时 baseline 写法：distance 槽置 1.0"""
    obs[idx + 5] = 1.0
    return idx + 6


def smart_transfer_to_nearby(
    game, team, unit_id, unit, target_type_restriction=None, **kwarg
):
    """
    Smart-transfers from the specified unit to a nearby neighbor. Prioritizes any
    nearby carts first, then any worker. Transfers the resource type which the unit
    has most of. Picks which cart/worker based on choosing a target that is most-full
    but able to take the most amount of resources.

    Args:
        team ([type]): [description]
        unit_id ([type]): [description]

    Returns:
        Action: Returns a TransferAction object, even if the request is an invalid
                transfer. Use TransferAction.is_valid() to check validity.
    """

    # Calculate how much resources could at-most be transferred
    resource_type = None
    resource_amount = 0
    target_unit = None

    if unit != None:
        for type, amount in unit.cargo.items():
            if amount > resource_amount:
                resource_type = type
                resource_amount = amount

        # Find the best nearby unit to transfer to
        unit_cell = game.map.get_cell_by_pos(unit.pos)
        adjacent_cells = game.map.get_adjacent_cells(unit_cell)

        for c in adjacent_cells:
            for id, u in c.units.items():
                # Apply the unit type target restriction
                if target_type_restriction == None or u.type == target_type_restriction:
                    if u.team == team:
                        # This unit belongs to our team, set it as the winning transfer target
                        # if it's the best match.
                        if target_unit is None:
                            target_unit = u
                        else:
                            # Compare this unit to the existing target
                            if target_unit.type == u.type:
                                # Transfer to the target with the least capacity, but can accept
                                # all of our resources
                                if (
                                    u.get_cargo_space_left() >= resource_amount
                                    and target_unit.get_cargo_space_left()
                                    >= resource_amount
                                ):
                                    # Both units can accept all our resources. Prioritize one that is most-full.
                                    if (
                                        u.get_cargo_space_left()
                                        < target_unit.get_cargo_space_left()
                                    ):
                                        # This new target it better, it has less space left and can take all our
                                        # resources
                                        target_unit = u

                                elif (
                                    target_unit.get_cargo_space_left()
                                    >= resource_amount
                                ):
                                    # Don't change targets. Current one is best since it can take all
                                    # the resources, but new target can't.
                                    pass

                                elif (
                                    u.get_cargo_space_left()
                                    > target_unit.get_cargo_space_left()
                                ):
                                    # Change targets, because neither target can accept all our resources and
                                    # this target can take more resources.
                                    target_unit = u
                            elif u.type == Constants.UNIT_TYPES.CART:
                                # Transfer to this cart instead of the current worker target
                                target_unit = u

    # Build the transfer action request
    target_unit_id = None
    if target_unit is not None:
        target_unit_id = target_unit.id

        # Update the transfer amount based on the room of the target
        if target_unit.get_cargo_space_left() < resource_amount:
            resource_amount = target_unit.get_cargo_space_left()

    return TransferAction(team, unit_id, target_unit_id, resource_type, resource_amount)


########################################################################################################################
# This is the Agent that you need to design for the competition
########################################################################################################################
class AgentPolicy(AgentWithModel):
    def __init__(self, mode="train", model=None) -> None:
        """
        Arguments:
            mode: "train" or "inference", which controls if this agent is for training or not.
            model: The pretrained model, or if None it will operate in training mode.
        """
        super().__init__(mode, model)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.actions_units = [
            partial(
                MoveAction, direction=Constants.DIRECTIONS.CENTER
            ),  # This is the do-nothing action
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            partial(
                smart_transfer_to_nearby,
                target_type_restriction=Constants.UNIT_TYPES.CART,
            ),  # Transfer to nearby cart
            partial(
                smart_transfer_to_nearby,
                target_type_restriction=Constants.UNIT_TYPES.WORKER,
            ),  # Transfer to nearby worker
            SpawnCityAction,
            PillageAction,
        ]
        self.actions_cities = [
            SpawnWorkerAction,
            SpawnCartAction,
            ResearchAction,
        ]
        self.action_space = spaces.Discrete(
            max(len(self.actions_units), len(self.actions_cities))
        )

        # ═══════════════════════════════════════════════════════════════════════
        # 结构化观察空间设计（Structured Observation Space）
        # ═══════════════════════════════════════════════════════════════════════
        #
        # 1. 自身状态 (Self State) - 9 维
        #    [unit_type×3, pos×2, cargo×3, cargo_space_left]
        #
        # 2. 资源 per-type nearest - 21 维
        #    wood/coal/uranium 各 [dir×5, dist, val]
        #    val = 采集优先度，归一化到已研究最高等级=1.0
        #
        # 3. 友方单位 (Friendly Units) - Top-1 closest + Top-1 furthest × 9 = 18 维
        #    每个单位槽: [dir×5, dist, type_worker, type_cart, cargo_space_left]
        #
        # 4. 友方城市 (Friendly Cities) - Top-1 closest + Top-1 furthest × 8 = 16 维
        #    每个城市槽: [dir×5, dist, fuel_efficiency, fuel_survival]
        #
        # 5. 全局信息 (Global State) - 9 维
        #    [is_night, progress, til_night, til_day,
        #     workers, carts, e_workers, e_carts, research_pts]
        #
        # 6. 全局资源密度图 (Resource Density Map) - 4×4×3 = 48 维
        #
        # 7. 友方城市密度图 (City Density Map) - 4×4×1 = 16 维
        #
        # ═══════════════════════════════════════════════════════════════════════
        # 总维度: 9 + 21 + 18 + 16 + 9 + 48 + 16 = 137 维
        # ═══════════════════════════════════════════════════════════════════════
        self.observation_shape = (137,)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.observation_shape, dtype=np.float16
        )

        self.object_nodes = {}

    def get_agent_type(self):
        """
        Returns the type of agent. Use AGENT for inference, and LEARNING for training a model.
        """
        if self.mode == "train":
            return Constants.AGENT_TYPE.LEARNING
        else:
            return Constants.AGENT_TYPE.AGENT

    def get_initial_observation(self, game, unit, city_tile, team):
        # Called once per turn to precompute per-turn constants and object node arrays.

        # ── 归一化常数 ────────────────────────────────────────────────────────
        self.max_distance = (game.map.width - 1) + (game.map.height - 1)
        self.max_city_tiles = game.map.width * game.map.height
        self.cargo_capacity = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"][
            "WORKER"
        ]

        # ── 昼夜周期常量 ──────────────────────────────────────────────────────
        self.day_length = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"]  # 30
        self.night_length = GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]  # 10
        self.cycle_len = self.day_length + self.night_length
        self.cycle_pos = game.state["turn"] % self.cycle_len

        # ── 夜晚剩余回合（用于 fuel_survival 计算）──────────────────────────
        self.nights_left = (
            self.night_length
            if self.cycle_pos < self.day_length
            else (self.cycle_len - self.cycle_pos)
        )
        self.nights_left = max(self.nights_left, 1)

        # ── 友方 / 敌方 city tile 数量（用于 unit count 归一化）─────────────
        opponent_team = (team + 1) % 2
        self.friendly_city_tile_count = 0
        self.enemy_city_tile_count = 0
        for city in game.cities.values():
            if city.team == team:
                self.friendly_city_tile_count += len(city.city_cells)
            else:
                self.enemy_city_tile_count += len(city.city_cells)
        self.friendly_city_tile_count = max(self.friendly_city_tile_count, 1)
        self.enemy_city_tile_count = max(self.enemy_city_tile_count, 1)

        # ── Build object_nodes ────────────────────────────────────────────────
        self.object_nodes = {}

        for cell in game.map.resources:
            key = cell.resource.type
            node = np.array([[cell.pos.x, cell.pos.y]])
            if key not in self.object_nodes:
                self.object_nodes[key] = node
            else:
                self.object_nodes[key] = np.concatenate(
                    (self.object_nodes[key], node), axis=0
                )

        for t in [team, (team + 1) % 2]:
            for u in game.state["teamStates"][team]["units"].values():
                key = str(u.type) if t == team else str(u.type) + "_opponent"
                node = np.array([[u.pos.x, u.pos.y]])
                if key not in self.object_nodes:
                    self.object_nodes[key] = node
                else:
                    self.object_nodes[key] = np.concatenate(
                        (self.object_nodes[key], node), axis=0
                    )

        for city in game.cities.values():
            for cells in city.city_cells:
                key = "city" if city.team == team else "city_opponent"
                node = np.array([[cells.pos.x, cells.pos.y]])
                if key not in self.object_nodes:
                    self.object_nodes[key] = node
                else:
                    self.object_nodes[key] = np.concatenate(
                        (self.object_nodes[key], node), axis=0
                    )

        # ── 4×4 全局资源密度桶 ────────────────────────────────────────────────
        # 将地图均分为 4×4 个区域，统计每个区域内各资源类型的格子数量。
        # 地图尺寸不一定能被 4 整除，使用浮点桶宽度并向下取整定位。
        #
        # counts[row][col][r] 归一化密度
        #   row, col ∈ [0, 3]
        #   r: 0=wood, 1=coal, 2=uranium
        #   归一化分母 = ceil(bucket_w) × ceil(bucket_h)，即该桶最大可能格子数
        #
        # 研究进度 masking：
        #   wood    (r=0): 永远可见
        #   coal    (r=1): 仅 researched["coal"]    == True 时可见，否则置 0
        #   uranium (r=2): 仅 researched["uranium"] == True 时可见，否则置 0
        #
        # 最终展开为 48 维向量，在 get_observation 末尾追加。
        # ─────────────────────────────────────────────────────────────────────
        N_BUCKETS = 4
        bucket_w = game.map.width / N_BUCKETS  # 每桶宽（浮点）
        bucket_h = game.map.height / N_BUCKETS  # 每桶高（浮点）
        # 每桶最大格子数（用于归一化，各桶面积相同）
        max_cells_per_bucket = math.ceil(bucket_w) * math.ceil(bucket_h)

        # counts[row][col][resource_idx]
        counts = np.zeros((N_BUCKETS, N_BUCKETS, 3), dtype=np.float32)
        researched_coal = game.state["teamStates"][team]["researched"]["coal"]
        researched_uranium = game.state["teamStates"][team]["researched"]["uranium"]
        rtype_to_idx = {
            Constants.RESOURCE_TYPES.WOOD: 0,
            Constants.RESOURCE_TYPES.COAL: 1,
            Constants.RESOURCE_TYPES.URANIUM: 2,
        }
        for cell in game.map.resources:
            r_idx = rtype_to_idx.get(cell.resource.type)
            if r_idx is None:
                continue
            # masking：未研究的资源对 agent 不可见，跳过统计
            if r_idx == 1 and not researched_coal:
                continue
            if r_idx == 2 and not researched_uranium:
                continue
            col = min(int(cell.pos.x / bucket_w), N_BUCKETS - 1)
            row = min(int(cell.pos.y / bucket_h), N_BUCKETS - 1)
            counts[row, col, r_idx] += 1.0

        # 归一化并存储为展开的 48 维向量
        self.resource_density_obs = (counts / max_cells_per_bucket).reshape(
            -1
        )  # shape: (48,)

        # ── 4×4 友方城市密度桶 ────────────────────────────────────────────────
        city_counts = np.zeros((N_BUCKETS, N_BUCKETS), dtype=np.float32)
        for city in game.cities.values():
            if city.team != team:
                continue
            for cell in city.city_cells:
                col = min(int(cell.pos.x / bucket_w), N_BUCKETS - 1)
                row = min(int(cell.pos.y / bucket_h), N_BUCKETS - 1)
                city_counts[row, col] += 1.0
        self.city_density_obs = (city_counts / max_cells_per_bucket).reshape(-1)

    def get_observation(self, game, unit, city_tile, team, is_new_turn):
        """
        结构化观察空间 (137维)
        1. 自身状态                 9维  [unit_type×3, pos×2, cargo×3, cargo_space_left]
        2. 资源 per-type nearest    21维  [dir×5, dist, val] × 3
        3. 友方单位 Top-1c+1f      18维  [dir×5, dist, type×2, cargo_left] × 2
        4. 友方城市 Top-1c+1f      16维  [dir×5, dist, fuel_eff, fuel_surv] × 2
        5. 全局信息                 9维  [is_night, progress, til_night, til_day,
                                         workers, carts, e_workers, e_carts, research_pts]
        6. 全局资源密度图 4×4×3    48维
        7. 友方城市密度图 4×4×1    16维
        """
        if is_new_turn:
            self.get_initial_observation(game, unit, city_tile, team)

        obs = np.zeros(self.observation_shape, dtype=np.float32)
        idx = 0

        pos = (
            unit.pos
            if unit is not None
            else (city_tile.pos if city_tile is not None else None)
        )
        pos_arr = np.array([pos.x, pos.y]) if pos is not None else None

        # ── 1. 自身状态 (9维) ─────────────────────────────────────────────────
        if unit is not None:
            obs[idx] = 1.0 if unit.type == Constants.UNIT_TYPES.WORKER else 0.0
            obs[idx + 1] = 1.0 if unit.type == Constants.UNIT_TYPES.CART else 0.0
        elif city_tile is not None:
            obs[idx + 2] = 1.0
        idx += 3

        if pos is not None:
            obs[idx] = pos.x / (game.map.width - 1)
            obs[idx + 1] = pos.y / (game.map.height - 1)
        idx += 2

        if unit is not None:
            cap = self.cargo_capacity
            obs[idx] = unit.cargo["wood"] / cap
            obs[idx + 1] = unit.cargo["coal"] / cap
            obs[idx + 2] = unit.cargo["uranium"] / cap
        idx += 3

        if unit is not None:
            obs[idx] = unit.get_cargo_space_left() / self.cargo_capacity
        idx += 1

        # ── 2. 资源 per-type nearest (21维) ───────────────────────────────────
        if pos is not None and pos_arr is not None:
            researched_coal = game.state["teamStates"][team]["researched"]["coal"]
            researched_uranium = game.state["teamStates"][team]["researched"]["uranium"]
            if researched_uranium:
                resource_val = {
                    Constants.RESOURCE_TYPES.URANIUM: 1.0,
                    Constants.RESOURCE_TYPES.COAL: 0.1,
                    Constants.RESOURCE_TYPES.WOOD: 0.025,
                }
            elif researched_coal:
                resource_val = {
                    Constants.RESOURCE_TYPES.COAL: 1.0,
                    Constants.RESOURCE_TYPES.WOOD: 0.1,
                    Constants.RESOURCE_TYPES.URANIUM: 0.0,
                }
            else:
                resource_val = {
                    Constants.RESOURCE_TYPES.WOOD: 1.0,
                    Constants.RESOURCE_TYPES.COAL: 0.0,
                    Constants.RESOURCE_TYPES.URANIUM: 0.0,
                }

            for rtype in [
                Constants.RESOURCE_TYPES.WOOD,
                Constants.RESOURCE_TYPES.COAL,
                Constants.RESOURCE_TYPES.URANIUM,
            ]:
                if rtype == Constants.RESOURCE_TYPES.COAL and not researched_coal:
                    idx += 7
                    continue
                if (
                    rtype == Constants.RESOURCE_TYPES.URANIUM
                    and not researched_uranium
                ):
                    idx += 7
                    continue
                if rtype not in self.object_nodes or len(self.object_nodes[rtype]) == 0:
                    idx = fill_empty_direction_distance(obs, idx)
                    idx += 1
                    continue
                nodes = self.object_nodes[rtype]
                ci = closest_node(pos_arr, nodes)
                closest = nodes[ci]
                target_pos = Position(closest[0], closest[1])
                idx = fill_direction_distance(
                    obs, idx, pos, target_pos, self.max_distance
                )
                obs[idx] = resource_val.get(rtype, 0.0)
                idx += 1
        else:
            idx += 21

        # ── 3. 友方单位 Top-1 closest + Top-1 furthest (18维) ────────────────
        if pos is not None and pos_arr is not None:
            friendly_nodes = []
            friendly_utypes = []
            friendly_cargo_lefts = []
            for utype in [Constants.UNIT_TYPES.WORKER, Constants.UNIT_TYPES.CART]:
                key = str(utype)
                if key not in self.object_nodes:
                    continue
                nodes = self.object_nodes[key]
                for node in nodes:
                    if unit is not None and node[0] == pos.x and node[1] == pos.y:
                        continue
                    node_pos = Position(node[0], node[1])
                    cell = game.map.get_cell_by_pos(node_pos)
                    cargo_left = (
                        next(iter(cell.units.values())).get_cargo_space_left()
                        if cell.units
                        else self.cargo_capacity
                    )
                    friendly_nodes.append(node)
                    friendly_utypes.append(utype)
                    friendly_cargo_lefts.append(cargo_left)

            def fill_unit_slot(i, start_idx):
                node = friendly_nodes[i]
                target_pos = Position(node[0], node[1])
                next_idx = fill_direction_distance(
                    obs, start_idx, pos, target_pos, self.max_distance
                )
                obs[next_idx] = (
                    1.0
                    if friendly_utypes[i] == Constants.UNIT_TYPES.WORKER
                    else 0.0
                )
                obs[next_idx + 1] = (
                    1.0
                    if friendly_utypes[i] == Constants.UNIT_TYPES.CART
                    else 0.0
                )
                obs[next_idx + 2] = (
                    friendly_cargo_lefts[i] / self.cargo_capacity
                )
                return next_idx + 3

            if friendly_nodes:
                nodes_arr = np.array(friendly_nodes)
                ci = closest_node(pos_arr, nodes_arr)
                fi = furthest_node(pos_arr, nodes_arr)
                idx = fill_unit_slot(ci, idx)
                idx = fill_unit_slot(fi, idx)
            else:
                idx = fill_empty_direction_distance(obs, idx)
                idx += 3
                idx = fill_empty_direction_distance(obs, idx)
                idx += 3
        else:
            idx += 18

        # ── 4. 友方城市 Top-1 closest + Top-1 furthest (16维) ────────────────
        if pos is not None and pos_arr is not None:
            city_data = []
            for city in game.cities.values():
                if city.team != team:
                    continue
                tile_coords = np.array([[c.pos.x, c.pos.y] for c in city.city_cells])
                ci = closest_node(pos_arr, tile_coords)
                closest_cell_pos = Position(tile_coords[ci][0], tile_coords[ci][1])

                upkeep = city.get_light_upkeep()
                default_upkeep = (
                    len(city.city_cells)
                    * GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["CITY"]
                )
                fuel_eff = upkeep / default_upkeep if default_upkeep > 0 else 1.0
                fuel_surv = (
                    min(city.fuel / (upkeep * self.nights_left), 1.0)
                    if upkeep > 0
                    else 1.0
                )
                city_data.append((fuel_eff, fuel_surv, closest_cell_pos))

            def fill_city_slot(item, start_idx):
                fuel_eff, fuel_surv, cell_pos = item
                next_idx = fill_direction_distance(
                    obs, start_idx, pos, cell_pos, self.max_distance
                )
                obs[next_idx] = fuel_eff
                obs[next_idx + 1] = fuel_surv
                return next_idx + 2

            if city_data:
                city_positions = np.array(
                    [[c[2].x, c[2].y] for c in city_data]
                )
                ci = closest_node(pos_arr, city_positions)
                fi = furthest_node(pos_arr, city_positions)
                idx = fill_city_slot(city_data[ci], idx)
                idx = fill_city_slot(city_data[fi], idx)
            else:
                idx = fill_empty_direction_distance(obs, idx)
                idx += 2
                idx = fill_empty_direction_distance(obs, idx)
                idx += 2
        else:
            idx += 16

        # ── 7. 全局信息 (11维) ────────────────────────────────────────────────
        obs[idx] = float(game.is_night())
        idx += 1
        obs[idx] = game.state["turn"] / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
        idx += 1

        turns_until_night = max(0, self.day_length - self.cycle_pos)
        obs[idx] = turns_until_night / self.day_length
        idx += 1

        turns_until_day = (
            max(0, self.cycle_len - self.cycle_pos)
            if self.cycle_pos >= self.day_length
            else 0
        )
        obs[idx] = turns_until_day / self.night_length
        idx += 1

        # 友方单位数量：归一化用友方 city tile 数量
        worker_count = len(self.object_nodes.get(str(Constants.UNIT_TYPES.WORKER), []))
        cart_count = len(self.object_nodes.get(str(Constants.UNIT_TYPES.CART), []))
        obs[idx] = worker_count / self.friendly_city_tile_count
        obs[idx + 1] = cart_count / self.friendly_city_tile_count
        idx += 2

        # 敌方单位数量：归一化用敌方 city tile 数量
        e_worker = len(
            self.object_nodes.get(str(Constants.UNIT_TYPES.WORKER) + "_opponent", [])
        )
        e_cart = len(
            self.object_nodes.get(str(Constants.UNIT_TYPES.CART) + "_opponent", [])
        )
        obs[idx] = e_worker / self.enemy_city_tile_count
        obs[idx + 1] = e_cart / self.enemy_city_tile_count
        idx += 2

        obs[idx] = min(
            game.state["teamStates"][team]["researchPoints"]
            / GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["URANIUM"],
            1.0,
        )
        idx += 1

        # ── 6. 全局资源密度图 4×4×3 = 48 维 ──────────────────────────────────
        obs[idx : idx + 48] = self.resource_density_obs
        idx += 48

        # ── 7. 友方城市密度图 4×4×1 = 16 维 ──────────────────────────────────
        obs[idx : idx + 16] = self.city_density_obs
        idx += 16

        assert idx == 137
        return obs

    def get_action_mask(self, game, unit=None, city_tile=None):
        """
        Computes a boolean action mask for the given unit or city tile.

        Returns a numpy bool array of length == number of actions in the
        relevant action list (actions_units or actions_cities).
        True  = action is ALLOWED
        False = action is FORBIDDEN

        Action index reference
        ──────────────────────
        actions_units  (unit is not None):
            0  MoveAction CENTER   (do-nothing / stay)
            1  MoveAction NORTH
            2  MoveAction WEST
            3  MoveAction SOUTH
            4  MoveAction EAST
            5  TransferAction → cart
            6  TransferAction → worker
            7  SpawnCityAction
            8  PillageAction

        actions_cities  (city_tile is not None):
            0  SpawnWorkerAction
            1  SpawnCartAction
            2  ResearchAction

        All masks are hard constraints (always wrong regardless of game
        state) and are completely independent of the reward function.
        """

        # ── CITY TILE MASKS ───────────────────────────────────────────────────
        if city_tile is not None:
            n = len(self.actions_cities)
            mask = np.ones(n, dtype=bool)

            # ── Mask: city tile in cooldown → all actions forbidden ───────────
            if not city_tile.can_act():
                return np.zeros(n, dtype=bool)

            # ── Mask: research already maxed (researchPoints >= URANIUM threshold) ──
            uranium_req = GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"][
                "URANIUM"
            ]
            if game.state["teamStates"][self.team]["researchPoints"] >= uranium_req:
                mask[2] = False  # ResearchAction — nothing left to research

            # ── Mask: unit count already at city tile cap → no spawning ───────
            # Each city tile can support one unit; spawning beyond cap is wasteful.
            city_tile_count = sum(
                len(city.city_cells)
                for city in game.cities.values()
                if city.team == self.team
            )
            unit_count = len(game.state["teamStates"][self.team]["units"])
            if unit_count >= city_tile_count:
                mask[0] = False  # SpawnWorkerAction
                mask[1] = False  # SpawnCartAction

            # ── Mask: no cart without a worker ───────────────────────────────
            worker_count = sum(
                1
                for u in game.state["teamStates"][self.team]["units"].values()
                if u.type == Constants.UNIT_TYPES.WORKER
            )
            if worker_count == 0:
                mask[1] = False  # SpawnCartAction

            return mask

        # ── UNIT MASKS ────────────────────────────────────────────────────────
        else:
            n = len(self.actions_units)
            mask = np.ones(n, dtype=bool)

            mask[0], mask[8] = False, False

            if unit is None:
                return mask

            # ── Mask: unit in cooldown ─────
            if not unit.can_act():
                return np.zeros(n, dtype=bool)

            pos = unit.pos

            # ── Mask 1: map boundary — forbid moves that leave the map ────────
            # Indices 1-4 correspond to NORTH / WEST / SOUTH / EAST.
            direction_checks = [
                (1, 0, -1),  # NORTH: y - 1
                (2, -1, 0),  # WEST:  x - 1
                (3, 0, +1),  # SOUTH: y + 1
                (4, +1, 0),  # EAST:  x + 1
            ]
            for idx, dx, dy in direction_checks:
                nx, ny = pos.x + dx, pos.y + dy
                if nx < 0 or ny < 0 or nx >= game.map.width or ny >= game.map.height:
                    mask[idx] = False

            # ── Mask 2: SpawnCity conditions not met ──────────────────────────
            # SpawnCityAction (index 7) requires:
            #   a) unit is a worker
            #   b) current cell has no resource
            #   c) current cell is not already a city tile
            #   d) cargo total >= CITY_BUILD_COST (100)
            build_cost = GAME_CONSTANTS["PARAMETERS"]["CITY_BUILD_COST"]
            if unit.type != Constants.UNIT_TYPES.WORKER:
                mask[7] = False
            else:
                cell = game.map.get_cell_by_pos(pos)
                cargo_total = (
                    unit.cargo["wood"] + unit.cargo["coal"] + unit.cargo["uranium"]
                )
                if (
                    cell.has_resource()
                    or cell.is_city_tile()
                    or cargo_total < build_cost
                ):
                    mask[7] = False

            # ── Mask 3: transfer — no valid adjacent friendly unit ────────────
            # TransferAction→cart (index 5) and TransferAction→worker (index 6)
            # are only meaningful when there is a friendly unit of the matching
            # type in an adjacent cell.
            unit_cell = game.map.get_cell_by_pos(pos)
            adjacent_cells = game.map.get_adjacent_cells(unit_cell)

            has_adjacent_cart = False
            has_adjacent_worker = False
            for c in adjacent_cells:
                for u in c.units.values():
                    if u.team == self.team:
                        if u.type == Constants.UNIT_TYPES.CART:
                            has_adjacent_cart = True
                        elif u.type == Constants.UNIT_TYPES.WORKER:
                            has_adjacent_worker = True

            if not has_adjacent_cart:
                mask[5] = False  # TransferAction → cart
            if not has_adjacent_worker:
                mask[6] = False  # TransferAction → worker

            return mask

    def get_padded_action_mask(self, game, unit=None, city_tile=None):
        """
        Return an action mask padded to action_space.n for MaskablePPO.
        When no real action is valid (e.g. city tile in cooldown), index 0 is
        marked valid as a placeholder; take_action() still skips execution.
        """
        mask = self.get_action_mask(game, unit=unit, city_tile=city_tile)
        n = self.action_space.n
        if len(mask) < n:
            padded = np.zeros(n, dtype=bool)
            padded[: len(mask)] = mask
            mask = padded
        if not np.any(mask):
            placeholder = np.zeros(n, dtype=bool)
            placeholder[0] = True
            return placeholder
        return mask

    def action_code_to_action(
        self, action_code, game, unit=None, city_tile=None, team=None
    ):
        """
        Takes an action in the environment according to actionCode:
            action_code: Index of action to take into the action array.
        Returns: An action.
        """
        # Map action_code index into to a constructed Action object
        try:
            x = None
            y = None
            if city_tile is not None:
                x = city_tile.pos.x
                y = city_tile.pos.y
            elif unit is not None:
                x = unit.pos.x
                y = unit.pos.y

            if city_tile != None:
                action = self.actions_cities[action_code % len(self.actions_cities)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y,
                )
            else:
                action = self.actions_units[action_code % len(self.actions_units)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y,
                )

            return action
        except Exception as e:
            # Not a valid action
            print(e)
            return None

    def take_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode.
        Invalid actions are skipped; MaskablePPO should only sample valid ones.
        """
        mask = self.get_action_mask(game, unit=unit, city_tile=city_tile)
        if not np.any(mask):
            return

        n = len(mask)
        if not mask[action_code % n]:
            return

        action = self.action_code_to_action(action_code, game, unit, city_tile, team)
        self.match_controller.take_action(action)

    def process_turn(self, game, team):
        """Inference with action masks (MaskablePPO)."""
        start_time = time.time()
        actions = []
        new_turn = True

        units = game.state["teamStates"][team]["units"].values()
        for unit in units:
            if unit.can_act():
                obs = self.get_observation(game, unit, None, unit.team, new_turn)
                mask = self.get_padded_action_mask(game, unit=unit)
                action_code, _states = self.model.predict(
                    obs, action_masks=mask, deterministic=False
                )
                real_mask = self.get_action_mask(game, unit=unit)
                if (
                    action_code is not None
                    and np.any(real_mask)
                    and real_mask[action_code % len(real_mask)]
                ):
                    actions.append(
                        self.action_code_to_action(
                            action_code,
                            game=game,
                            unit=unit,
                            city_tile=None,
                            team=unit.team,
                        )
                    )
                new_turn = False

        cities = game.cities.values()
        for city in cities:
            if city.team == team:
                for cell in city.city_cells:
                    city_tile = cell.city_tile
                    if city_tile.can_act():
                        obs = self.get_observation(
                            game, None, city_tile, city.team, new_turn
                        )
                        mask = self.get_padded_action_mask(game, city_tile=city_tile)
                        action_code, _states = self.model.predict(
                            obs, action_masks=mask, deterministic=False
                        )
                        real_mask = self.get_action_mask(game, city_tile=city_tile)
                        if (
                            action_code is not None
                            and np.any(real_mask)
                            and real_mask[action_code % len(real_mask)]
                        ):
                            actions.append(
                                self.action_code_to_action(
                                    action_code,
                                    game=game,
                                    unit=None,
                                    city_tile=city_tile,
                                    team=city.team,
                                )
                            )
                        new_turn = False

        time_taken = time.time() - start_time
        if time_taken > 0.5:
            print(
                "WARNING: Inference took %.3f seconds for computing actions. Limit is 1 second."
                % time_taken,
                file=sys.stderr,
            )

        return actions

    def game_start(self, game):
        """
        This function is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.

        Args:
            game ([type]): Game.
        """
        self.city_tiles_last = 0
        self.fuel_collected_last = 0
        # upkeep 效率追踪：记录上一回合城市总 fuel，用于计算净消耗
        self.city_fuel_last = 0
        # cooldown 追踪：记录上一回合各单位的 cooldown 值，用于检测溢出
        self.unit_cooldowns_last = {}  # {unit_id: cooldown}
        # research 状态追踪：记录上一回合的解锁状态，用于检测新资源解锁
        self.researched_coal_last = False
        self.researched_uranium_last = (
            False  # 生存追踪：记录上一回合的单位和城市数量，用于计算生存奖励
        )
        self.workers_last = 0
        self.carts_last = 0
        self.cities_last = 0  # 城市数量（不是 tile 数量）
        # 同 cargo 滞留：{unit_id: (wood, coal, uranium)} 与连续相同回合数
        self.unit_cargo_last = {}
        self.unit_same_cargo_turns = {}
        # 运 cart 奖励：友方 cart 总载货量
        self.cart_cargo_sum_last = 0
        # cart 装载持续回合（用于指数奖励）
        self.cart_fill_turns = {}

        if self.mode == "train":
            self._heuristic_games_played = (
                getattr(self, "_heuristic_games_played", 0) + 1
            )
            decay_scale = getattr(self, "heuristic_decay_scale", 6000)
            self.heuristic_prob = math.exp(-self._heuristic_games_played / decay_scale)
        else:
            # 推理/评估时完全关闭 heuristic，让 RL 策略独立运行
            self.heuristic_prob = 0.0

    def get_reward(self, game, is_game_finished, is_new_turn, is_game_error):
        """
        Returns the reward function for this step of the game. Reward should be a
        delta increment to the reward, not the total current reward.
        """
        if is_game_error:
            # Game environment step failed, assign a game lost reward to not incentivise this
            print("Game failed due to error")
            return -1.0

        if not is_new_turn and not is_game_finished:
            # Only apply rewards at the start of each turn or at game end
            return 0

        # Get some basic stats
        unit_count = len(game.state["teamStates"][self.team]["units"])

        city_count = 0
        city_count_opponent = 0
        city_tile_count = 0
        city_tile_count_opponent = 0
        for city in game.cities.values():
            if city.team == self.team:
                city_count += 1
            else:
                city_count_opponent += 1

            for cell in city.city_cells:
                if city.team == self.team:
                    city_tile_count += 1
                else:
                    city_tile_count_opponent += 1

        # 统计单位类型数量
        worker_count = 0
        cart_count = 0
        for unit in game.state["teamStates"][self.team]["units"].values():
            if unit.type == Constants.UNIT_TYPES.WORKER:
                worker_count += 1
            else:
                cart_count += 1

        rewards = {}

        # ═══════════════════════════════════════════════════════════════════════
        # 城市扩张和单位创建
        # ═══════════════════════════════════════════════════════════════════════

        rewards["rew/r_workers"] = (worker_count - self.workers_last) * 0.05
        rewards["rew/r_carts"] = (cart_count - self.carts_last) * 0.15
        self.workers_last = worker_count
        self.carts_last = cart_count

        # Give a reward for city creation/death. 0.1 reward per city.
        rewards["rew/r_city_tiles"] = (city_tile_count - self.city_tiles_last) * 0.1
        self.city_tiles_last = city_tile_count

        # 每轮持续奖励：鼓励维持更多单位和城市
        rewards["rew/r_units_alive"] = unit_count * 0.01
        rewards["rew/r_city_tiles_alive"] = city_tile_count * 0.02

        cart_capacity = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]

        # 运到 city：燃料增量（worker 在城格卸货）
        fuel_now = game.stats["teamStats"][self.team]["fuelGenerated"]
        fuel_delta = fuel_now - self.fuel_collected_last
        rewards["rew/r_to_city"] = fuel_delta / 5000
        self.fuel_collected_last = fuel_now

        # 运到 cart：友方 cart 总载货量增加量
        cart_cargo_sum = 0
        for unit in game.state["teamStates"][self.team]["units"].values():
            if unit.type == Constants.UNIT_TYPES.CART:
                cart_cargo_sum += (
                    unit.cargo["wood"] + unit.cargo["coal"] + unit.cargo["uranium"]
                )
        cart_cargo_delta = cart_cargo_sum - self.cart_cargo_sum_last
        rewards["rew/r_to_cart"] = max(0, cart_cargo_delta) / cart_capacity
        self.cart_cargo_sum_last = cart_cargo_sum

        # # cart 装载比例奖励：有货时 streak+1，指数增长但封顶（避免 1.5^N 爆炸）
        # cart_fill_reward = 0.0
        # cart_fill_base = 0.01
        # cart_fill_growth = 1.5
        # max_exp_streak = 12
        # current_cart_ids = set()
        # for unit in game.state["teamStates"][self.team]["units"].values():
        #     if unit.type != Constants.UNIT_TYPES.CART:
        #         continue
        #     current_cart_ids.add(unit.id)
        #     cargo_total = (
        #         unit.cargo["wood"] + unit.cargo["coal"] + unit.cargo["uranium"]
        #     )
        #     fill_ratio = cargo_total / cart_capacity
        #     if fill_ratio > 0:
        #         self.cart_fill_turns[unit.id] = self.cart_fill_turns.get(unit.id, 0) + 1
        #         streak = self.cart_fill_turns[unit.id]
        #         exp_streak = min(max(streak - 1, 0), max_exp_streak)
        #         cart_fill_reward += (
        #             fill_ratio * cart_fill_base * (cart_fill_growth**exp_streak)
        #         )
        #     else:
        #         self.cart_fill_turns[unit.id] = 0
        # for uid in list(self.cart_fill_turns.keys()):
        #     if uid not in current_cart_ids:
        #         del self.cart_fill_turns[uid]
        # rewards["rew/r_cart_fill"] = cart_fill_reward

        # ═══════════════════════════════════════════════════════════════════════
        # 研究解锁一次性奖励
        # ═══════════════════════════════════════════════════════════════════════
        # coal 解锁（研究点数首次 >= COAL 阈值）和 uranium 解锁（首次 >= 200）
        # 各给一次固定奖励，仅在解锁发生的那一回合触发。
        research_pts = game.state["teamStates"][self.team]["researchPoints"]
        coal_req = GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["COAL"]
        uranium_req = GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["URANIUM"]

        researched_coal = game.state["teamStates"][self.team]["researched"][
            Constants.RESOURCE_TYPES.COAL
        ]
        researched_uranium = game.state["teamStates"][self.team]["researched"][
            Constants.RESOURCE_TYPES.URANIUM
        ]

        # coal 解锁：上一回合未解锁，本回合解锁
        if researched_coal and not self.researched_coal_last:
            rewards["rew/r_unlock_coal"] = 2.0
        else:
            rewards["rew/r_unlock_coal"] = 0.0
        self.researched_coal_last = researched_coal

        # uranium 解锁：研究点数首次达到 200（等价于 researched["uranium"] 刚变 True）
        if researched_uranium and not self.researched_uranium_last:
            rewards["rew/r_unlock_uranium"] = 5.0
        else:
            rewards["rew/r_unlock_uranium"] = 0.0
        self.researched_uranium_last = researched_uranium

        # ═══════════════════════════════════════════════════════════════════════
        # 采集质量奖励：鼓励 unit 采集与当前研究等级匹配的资源
        # ═══════════════════════════════════════════════════════════════════════
        #
        # 仅在 unit cargo 未满时触发（满了应该去建城，不给此奖励）。
        # 根据本回合 cargo 净增量中比例最高的资源类型，给予对应奖励系数
        # 如果当前采集的资源不是已研究的最高等级，每少一级减半。

        cargo_quality_reward = 0.0
        base_quality_reward = 0.2
        cargo_capacity = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]

        # 确定当前已研究的最高资源等级
        # uranium > coal > wood（每少一级奖励减半）

        if researched_uranium:
            tier_multipliers = {
                "uranium": base_quality_reward,
                "coal": base_quality_reward / 4,
                "wood": base_quality_reward / 8,
            }
        elif researched_coal:
            tier_multipliers = {
                "uranium": 0.0,
                "coal": base_quality_reward,
                "wood": base_quality_reward / 4,
            }
        else:
            tier_multipliers = {
                "uranium": 0.0,
                "coal": 0.0,
                "wood": base_quality_reward,
            }

        for unit in game.state["teamStates"][self.team]["units"].values():
            cargo = unit.cargo  # {"wood": int, "coal": int, "uranium": int}
            cargo_total = cargo["wood"] + cargo["coal"] + cargo["uranium"]

            # cargo 已满则跳过
            if cargo_total >= cargo_capacity:
                continue

            # 用 unit 当前所在格的资源类型判断正在采集什么
            cell = game.map.get_cell_by_pos(unit.pos)
            if cell.has_resource():
                resource_type = cell.resource.type  # "wood" / "coal" / "uranium"
                multiplier = tier_multipliers.get(resource_type, 0.0)
                if multiplier > 0.0:
                    cargo_quality_reward += multiplier

        rewards["rew/r_cargo_quality"] = cargo_quality_reward

        # # ═══════════════════════════════════════════════════════════════════════
        # # 同 cargo 滞留惩罚：cargo 连续多回合完全不变时施加指数惩罚
        # # 例：wood=40 连续 4 回合 → 第 1 回合无惩罚，第 2 回合起指数递增
        # # ═══════════════════════════════════════════════════════════════════════
        # same_cargo_penalty = 0.0
        # penalty_base = 0.01
        # penalty_growth = 1.5
        # same_cargo_grace = 1
        # max_exp_streak = 12

        # current_unit_ids = set(game.state["teamStates"][self.team]["units"].keys())
        # for uid in list(self.unit_cargo_last.keys()):
        #     if uid not in current_unit_ids:
        #         self.unit_cargo_last.pop(uid, None)
        #         self.unit_same_cargo_turns.pop(uid, None)

        # for unit in game.state["teamStates"][self.team]["units"].values():
        #     if unit.type != Constants.UNIT_TYPES.WORKER:
        #         continue

        #     cargo_t = (unit.cargo["wood"], unit.cargo["coal"], unit.cargo["uranium"])
        #     if sum(cargo_t) == 0:
        #         self.unit_same_cargo_turns[unit.id] = 0
        #         self.unit_cargo_last[unit.id] = cargo_t
        #         continue

        #     if self.unit_cargo_last.get(unit.id) == cargo_t:
        #         self.unit_same_cargo_turns[unit.id] = (
        #             self.unit_same_cargo_turns.get(unit.id, 0) + 1
        #         )
        #     else:
        #         self.unit_same_cargo_turns[unit.id] = 1
        #     self.unit_cargo_last[unit.id] = cargo_t

        #     streak = self.unit_same_cargo_turns[unit.id]
        #     if streak > same_cargo_grace:
        #         excess = min(streak - same_cargo_grace, max_exp_streak)
        #         same_cargo_penalty -= penalty_base * (penalty_growth**excess)

        # rewards["rew/r_same_cargo_penalty"] = same_cargo_penalty

        # ═══════════════════════════════════════════════════════════════════════
        # 游戏结束奖励
        # ═══════════════════════════════════════════════════════════════════════

        # Give a reward of 1.0 per city tile alive at the end of the game
        if is_game_finished:
            self.is_last_turn = True
            if city_tile_count > 0:
                rewards["rew/r_survival"] = (
                    worker_count + cart_count
                ) * 0.5 + city_tile_count
            else:
                rewards["rew/r_survival"] = -1.0  # Game lost

            """
            # Example of a game win/loss reward instead
            if game.get_winning_team() == self.team:
                rewards["rew/r_game_win"] = 100.0 # Win
            else:
                rewards["rew/r_game_win"] = -100.0 # Loss
            """

        reward = 0
        for name, value in rewards.items():
            reward += value

        return reward

    def research_heuristic(self, game, unit_threshold=1.0):
        """
        Heuristic: if current unit count (workers + carts) exceeds
        (city_tile_capacity * unit_threshold), all idle city tiles issue a
        ResearchAction instead of spawning more units.

        If the highest research tier (uranium) is already unlocked, there is
        nothing left to research, so the heuristic returns immediately and
        lets RL decide what to do with idle city tiles.
        """
        # Nothing left to research if uranium is already unlocked
        if game.state["teamStates"][self.team]["researched"][
            Constants.RESOURCE_TYPES.URANIUM
        ]:
            return

        # --- compute capacity and current unit count for our team ---
        city_tile_count = 0
        for city in game.cities.values():
            if city.team == self.team:
                city_tile_count += len(city.city_cells)

        if city_tile_count == 0:
            return  # no cities yet, nothing to do

        unit_count = len(game.state["teamStates"][self.team]["units"])
        unit_pct = unit_count / city_tile_count  # 0.0 → 1.0+

        if unit_pct < unit_threshold:
            return  # below threshold → let RL decide (spawn or research)

        # --- above threshold: force all idle city tiles to research ---
        for city in game.cities.values():
            if city.team != self.team:
                continue
            for cell in city.city_cells:
                city_tile = cell.city_tile
                if not city_tile.can_act():
                    continue
                action = ResearchAction(
                    team=self.team,
                    x=city_tile.pos.x,
                    y=city_tile.pos.y,
                    unit_id=None,
                )
                self.match_controller.take_action(action)

    def turn_heurstics(self, game, is_first_turn):
        """
        This is called pre-observation actions to allow for hardcoded heuristics
        to control a subset of units. Any unit or city that gets an action from this
        callback, will not create an observation+action.

        Args:
            game ([type]): Game in progress
            is_first_turn (bool): True if it's the first turn of a game.
        """
        prob = getattr(self, "heuristic_prob", 1.0)

        # if random.random() < prob:
        self.research_heuristic(game)
        return
