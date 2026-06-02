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

def smart_transfer_to_nearby(game, team, unit_id, unit, target_type_restriction=None, **kwarg):
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
                                if( u.get_cargo_space_left() >= resource_amount and 
                                    target_unit.get_cargo_space_left() >= resource_amount ):
                                    # Both units can accept all our resources. Prioritize one that is most-full.
                                    if u.get_cargo_space_left() < target_unit.get_cargo_space_left():
                                        # This new target it better, it has less space left and can take all our
                                        # resources
                                        target_unit = u
                                    
                                elif( target_unit.get_cargo_space_left() >= resource_amount ):
                                    # Don't change targets. Current one is best since it can take all
                                    # the resources, but new target can't.
                                    pass
                                    
                                elif( u.get_cargo_space_left() > target_unit.get_cargo_space_left() ):
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
            partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),  # This is the do-nothing action
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.CART), # Transfer to nearby cart
            partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.WORKER), # Transfer to nearby worker
            SpawnCityAction,
            PillageAction,
        ]
        self.actions_cities = [
            SpawnWorkerAction,
            SpawnCartAction,
            ResearchAction,
        ]
        self.action_space = spaces.Discrete(max(len(self.actions_units), len(self.actions_cities)))

        # ═══════════════════════════════════════════════════════════════════════
        # 结构化观察空间设计（Structured Observation Space）
        # ═══════════════════════════════════════════════════════════════════════
        #
        # 1. 自身状态 (Self State) - 8 维
        #    [unit_type×3, pos×2, cargo×3]
        #
        # 2. 资源信息 (Resources) - (Top-2 closest + Top-2 furthest) × 4 = 16 维
        #    每个资源槽: [type_wood, type_coal, type_uranium, distance]
        #
        # 3. 友方单位 (Friendly Units) - (Top-2 closest + Top-2 furthest) × 4 = 16 维
        #    每个单位槽: [type_worker, type_cart, distance, cargo_space_left]
        #
        # 4. 敌方单位 (Enemy Units) - (Top-2 closest + Top-2 furthest) × 4 = 16 维
        #    每个单位槽: [type_worker, type_cart, distance, cargo_space_left]
        #
        # 5. 友方城市 (Friendly Cities) - (Top-2 closest + Top-2 furthest) × 3 = 12 维
        #    每个城市槽: [distance, fuel_efficiency, fuel_survival]
        #
        # 6. 敌方城市 (Enemy Cities) - (Top-2 closest + Top-2 furthest) × 2 = 8 维
        #    每个城市槽: [distance, city_tile_count]
        #
        # 7. 全局信息 (Global State) - 11 维
        #    [is_night, progress, til_night, til_day,
        #     workers, carts, e_workers, e_carts,
        #     research_pts, coal_ok, uranium_ok]
        #
        # ═══════════════════════════════════════════════════════════════════════
        # 总维度: 8 + 16 + 16 + 16 + 12 + 8 + 11 = 87 维
        # ═══════════════════════════════════════════════════════════════════════
        self.observation_shape = (41,)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float16)

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
        self.max_distance  = (game.map.width - 1) + (game.map.height - 1)
        self.max_city_tiles = game.map.width * game.map.height
        self.cargo_capacity = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]

        # ── 昼夜周期常量 ──────────────────────────────────────────────────────
        self.day_length   = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"]    # 30
        self.night_length = GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]  # 10
        self.cycle_len    = self.day_length + self.night_length
        self.cycle_pos    = game.state["turn"] % self.cycle_len

        # ── 夜晚剩余回合（用于 fuel_survival 计算）──────────────────────────
        self.nights_left = (
            self.night_length if self.cycle_pos < self.day_length
            else (self.cycle_len - self.cycle_pos)
        )
        self.nights_left = max(self.nights_left, 1)

        # ── 友方 / 敌方 city tile 数量（用于 unit count 归一化）─────────────
        opponent_team = (team + 1) % 2
        self.friendly_city_tile_count = 0
        self.enemy_city_tile_count    = 0
        for city in game.cities.values():
            if city.team == team:
                self.friendly_city_tile_count += len(city.city_cells)
            else:
                self.enemy_city_tile_count += len(city.city_cells)
        self.friendly_city_tile_count = max(self.friendly_city_tile_count, 1)
        self.enemy_city_tile_count    = max(self.enemy_city_tile_count, 1)

        # ── Build object_nodes ────────────────────────────────────────────────
        self.object_nodes = {}

        for cell in game.map.resources:
            key = cell.resource.type
            node = np.array([[cell.pos.x, cell.pos.y]])
            if key not in self.object_nodes:
                self.object_nodes[key] = node
            else:
                self.object_nodes[key] = np.concatenate((self.object_nodes[key], node), axis=0)

        for t in [team, (team + 1) % 2]:
            for u in game.state["teamStates"][team]["units"].values():
                key = str(u.type) if t == team else str(u.type) + "_opponent"
                node = np.array([[u.pos.x, u.pos.y]])
                if key not in self.object_nodes:
                    self.object_nodes[key] = node
                else:
                    self.object_nodes[key] = np.concatenate((self.object_nodes[key], node), axis=0)

        for city in game.cities.values():
            for cells in city.city_cells:
                key = "city" if city.team == team else "city_opponent"
                node = np.array([[cells.pos.x, cells.pos.y]])
                if key not in self.object_nodes:
                    self.object_nodes[key] = node
                else:
                    self.object_nodes[key] = np.concatenate((self.object_nodes[key], node), axis=0)

    def get_observation(self, game, unit, city_tile, team, is_new_turn):
        """
        结构化观察空间 (87维)
        1. 自身状态                8维  [unit_type×3, pos×2, cargo×3]
        2. 资源 Top-2close+2far   16维  [type_oh×3, dist] × 4
        3. 友方单位 Top-2c+2f     16维  [type_oh×2, dist, cargo_left] × 4
        4. 敌方单位 Top-2c+2f     16维  [type_oh×2, dist, cargo_left] × 4
        5. 友方城市 Top-2c+2f     12维  [dist, fuel_eff, fuel_surv] × 4
        6. 敌方城市 Top-2c+2f      8维  [dist, tile_count] × 4
        7. 全局信息               11维
        """
        if is_new_turn:
            self.get_initial_observation(game, unit, city_tile, team)

        obs = np.zeros(self.observation_shape, dtype=np.float32)
        idx = 0

        pos = unit.pos if unit is not None else (city_tile.pos if city_tile is not None else None)
        pos_arr = np.array([pos.x, pos.y]) if pos is not None else None

        # ── 辅助：从候选列表中取 top-2 closest + top-2 furthest ───────────────
        def top2_close_far(items, key_dist):
            """items: list of any, key_dist: callable(item)->float
               返回 [close0, close1, far0, far1]，不足用 None 补齐"""
            if not items:
                return [None, None, None, None]
            items_sorted = sorted(items, key=key_dist)
            close = items_sorted[:2]
            far   = list(reversed(items_sorted[-2:]))
            # 如果 close 和 far 有重叠（items<=2），far 补 None
            result = close + far
            # pad
            while len(result) < 4:
                result.append(None)
            return result[:4]

        # ── 1. 自身状态 (8维) ─────────────────────────────────────────────────
        if unit is not None:
            obs[idx]     = 1.0 if unit.type == Constants.UNIT_TYPES.WORKER else 0.0
            obs[idx + 1] = 1.0 if unit.type == Constants.UNIT_TYPES.CART   else 0.0
        elif city_tile is not None:
            obs[idx + 2] = 1.0
        idx += 3

        if pos is not None:
            obs[idx]     = pos.x / (game.map.width  - 1)
            obs[idx + 1] = pos.y / (game.map.height - 1)
        idx += 2

        if unit is not None:
            cap = self.cargo_capacity
            obs[idx]     = unit.cargo["wood"]    / cap
            obs[idx + 1] = unit.cargo["coal"]    / cap
            obs[idx + 2] = unit.cargo["uranium"] / cap
        idx += 3

        # ── 2. 资源 Top-2 closest + Top-2 furthest (16维) ────────────────────
        if pos_arr is not None:
            resources = []
            for rtype in [Constants.RESOURCE_TYPES.WOOD,
                          Constants.RESOURCE_TYPES.COAL,
                          Constants.RESOURCE_TYPES.URANIUM]:
                if rtype == Constants.RESOURCE_TYPES.COAL:
                    if not game.state["teamStates"][team]["researched"]["coal"]:
                        continue
                elif rtype == Constants.RESOURCE_TYPES.URANIUM:
                    if not game.state["teamStates"][team]["researched"]["uranium"]:
                        continue
                if rtype not in self.object_nodes:
                    continue
                nodes = self.object_nodes[rtype]
                dists = np.sum((nodes - pos_arr) ** 2, axis=1) ** 0.5
                for i in range(len(nodes)):
                    resources.append((dists[i], rtype))

            slots = top2_close_far(resources, key_dist=lambda x: x[0])
            for slot in slots:
                if slot is not None:
                    dist, rtype = slot
                    if rtype == Constants.RESOURCE_TYPES.WOOD:
                        obs[idx]     = 1.0
                    elif rtype == Constants.RESOURCE_TYPES.COAL:
                        obs[idx + 1] = 1.0
                    elif rtype == Constants.RESOURCE_TYPES.URANIUM:
                        obs[idx + 2] = 1.0
                    obs[idx + 3] = dist / self.max_distance
                idx += 4
        else:
            idx += 16

        # ── 3. 友方单位 Top-2 closest + Top-2 furthest (16维) ────────────────
        # if pos_arr is not None:
        #     friendly_units = []
        #     for utype in [Constants.UNIT_TYPES.WORKER, Constants.UNIT_TYPES.CART]:
        #         key = str(utype)
        #         if key not in self.object_nodes:
        #             continue
        #         nodes = self.object_nodes[key]
        #         for node in nodes:
        #             node_pos = Position(node[0], node[1])
        #             if unit is not None and node[0] == pos.x and node[1] == pos.y:
        #                 continue  # 排除自己
        #             dist = ((node - pos_arr) ** 2).sum() ** 0.5
        #             cell = game.map.get_cell_by_pos(node_pos)
        #             cargo_left = next(iter(cell.units.values())).get_cargo_space_left() if cell.units else self.cargo_capacity
        #             friendly_units.append((dist, utype, cargo_left))

        #     slots = top2_close_far(friendly_units, key_dist=lambda x: x[0])
        #     for slot in slots:
        #         if slot is not None:
        #             dist, utype, cargo_left = slot
        #             obs[idx]     = 1.0 if utype == Constants.UNIT_TYPES.WORKER else 0.0
        #             obs[idx + 1] = 1.0 if utype == Constants.UNIT_TYPES.CART   else 0.0
        #             obs[idx + 2] = dist / self.max_distance
        #             obs[idx + 3] = cargo_left / self.cargo_capacity
        #         idx += 4
        # else:
        #     idx += 16

        # ── 4. 敌方单位 Top-2 closest + Top-2 furthest (16维) ────────────────
        # if pos_arr is not None:
        #     enemy_units = []
        #     for utype in [Constants.UNIT_TYPES.WORKER, Constants.UNIT_TYPES.CART]:
        #         key = str(utype) + "_opponent"
        #         if key not in self.object_nodes:
        #             continue
        #         nodes = self.object_nodes[key]
        #         for node in nodes:
        #             node_pos = Position(node[0], node[1])
        #             dist = ((node - pos_arr) ** 2).sum() ** 0.5
        #             cell = game.map.get_cell_by_pos(node_pos)
        #             cargo_left = next(iter(cell.units.values())).get_cargo_space_left() if cell.units else self.cargo_capacity
        #             enemy_units.append((dist, utype, cargo_left))

        #     slots = top2_close_far(enemy_units, key_dist=lambda x: x[0])
        #     for slot in slots:
        #         if slot is not None:
        #             dist, utype, cargo_left = slot
        #             obs[idx]     = 1.0 if utype == Constants.UNIT_TYPES.WORKER else 0.0
        #             obs[idx + 1] = 1.0 if utype == Constants.UNIT_TYPES.CART   else 0.0
        #             obs[idx + 2] = dist / self.max_distance
        #             obs[idx + 3] = cargo_left / self.cargo_capacity
        #         idx += 4
        # else:
        #     idx += 16

        # ── 5. 友方城市 Top-2 closest + Top-2 furthest (12维) ────────────────
        if pos_arr is not None:
            friendly_cities = []
            for city in game.cities.values():
                if city.team != team:
                    continue
                tile_coords = np.array([[c.pos.x, c.pos.y] for c in city.city_cells])
                ci   = closest_node(pos_arr, tile_coords)
                dist = ((tile_coords[ci] - pos_arr) ** 2).sum() ** 0.5

                upkeep         = city.get_light_upkeep()
                default_upkeep = len(city.city_cells) * GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["CITY"]
                fuel_eff  = upkeep / default_upkeep if default_upkeep > 0 else 1.0
                fuel_surv = min(city.fuel / (upkeep * self.nights_left), 1.0) if upkeep > 0 else 1.0
                friendly_cities.append((dist, fuel_eff, fuel_surv))

            slots = top2_close_far(friendly_cities, key_dist=lambda x: x[0])
            for slot in slots:
                if slot is not None:
                    dist, fuel_eff, fuel_surv = slot
                    obs[idx]     = dist / self.max_distance
                    obs[idx + 1] = fuel_eff
                    obs[idx + 2] = fuel_surv
                idx += 3
        else:
            idx += 12

        # ── 6. 敌方城市 Top-2 closest + Top-2 furthest (8维) ─────────────────
        if pos_arr is not None:
            opponent_team = (team + 1) % 2
            enemy_cities = []
            for city in game.cities.values():
                if city.team != opponent_team:
                    continue
                tile_coords = np.array([[c.pos.x, c.pos.y] for c in city.city_cells])
                ci   = closest_node(pos_arr, tile_coords)
                dist = ((tile_coords[ci] - pos_arr) ** 2).sum() ** 0.5
                enemy_cities.append((dist, len(city.city_cells)))

            slots = top2_close_far(enemy_cities, key_dist=lambda x: x[0])
            for slot in slots:
                if slot is not None:
                    dist, tile_count = slot
                    obs[idx]     = dist / self.max_distance
                    obs[idx + 1] = tile_count / self.max_city_tiles
                idx += 2
        else:
            idx += 8

        # ── 7. 全局信息 (11维) ────────────────────────────────────────────────
        obs[idx] = float(game.is_night());  idx += 1
        obs[idx] = game.state["turn"] / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]; idx += 1

        turns_until_night = max(0, self.day_length - self.cycle_pos)
        obs[idx] = turns_until_night / self.day_length;  idx += 1

        turns_until_day = max(0, self.cycle_len - self.cycle_pos) if self.cycle_pos >= self.day_length else 0
        obs[idx] = turns_until_day / self.night_length;  idx += 1

        # 友方单位数量：归一化用友方 city tile 数量
        worker_count = len(self.object_nodes.get(str(Constants.UNIT_TYPES.WORKER), []))
        cart_count   = len(self.object_nodes.get(str(Constants.UNIT_TYPES.CART),   []))
        obs[idx]     = worker_count / self.friendly_city_tile_count
        obs[idx + 1] = cart_count   / self.friendly_city_tile_count
        idx += 2

        # 敌方单位数量：归一化用敌方 city tile 数量
        e_worker = len(self.object_nodes.get(str(Constants.UNIT_TYPES.WORKER) + "_opponent", []))
        e_cart   = len(self.object_nodes.get(str(Constants.UNIT_TYPES.CART)   + "_opponent", []))
        obs[idx]     = e_worker / self.enemy_city_tile_count
        obs[idx + 1] = e_cart   / self.enemy_city_tile_count
        idx += 2

        obs[idx] = min(
            game.state["teamStates"][team]["researchPoints"] /
            GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["URANIUM"], 1.0
        );  idx += 1
        obs[idx] = float(game.state["teamStates"][team]["researched"]["coal"]);    idx += 1
        obs[idx] = float(game.state["teamStates"][team]["researched"]["uranium"]); idx += 1

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

            # ── Mask: research already maxed (uranium unlocked) ───────────────
            if game.state["teamStates"][self.team]["researched"][Constants.RESOURCE_TYPES.URANIUM]:
                mask[2] = False  # ResearchAction

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

            if unit is None:
                return mask

            # ── Mask: unit in cooldown → only CENTER (do-nothing) allowed ─────
            if not unit.can_act():
                return np.array([True, False, False, False, False, False, False, False, False], dtype=bool)

            pos = unit.pos

            # ── Mask 1: map boundary — forbid moves that leave the map ────────
            # Indices 1-4 correspond to NORTH / WEST / SOUTH / EAST.
            direction_checks = [
                (1, 0, -1),           # NORTH: y - 1
                (2, -1, 0),           # WEST:  x - 1
                (3, 0, +1),           # SOUTH: y + 1
                (4, +1, 0),           # EAST:  x + 1
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
                cargo_total = unit.cargo["wood"] + unit.cargo["coal"] + unit.cargo["uranium"]
                if cell.has_resource() or cell.is_city_tile() or cargo_total < build_cost:
                    mask[7] = False

            # ── Mask 3: transfer — no valid adjacent friendly unit ────────────
            # TransferAction→cart (index 5) and TransferAction→worker (index 6)
            # are only meaningful when there is a friendly unit of the matching
            # type in an adjacent cell.
            unit_cell = game.map.get_cell_by_pos(pos)
            adjacent_cells = game.map.get_adjacent_cells(unit_cell)

            has_adjacent_cart   = False
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

    def apply_mask(self, action_code, mask):
        """
        If the RL-chosen action_code is forbidden by the mask, redirect to
        the lowest-index allowed action.  If the mask is all-False (should
        never happen in practice), return action_code unchanged.

        This keeps the RL obs→action sample intact in the replay buffer while
        still enforcing hard constraints at execution time.
        """
        if mask[action_code % len(mask)]:
            return action_code  # action is allowed, no change

        return action_code

    def action_code_to_action(self, action_code, game, unit=None, city_tile=None, team=None):
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
                action =  self.actions_cities[action_code%len(self.actions_cities)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )
            else:
                action =  self.actions_units[action_code%len(self.actions_units)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )
            
            return action
        except Exception as e:
            # Not a valid action
            print(e)
            return None

    def take_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            actionCode: Index of action to take into the action array.

        Before executing, applies the action mask so that hard-constraint
        violations (e.g. spawning a cart with no workers) are redirected to
        a safe action.  The RL network still produced the original action_code
        and its obs→action sample is already in the replay buffer, so no
        information is lost.
        """
        # Apply hard-constraint mask: redirect forbidden actions to a safe one
        mask = self.get_action_mask(game, unit=unit, city_tile=city_tile)
        action_code = self.apply_mask(action_code, mask)

        action = self.action_code_to_action(action_code, game, unit, city_tile, team)
        self.match_controller.take_action(action)

    def game_start(self, game):
        """
        This function is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.

        Args:
            game ([type]): Game.
        """
        self.units_last = 0
        self.city_tiles_last = 0
        self.fuel_collected_last = 0
        # upkeep 效率追踪：记录上一回合城市总 fuel，用于计算净消耗
        self.city_fuel_last = 0
        # cooldown 追踪：记录上一回合各单位的 cooldown 值，用于检测溢出
        self.unit_cooldowns_last = {}  # {unit_id: cooldown}
        # research 状态追踪：记录上一回合的解锁状态，用于检测新资源解锁
        self.researched_coal_last = False
        self.researched_uranium_last = False
        # 生存追踪：记录上一回合的单位和城市数量，用于计算生存奖励
        self.workers_last = 0
        self.carts_last = 0
        self.cities_last = 0  # 城市数量（不是 tile 数量）

        # ── Heuristic Curriculum ──────────────────────────────────────────────
        # 每局游戏开始时累加局数计数器，用于指数衰减 heuristic 介入概率。
        # 仅在训练模式下生效；推理模式下 heuristic_prob 固定为 0（完全由 RL 决策）。
        #
        # 衰减公式：heuristic_prob = exp(-games_played / decay_scale)
        #
        # 校准基准（10M steps，每局约 360 steps → 约 27800 局）：
        #   decay_scale=6000  → 训练结束时 prob ≈ 0.01（推荐）
        #   decay_scale=10000 → 训练结束时 prob ≈ 0.06（更保守，退出更慢）
        #
        # 若训练 step 数不同，可在训练前覆盖：
        #   player.heuristic_decay_scale = total_steps / steps_per_game / 4.6
        #   （4.6 ≈ -ln(0.01)，使训练结束时 prob 恰好降至 1%）
        if self.mode == "train":
            self._heuristic_games_played = getattr(self, "_heuristic_games_played", 0) + 1
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
        # 中等优先级奖励：城市扩张和单位创建
        # ═══════════════════════════════════════════════════════════════════════
        
        # Give a reward for unit creation/death. 0.05 reward per unit.
        rewards["rew/r_units"] = (unit_count - self.units_last) * 0.25
        self.units_last = unit_count

        # Give a reward for city creation/death. 0.1 reward per city.
        rewards["rew/r_city_tiles"] = (city_tile_count - self.city_tiles_last) * 0.5
        self.city_tiles_last = city_tile_count

        # ═══════════════════════════════════════════════════════════════════════
        # 采集质量奖励：鼓励 unit 采集与当前研究等级匹配的资源
        # ═══════════════════════════════════════════════════════════════════════
        #
        # 仅在 unit cargo 未满时触发（满了应该去建城，不给此奖励）。
        # 根据本回合 cargo 净增量中比例最高的资源类型，给予对应奖励系数：
        #   uranium（最高研究）: 1.0
        #   coal（中级研究）:    0.5
        #   wood（基础）:        0.25
        # 如果当前采集的资源不是已研究的最高等级，每少一级减半。
        #
        # cargo_quality_reward = 0.0
        # noop_penalty = 0
        # base_quality_reward = 0.05
        # cargo_capacity = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]

        # # 确定当前已研究的最高资源等级
        # # uranium > coal > wood（每少一级奖励减半）
        # researched_uranium = game.state["teamStates"][self.team]["researched"][Constants.RESOURCE_TYPES.URANIUM]
        # researched_coal = game.state["teamStates"][self.team]["researched"][Constants.RESOURCE_TYPES.COAL]

        # if researched_uranium:
        #     tier_multipliers = {"uranium": base_quality_reward, "coal": base_quality_reward/2, "wood": base_quality_reward/4}
        # elif researched_coal:
        #     tier_multipliers = {"uranium": 0.0, "coal": base_quality_reward, "wood": base_quality_reward/2}
        # else:
        #     tier_multipliers = {"uranium": 0.0, "coal": 0.0, "wood": base_quality_reward}

        # for unit in game.state["teamStates"][self.team]["units"].values():
        #     cargo = unit.cargo  # {"wood": int, "coal": int, "uranium": int}
        #     cargo_total = cargo["wood"] + cargo["coal"] + cargo["uranium"]

        #     # cargo 已满则跳过
        #     if cargo_total >= cargo_capacity:
        #         continue

        #     # 用 unit 当前所在格的资源类型判断正在采集什么
        #     cell = game.map.get_cell_by_pos(unit.pos)
        #     if cell.has_resource():
        #         resource_type = cell.resource.type  # "wood" / "coal" / "uranium"
        #         multiplier = tier_multipliers.get(resource_type, 0.0)
        #         if multiplier > 0.0:
        #             cargo_quality_reward += multiplier
        #     elif not cell.is_city_tile() and cell.road <= game.configs["parameters"]["MIN_ROAD"]:
        #         # 格子既没有资源、不是城市、也没有道路 → 纯空地，给予惩罚
        #         cargo_quality_reward -= 0.01

        # rewards["rew/r_cargo_quality"] = cargo_quality_reward

        # ═══════════════════════════════════════════════════════════════════════
        # 不作为奖励：
        # ═══════════════════════════════════════════════════════════════════════
        noop_reward = 0
        for unit in game.state["teamStates"][self.team]["units"].values():
            if not cell.is_city_tile() and cell.road <= game.configs["parameters"]["MIN_ROAD"]:
                # 格子既没有资源、不是城市、也没有道路 → 纯空地，给予惩罚
                noop_reward -= 0.02
        rewards["rew/r_noop_penalty"] = noop_reward

        # ═══════════════════════════════════════════════════════════════════════
        # 游戏结束奖励
        # ═══════════════════════════════════════════════════════════════════════
        
        # Give a reward of 1.0 per city tile alive at the end of the game
        rewards["rew/r_city_tiles_end"] = 0
        if is_game_finished:
            self.is_last_turn = True
            rewards["rew/r_city_tiles_end"] = city_tile_count * 0.2

            '''
            # Example of a game win/loss reward instead
            if game.get_winning_team() == self.team:
                rewards["rew/r_game_win"] = 100.0 # Win
            else:
                rewards["rew/r_game_win"] = -100.0 # Loss
            '''
        
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
        if game.state["teamStates"][self.team]["researched"][Constants.RESOURCE_TYPES.URANIUM]:
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

        Each heuristic is gated by `heuristic_prob` (set in game_start), which
        decays exponentially over training games so that RL gradually takes over:

            heuristic_prob = exp(-games_played / heuristic_decay_scale)

        In inference mode heuristic_prob is always 0.0 so RL runs unassisted.
        To adjust the decay speed, set `agent.heuristic_decay_scale` before
        training (default 6000 games).

        Args:
            game ([type]): Game in progress
            is_first_turn (bool): True if it's the first turn of a game.
        """
        prob = getattr(self, "heuristic_prob", 1.0)

        # Each heuristic is independently sampled so they can decay at the same
        # rate but don't always activate together (adds stochasticity).
        # if random.random() < prob:
        self.research_heuristic(game)
        # no_cart_without_worker is now enforced via action masking in take_action(),
        # so it no longer needs to be called here as a hard override.
        return