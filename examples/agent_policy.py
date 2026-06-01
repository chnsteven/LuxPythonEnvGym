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
        #    - unit_type: 1x worker, 1x cart, 1x citytile (one-hot)  → 3
        #    - position: 1x normalized_x, 1x normalized_y             → 2
        #    - cargo: 1x wood, 1x coal, 1x uranium (normalized)       → 3
        #
        # 2. 资源信息 (Resources) - Top-3 × 4 = 12 维
        #    每个资源: [type_wood, type_coal, type_uranium, distance]
        #    - 3x type (one-hot): wood/coal/uranium
        #    - 1x distance (normalized)
        #    按距离排序，取最近的 3 个资源点
        #
        # 3. 友方单位 (Friendly Units) - Top-3 × 4 = 12 维
        #    每个单位: [type_worker, type_cart, distance, cargo_space_left]
        #    - 2x type (one-hot): worker/cart
        #    - 1x distance (normalized)
        #    - 1x cargo_space_left (normalized)
        #    按距离排序，取最近的 3 个友方单位
        #
        # 4. 敌方单位 (Enemy Units) - Top-3 × 4 = 12 维
        #    每个单位: [type_worker, type_cart, distance, cargo_space_left]
        #    - 2x type (one-hot): worker/cart
        #    - 1x distance (normalized)
        #    - 1x cargo_space_left (normalized)
        #    按距离排序，取最近的 3 个敌方单位
        #
        # 5. 友方城市 (Friendly Cities) - Top-2 × 2 = 4 维
        #    每个城市: [distance, fuel_efficiency, fuel_survival]
        #    - 1x distance (normalized by MAX_DISTANCE)
        #    - 1x fuel_efficiency = actual_upkeep / default_upkeep
        #      = (tiles*23 - adjacency_bonus) / (tiles*23)，越低城市越紧凑
        #    - 1x fuel_survival = city.fuel / (upkeep * nights_left)
        #      白天 nights_left = NIGHT_LENGTH，夜晚 nights_left = 当前夜晚剩余回合
        #      = 1.0 表示刚好能撑过本次夜晚，>1.0 clip 到 1.0
        #
        # 6. 敌方城市 (Enemy Cities) - Top-2 × 2 = 4 维
        #    每个城市: [distance, city_tile_count]
        #    - 1x distance (normalized)
        #    - 1x city_tile_count (normalized)
        #    按距离排序，取最近的 2 个敌方城市
        #
        # 7. 任务/全局信息 (Global State) - 11 维
        #    - 1x is_night (0/1)
        #    - 1x game_progress (turn / MAX_DAYS)
        #    - 1x turns_until_night (normalized)
        #    - 1x turns_until_day (normalized)
        #    - 2x unit_counts: [friendly_workers, friendly_carts] (normalized)
        #    - 2x opponent_unit_counts: [enemy_workers, enemy_carts] (normalized)
        #    - 1x research_points (normalized)
        #    - 1x researched_coal (0/1)
        #    - 1x researched_uranium (0/1)
        #
        # ═══════════════════════════════════════════════════════════════════════
        # 总维度: 8 + 12 + 12 + 12 + 6 + 4 + 11 = 65 维
        # ═══════════════════════════════════════════════════════════════════════
        self.observation_shape = (49,)
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
        # It's a new turn this event. This flag is set True for only the first observation from each turn.
        # Update any per-turn fixed observation space that doesn't change per unit/city controlled.

        # ── 归一化常数（每局地图尺寸固定，每回合只算一次）────────────────────
        # 曼哈顿距离最大值：对角线两端 (0,0) → (W-1, H-1)
        self.max_distance = (game.map.width - 1) + (game.map.height - 1)
        # 城市 tile / 单位数量上限：整张地图面积
        self.max_city_tiles = game.map.width * game.map.height
        self.max_units = game.map.width * game.map.height
        # 货物容量：RESOURCE_CAPACITY.WORKER = 100
        self.cargo_capacity = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]

        # ── 昼夜周期常量（游戏常量，每回合只算一次）─────────────────────────
        self.day_length = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"]    # 30
        self.night_length = GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]  # 10
        self.cycle_len = self.day_length + self.night_length
        self.cycle_pos = game.state["turn"] % self.cycle_len

        # ── 夜晚剩余回合（用于 fuel_survival 计算）───────────────────────────
        # 白天：下一个夜晚是完整的 NIGHT_LENGTH；夜晚：当前夜晚剩余回合
        self.nights_left = (
            self.night_length if self.cycle_pos < self.day_length
            else (self.cycle_len - self.cycle_pos)
        )
        self.nights_left = max(self.nights_left, 1)  # 避免除零

        # Build a list of object nodes by type for quick distance-searches
        self.object_nodes = {}

        # Add resources
        for cell in game.map.resources:
            if cell.resource.type not in self.object_nodes:
                self.object_nodes[cell.resource.type] = np.array([[cell.pos.x, cell.pos.y]])
            else:
                self.object_nodes[cell.resource.type] = np.concatenate(
                    (
                        self.object_nodes[cell.resource.type],
                        [[cell.pos.x, cell.pos.y]]
                    ),
                    axis=0
                )

        # Add your own and opponent units
        for t in [team, (team + 1) % 2]:
            for u in game.state["teamStates"][team]["units"].values():

                key = str(u.type)
                if t != team:
                    key = str(u.type) + "_opponent"

                if key not in self.object_nodes:
                    self.object_nodes[key] = np.array([[u.pos.x, u.pos.y]])
                else:
                    self.object_nodes[key] = np.concatenate(
                        (
                            self.object_nodes[key],
                            [[u.pos.x, u.pos.y]]
                        )
                        , axis=0
                    )

        # Add your own and opponent cities
        for city in game.cities.values():
            for cells in city.city_cells:
                key = "city"
                if city.team != team:
                    key = "city_opponent"

                if key not in self.object_nodes:
                    self.object_nodes[key] = np.array([[cells.pos.x, cells.pos.y]])
                else:
                    self.object_nodes[key] = np.concatenate(
                        (
                            self.object_nodes[key],
                            [[cells.pos.x, cells.pos.y]]
                        )
                        , axis=0
                    )

    def get_observation(self, game, unit, city_tile, team, is_new_turn):
        """
        结构化观察空间实现 (Structured Observation Space)
        
        总维度: 64
        1. 自身状态 (7维)
        2. 资源信息 Top-3 (12维)
        3. 友方单位 Top-3 (12维)
        4. 敌方单位 Top-3 (12维)
        5. 友方城市 Top-2 (6维)
        6. 敌方城市 Top-2 (4维)
        7. 全局信息 (11维)
        """
        if is_new_turn:
            self.get_initial_observation(game, unit, city_tile, team)

        obs = np.zeros(self.observation_shape, dtype=np.float32)
        idx = 0
        
        # 获取当前位置
        pos = unit.pos if unit is not None else (city_tile.pos if city_tile is not None else None)
        
        # ═══════════════════════════════════════════════════════════════════════
        # 1. 自身状态 (Self State) - 7 维
        # ═══════════════════════════════════════════════════════════════════════
        # 1.1 单位类型 (3维 one-hot)
        if unit is not None:
            if unit.type == Constants.UNIT_TYPES.WORKER:
                obs[idx] = 1.0  # worker
            else:
                obs[idx + 1] = 1.0  # cart
        elif city_tile is not None:
            obs[idx + 2] = 1.0  # citytile
        idx += 3
        
        # 1.2 位置 (2维 normalized by map size)
        if pos is not None:
            obs[idx] = pos.x / (game.map.width - 1)
            obs[idx + 1] = pos.y / (game.map.height - 1)
        idx += 2
        
        # 1.3 货物 (3维 normalized by RESOURCE_CAPACITY.WORKER = 100)
        if unit is not None:
            capacity = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
            obs[idx] = unit.cargo["wood"] / capacity
            obs[idx + 1] = unit.cargo["coal"] / capacity
            obs[idx + 2] = unit.cargo["uranium"] / capacity
        idx += 3
        
        # ═══════════════════════════════════════════════════════════════════════
        # 2. 资源信息 (Resources) - Top-3 × 4 = 12 维
        # ═══════════════════════════════════════════════════════════════════════
        if pos is not None:
            resources = []
            for resource_type in [Constants.RESOURCE_TYPES.WOOD, 
                                 Constants.RESOURCE_TYPES.COAL, 
                                 Constants.RESOURCE_TYPES.URANIUM]:
                # 因果 mask：未研究的资源类型直接跳过，不进入观察
                if resource_type == Constants.RESOURCE_TYPES.COAL:
                    if not game.state["teamStates"][team]["researched"]["coal"]:
                        continue
                elif resource_type == Constants.RESOURCE_TYPES.URANIUM:
                    if not game.state["teamStates"][team]["researched"]["uranium"]:
                        continue

                if resource_type in self.object_nodes:
                    for node in self.object_nodes[resource_type]:
                        node_pos = Position(node[0], node[1])
                        distance = pos.distance_to(node_pos)
                        cell = game.map.get_cell_by_pos(node_pos)
                        amount = cell.resource.amount if cell.has_resource() else 0
                        resources.append({
                            'type': resource_type,
                            'distance': distance,
                            'amount': amount
                        })
            
            # 按距离排序，取Top-3
            resources.sort(key=lambda x: x['distance'])
            for i in range(3):
                if i < len(resources):
                    r = resources[i]
                    # type one-hot (3维)
                    if r['type'] == Constants.RESOURCE_TYPES.WOOD:
                        obs[idx] = 1.0
                    elif r['type'] == Constants.RESOURCE_TYPES.COAL:
                        obs[idx + 1] = 1.0
                    elif r['type'] == Constants.RESOURCE_TYPES.URANIUM:
                        obs[idx + 2] = 1.0
                    # distance (1维): normalized by MAX_DISTANCE = (W-1)+(H-1)
                    obs[idx + 3] = r['distance'] / self.max_distance
                idx += 4
        else:
            idx += 12
        
        # ═══════════════════════════════════════════════════════════════════════
        # 3. 友方单位 (Friendly Units) - Top-3 × 4 = 12 维
        # ═══════════════════════════════════════════════════════════════════════
        if pos is not None:
            friendly_units = []
            for unit_type in [Constants.UNIT_TYPES.WORKER, Constants.UNIT_TYPES.CART]:
                key = str(unit_type)
                if key in self.object_nodes:
                    for node in self.object_nodes[key]:
                        node_pos = Position(node[0], node[1])
                        # 排除自己
                        if unit is not None and node_pos.x == pos.x and node_pos.y == pos.y:
                            continue
                        distance = pos.distance_to(node_pos)
                        cell = game.map.get_cell_by_pos(node_pos)
                        if len(cell.units) > 0:
                            u = next(iter(cell.units.values()))
                            cargo_left = u.get_cargo_space_left()
                            friendly_units.append({
                                'type': unit_type,
                                'distance': distance,
                                'cargo_left': cargo_left
                            })
            
            # 按距离排序，取Top-3
            friendly_units.sort(key=lambda x: x['distance'])
            for i in range(3):
                if i < len(friendly_units):
                    u = friendly_units[i]
                    # type one-hot (2维)
                    if u['type'] == Constants.UNIT_TYPES.WORKER:
                        obs[idx] = 1.0
                    else:
                        obs[idx + 1] = 1.0
                    # distance (1维): normalized by MAX_DISTANCE
                    obs[idx + 2] = u['distance'] / self.max_distance
                    # cargo_left (1维): normalized by RESOURCE_CAPACITY.WORKER = 100
                    obs[idx + 3] = u['cargo_left'] / self.cargo_capacity
                idx += 4
        else:
            idx += 12
        
        # ═══════════════════════════════════════════════════════════════════════
        # 4. 敌方单位 (Enemy Units) - Top-3 × 4 = 12 维
        # ═══════════════════════════════════════════════════════════════════════
        # if pos is not None:
        #     enemy_units = []
        #     for unit_type in [Constants.UNIT_TYPES.WORKER, Constants.UNIT_TYPES.CART]:
        #         key = str(unit_type) + "_opponent"
        #         if key in self.object_nodes:
        #             for node in self.object_nodes[key]:
        #                 node_pos = Position(node[0], node[1])
        #                 distance = pos.distance_to(node_pos)
        #                 cell = game.map.get_cell_by_pos(node_pos)
        #                 if len(cell.units) > 0:
        #                     u = next(iter(cell.units.values()))
        #                     cargo_left = u.get_cargo_space_left()
        #                     enemy_units.append({
        #                         'type': unit_type,
        #                         'distance': distance,
        #                         'cargo_left': cargo_left
        #                     })
            
        #     # 按距离排序，取Top-3
        #     enemy_units.sort(key=lambda x: x['distance'])
        #     for i in range(3):
        #         if i < len(enemy_units):
        #             u = enemy_units[i]
        #             # type one-hot (2维)
        #             if u['type'] == Constants.UNIT_TYPES.WORKER:
        #                 obs[idx] = 1.0
        #             else:
        #                 obs[idx + 1] = 1.0
        #             # distance (1维): normalized by MAX_DISTANCE
        #             obs[idx + 2] = u['distance'] / self.max_distance
        #             # cargo_left (1维): normalized by RESOURCE_CAPACITY.WORKER = 100
        #             obs[idx + 3] = u['cargo_left'] / self.cargo_capacity
        #         idx += 4
        # else:
        #     idx += 12
        
        # ═══════════════════════════════════════════════════════════════════════
        # 5. 友方城市 (Friendly Cities) - Top-2 × 3 = 6 维
        # ═══════════════════════════════════════════════════════════════════════
        if pos is not None:
            # 使用 get_initial_observation 中预计算的夜晚剩余回合
            friendly_cities = []
            for city in game.cities.values():
                if city.team == team:
                    # 找到城市中最近的tile
                    min_dist = float('inf')
                    for cell in city.city_cells:
                        dist = pos.distance_to(cell.pos)
                        if dist < min_dist:
                            min_dist = dist
                    
                    upkeep = city.get_light_upkeep()
                    default_upkeep = len(city.city_cells) * GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["CITY"]
                    # fuel_efficiency: actual_upkeep / default_upkeep
                    # 越低说明 adjacency bonus 越高（城市越紧凑），范围 (0, 1]
                    fuel_efficiency = upkeep / default_upkeep if default_upkeep > 0 else 1.0
                    # fuel_survival: city.fuel / (upkeep * nights_left)
                    # 1.0 = 刚好能撑过本次夜晚，>1.0 clip 到 1.0
                    fuel_survival = min(city.fuel / (upkeep * self.nights_left), 1.0) if upkeep > 0 else 1.0
                    
                    friendly_cities.append({
                        'distance': min_dist,
                        'fuel_efficiency': fuel_efficiency,
                        'fuel_survival': fuel_survival,
                    })
            
            # 按距离排序，取Top-2
            friendly_cities.sort(key=lambda x: x['distance'])
            for i in range(2):
                if i < len(friendly_cities):
                    c = friendly_cities[i]
                    # distance (1维): normalized by MAX_DISTANCE
                    obs[idx] = c['distance'] / self.max_distance
                    # fuel_efficiency (1维): actual_upkeep / default_upkeep，越低城市越紧凑
                    obs[idx + 1] = c['fuel_efficiency']
                    # fuel_survival (1维): city.fuel / (upkeep * nights_left)，1.0=能撑过本次夜晚
                    obs[idx + 2] = c['fuel_survival']
                idx += 3
        else:
            idx += 6
        
        # ═══════════════════════════════════════════════════════════════════════
        # 6. 敌方城市 (Enemy Cities) - Top-2 × 2 = 4 维
        # ═══════════════════════════════════════════════════════════════════════
        # if pos is not None:
        #     enemy_cities = []
        #     opponent_team = (team + 1) % 2
        #     for city in game.cities.values():
        #         if city.team == opponent_team:
        #             # 找到城市中最近的tile
        #             min_dist = float('inf')
        #             for cell in city.city_cells:
        #                 dist = pos.distance_to(cell.pos)
        #                 if dist < min_dist:
        #                     min_dist = dist
                    
        #             tile_count = len(city.city_cells)
                    
        #             enemy_cities.append({
        #                 'distance': min_dist,
        #                 'tile_count': tile_count
        #             })
            
        #     # 按距离排序，取Top-2
        #     enemy_cities.sort(key=lambda x: x['distance'])
        #     for i in range(2):
        #         if i < len(enemy_cities):
        #             c = enemy_cities[i]
        #             # distance (1维): normalized by MAX_DISTANCE
        #             obs[idx] = c['distance'] / self.max_distance
        #             # tile_count (1维): normalized by map area (W*H)
        #             obs[idx + 1] = c['tile_count'] / self.max_city_tiles
        #         idx += 2
        # else:
        #     idx += 4
        
        # ═══════════════════════════════════════════════════════════════════════
        # 7. 全局信息 (Global State) - 11 维
        # ═══════════════════════════════════════════════════════════════════════
        # 7.1 is_night (1维)
        obs[idx] = float(game.is_night())
        idx += 1
        
        # 7.2 game_progress (1维)
        obs[idx] = game.state["turn"] / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
        idx += 1
        
        # 7.3 turns_until_night (1维)
        turns_until_night = max(0, self.day_length - self.cycle_pos)
        obs[idx] = turns_until_night / self.day_length
        idx += 1
        
        # 7.4 turns_until_day (1维)
        turns_until_day = max(0, self.cycle_len - self.cycle_pos) if self.cycle_pos >= self.day_length else 0
        obs[idx] = turns_until_day / self.night_length
        idx += 1
        
        # 7.5 友方单位数量 (2维): normalized by map area (unit cap = city tile cap = W*H)
        worker_count = len(self.object_nodes.get(str(Constants.UNIT_TYPES.WORKER), []))
        cart_count = len(self.object_nodes.get(str(Constants.UNIT_TYPES.CART), []))
        obs[idx] = worker_count / self.max_units
        obs[idx + 1] = cart_count / self.max_units
        idx += 2
        
        # 7.6 敌方单位数量 (2维): normalized by map area
        enemy_worker_count = len(self.object_nodes.get(str(Constants.UNIT_TYPES.WORKER) + "_opponent", []))
        enemy_cart_count = len(self.object_nodes.get(str(Constants.UNIT_TYPES.CART) + "_opponent", []))
        obs[idx] = enemy_worker_count / self.max_units
        obs[idx + 1] = enemy_cart_count / self.max_units
        idx += 2
        
        # 7.7 research_points (1维): normalized by RESEARCH_REQUIREMENTS.URANIUM = 200
        obs[idx] = min(game.state["teamStates"][team]["researchPoints"] /
                       GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["URANIUM"], 1.0)
        idx += 1
        
        # 7.8 researched_coal (1维)
        obs[idx] = float(game.state["teamStates"][team]["researched"]["coal"])
        idx += 1
        
        # 7.9 researched_uranium (1维)
        obs[idx] = float(game.state["teamStates"][team]["researched"]["uranium"])
        idx += 1
        
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

            # ── Mask 5: research already maxed (uranium unlocked) ─────────────
            # ResearchAction (index 2) is pointless once uranium is researched.
            if game.state["teamStates"][self.team]["researched"][Constants.RESOURCE_TYPES.URANIUM]:
                mask[2] = False  # ResearchAction

            # ── Heuristic constraint: no cart without a worker ────────────────
            # SpawnCartAction (index 1) is forbidden when the team has zero
            # workers alive — a cart with no workers cannot collect resources.
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

        # Find the first allowed action
        allowed = np.where(mask)[0]
        if len(allowed) == 0:
            return action_code  # fallback: nothing to mask against
        return int(allowed[0])

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
        rewards["rew/r_units"] = (unit_count - self.units_last) * 0.1
        self.units_last = unit_count

        # Give a reward for city creation/death. 0.1 reward per city.
        rewards["rew/r_city_tiles"] = (city_tile_count - self.city_tiles_last) * 0.2
        self.city_tiles_last = city_tile_count

        # ═══════════════════════════════════════════════════════════════════════
        # 资源收集奖励
        # ═══════════════════════════════════════════════════════════════════════
        
        # Reward collecting fuel.
        # When a new resource tier is unlocked (coal or uranium), give a one-time
        # unlock bonus and boost the fuel-collection reward multiplier for that turn
        # to reinforce the value of mining the newly accessible resource.
        #
        # Research thresholds (from game_constants.json):
        #   coal:    50 research points  → fuel rate ×10 vs wood
        #   uranium: 200 research points → fuel rate ×40 vs wood
        #
        # Multiplier logic:
        #   - uranium just unlocked → 3× fuel reward this turn + 1.0 unlock bonus
        #   - coal just unlocked    → 2× fuel reward this turn + 0.5 unlock bonus
        #   - uranium already known → 2× fuel reward (mining high-value resource)
        #   - coal already known    → 1.5× fuel reward
        #   - only wood available   → 1× (baseline)
        researched_coal    = game.state["teamStates"][self.team]["researched"]["coal"]
        researched_uranium = game.state["teamStates"][self.team]["researched"]["uranium"]

        just_unlocked_coal    = researched_coal    and not self.researched_coal_last
        just_unlocked_uranium = researched_uranium and not self.researched_uranium_last

        rewards["rew/r_unlock_coal"]    = 0.5 if just_unlocked_coal    else 0.0
        rewards["rew/r_unlock_uranium"] = 1.0 if just_unlocked_uranium else 0.0

        if just_unlocked_uranium:
            fuel_multiplier = 3.0
        elif just_unlocked_coal:
            fuel_multiplier = 2.0
        elif researched_uranium:
            fuel_multiplier = 2.0
        elif researched_coal:
            fuel_multiplier = 1.5
        else:
            fuel_multiplier = 1.0

        fuel_collected = game.stats["teamStats"][self.team]["fuelGenerated"]
        rewards["rew/r_fuel_collected"] = (fuel_collected - self.fuel_collected_last) / 20000 * fuel_multiplier
        self.fuel_collected_last = fuel_collected

        # Update research state for next turn
        self.researched_coal_last    = researched_coal
        self.researched_uranium_last = researched_uranium
        
        # ═══════════════════════════════════════════════════════════════════════
        # 游戏结束奖励
        # ═══════════════════════════════════════════════════════════════════════
        
        # Give a reward of 1.0 per city tile alive at the end of the game
        rewards["rew/r_city_tiles_end"] = 0
        if is_game_finished:
            self.is_last_turn = True
            rewards["rew/r_city_tiles_end"] = city_tile_count * 0.01

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