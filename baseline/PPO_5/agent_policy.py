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

        # Observation space: (Basic minimum for a miner agent)
        # Object:
        #   1x is worker
        #   1x is cart
        #   1x is citytile
        #   5x direction_nearest_wood
        #   1x distance_nearest_wood
        #   1x amount
        #
        #   5x direction_nearest_coal
        #   1x distance_nearest_coal
        #   1x amount
        #
        #   5x direction_nearest_uranium
        #   1x distance_nearest_uranium
        #   1x amount
        #
        #   5x direction_nearest_city
        #   1x distance_nearest_city
        #   1x amount of fuel
        #
        #   5x direction_nearest_worker
        #   1x distance_nearest_worker
        #   1x amount of cargo
        #
        #   28x (the same as above, but direction, distance, and amount to the furthest of each)
        #
        # Unit:
        #   1x cargo size
        # State:
        #   1x is night
        #   1x percent of game done
        #   2x citytile counts [cur player, opponent]
        #   2x worker counts [cur player, opponent]
        #   2x cart counts [cur player, opponent]
        #   1x research points [cur player]
        #   1x researched coal [cur player]
        #   1x researched uranium [cur player]
        # 时间与城市状态：
        #   1x turns_until_night（归一化）
        #   1x turns_until_day（归一化）
        #   1x city_survival_turns（最近己方城市能存活回合数，归一化）
        #   1x city_size（己方城市总 tile 数，归一化）
        # 资源与单位密度（新增）：
        #   1x wood_amount（当前位置 wood 资源量，÷ 500）
        #   1x coal_amount（当前位置 coal 资源量，已研究才有值，÷ 500）
        #   1x uranium_amount（当前位置 uranium 资源量，已研究才有值，÷ 500）
        #   1x friendly_unit_density（己方单位密度）
        #   1x opponent_unit_density（对手单位密度）
        #
        # 维度计算：
        #   Object:  3 (type flags)
        #            + 5 types × 7 (5 dir + 1 dist + 1 amount) × 2 (nearest + furthest)
        #            = 3 + 70 = 73
        #   Unit:    1
        #   State:   1 + 1 + 2 + 2 + 2 + 1 + 1 + 1 = 11
        #   时间城市: 1 + 1 + 1 + 1 = 4
        #   新增:    1 + 1 + 1 + 1 + 1 = 5
        #   TOTAL  = 73 + 1 + 11 + 4 + 5 = 94
        self.observation_shape = (94,)
        self.observation_space = spaces.Box(low=0, high=1, shape=
        self.observation_shape, dtype=np.float16)

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
        Implements getting a observation from the current game for this unit or city
        """
        observation_index = 0
        if is_new_turn:
           self.get_initial_observation(game, unit, city_tile, team)

        # Observation space: (Basic minimum for a miner agent)
        # Object:
        #   1x is worker
        #   1x is cart
        #   1x is citytile
        #   5x direction_nearest_wood
        #   1x distance_nearest_wood
        #   1x amount
        #
        #   5x direction_nearest_coal
        #   1x distance_nearest_coal
        #   1x amount
        #
        #   5x direction_nearest_uranium
        #   1x distance_nearest_uranium
        #   1x amount
        #
        #   5x direction_nearest_city
        #   1x distance_nearest_city
        #   1x amount of fuel
        #
        #   5x direction_nearest_worker
        #   1x distance_nearest_worker
        #   1x amount of cargo
        #
        #   28x (the same as above, but direction, distance, and amount to the furthest of each)
        #
        # Unit:
        #   1x cargo size
        # State:
        #   1x is night
        #   1x percent of game done
        #   2x citytile counts [cur player, opponent]
        #   2x worker counts [cur player, opponent]
        #   2x cart counts [cur player, opponent]
        #   1x research points [cur player]
        #   1x researched coal [cur player]
        #   1x researched uranium [cur player]
        obs = np.zeros(self.observation_shape)
        
        # Update the type of this object
        #   1x is worker
        #   1x is cart
        #   1x is citytile
        observation_index = 0
        if unit is not None:
            if unit.type == Constants.UNIT_TYPES.WORKER:
                obs[observation_index] = 1.0 # Worker
            else:
                obs[observation_index+1] = 1.0 # Cart
        if city_tile is not None:
            obs[observation_index+2] = 1.0 # CityTile
        observation_index += 3
        
        pos = None
        if unit is not None:
            pos = unit.pos
        else:
            pos = city_tile.pos

        if pos is None:
            observation_index += 7 * 5 * 2
        else:
            # Encode the direction to the nearest objects
            #   5x direction_nearest
            #   1x distance
            for distance_function in [closest_node, furthest_node]:
                for key in [
                    Constants.RESOURCE_TYPES.WOOD,
                    Constants.RESOURCE_TYPES.COAL,
                    Constants.RESOURCE_TYPES.URANIUM,
                    "city",
                    str(Constants.UNIT_TYPES.WORKER)]:
                    # Process the direction to and distance to this object type

                    # Encode the direction to the nearest object (excluding itself)
                    #   5x direction
                    #   1x distance
                    if key in self.object_nodes:
                        if (
                                (key == "city" and city_tile is not None) or
                                (unit is not None and str(unit.type) == key and len(game.map.get_cell_by_pos(unit.pos).units) <= 1 )
                        ):
                            # Filter out the current unit from the closest-search
                            closest_index = closest_node((pos.x, pos.y), self.object_nodes[key])
                            filtered_nodes = np.delete(self.object_nodes[key], closest_index, axis=0)
                        else:
                            filtered_nodes = self.object_nodes[key]

                        if len(filtered_nodes) == 0:
                            # No other object of this type
                            obs[observation_index + 5] = 1.0
                        else:
                            # There is another object of this type
                            closest_index = distance_function((pos.x, pos.y), filtered_nodes)

                            if closest_index is not None and closest_index >= 0:
                                closest = filtered_nodes[closest_index]
                                closest_position = Position(closest[0], closest[1])
                                direction = pos.direction_to(closest_position)
                                mapping = {
                                    Constants.DIRECTIONS.CENTER: 0,
                                    Constants.DIRECTIONS.NORTH: 1,
                                    Constants.DIRECTIONS.WEST: 2,
                                    Constants.DIRECTIONS.SOUTH: 3,
                                    Constants.DIRECTIONS.EAST: 4,
                                }
                                obs[observation_index + mapping[direction]] = 1.0  # One-hot encoding direction

                                # 0 to 1 distance
                                distance = pos.distance_to(closest_position)
                                obs[observation_index + 5] = min(distance / 20.0, 1.0)

                                # 0 to 1 value (amount of resource, cargo for unit, or fuel for city)
                                if key == "city":
                                    # City fuel as % of upkeep for 200 turns
                                    c = game.cities[game.map.get_cell_by_pos(closest_position).city_tile.city_id]
                                    obs[observation_index + 6] = min(
                                        c.fuel / (c.get_light_upkeep() * 200.0),
                                        1.0
                                    )
                                elif key in [Constants.RESOURCE_TYPES.WOOD, Constants.RESOURCE_TYPES.COAL,
                                             Constants.RESOURCE_TYPES.URANIUM]:
                                    # Resource amount
                                    obs[observation_index + 6] = min(
                                        game.map.get_cell_by_pos(closest_position).resource.amount / 500,
                                        1.0
                                    )
                                else:
                                    # Unit cargo
                                    obs[observation_index + 6] = min(
                                        next(iter(game.map.get_cell_by_pos(
                                            closest_position).units.values())).get_cargo_space_left() / 100,
                                        1.0
                                    )

                    observation_index += 7

        if unit is not None:
            # Encode the cargo space
            #   1x cargo size
            obs[observation_index] = unit.get_cargo_space_left() / GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"][
                "WORKER"]
            observation_index += 1
        else:
            observation_index += 1

        # Game state observations

        #   1x is night
        obs[observation_index] = game.is_night()
        observation_index += 1

        #   1x percent of game done
        obs[observation_index] = game.state["turn"] / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
        observation_index += 1

        #   2x citytile counts [cur player, opponent]
        #   2x worker counts [cur player, opponent]
        #   2x cart counts [cur player, opponent]
        max_count = 30
        for key in ["city", str(Constants.UNIT_TYPES.WORKER), str(Constants.UNIT_TYPES.CART)]:
            if key in self.object_nodes:
                obs[observation_index] = len(self.object_nodes[key]) / max_count
            if (key + "_opponent") in self.object_nodes:
                obs[observation_index + 1] = len(self.object_nodes[(key + "_opponent")]) / max_count
            observation_index += 2

        #   1x research points [cur player]
        #   1x researched coal [cur player]
        #   1x researched uranium [cur player]
        obs[observation_index] = game.state["teamStates"][team]["researchPoints"] / 200.0
        obs[observation_index+1] = float(game.state["teamStates"][team]["researched"]["coal"])
        obs[observation_index+2] = float(game.state["teamStates"][team]["researched"]["uranium"])
        observation_index += 3

        # 昼夜倒计时（2 维）
        day_length   = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"]   # 30
        night_length = GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]  # 10
        cycle_len    = day_length + night_length                      # 40
        cycle_pos    = game.state["turn"] % cycle_len

        #   1x turns_until_night（白天剩余回合，归一化；夜晚时为 0）
        turns_until_night = max(0, day_length - cycle_pos)
        obs[observation_index] = turns_until_night / day_length
        observation_index += 1

        #   1x turns_until_day（夜晚剩余回合，归一化；白天时为 0）
        turns_until_day = max(0, cycle_len - cycle_pos) if cycle_pos >= day_length else 0
        obs[observation_index] = turns_until_day / night_length
        observation_index += 1

        # 最近己方城市存活回合数（1 维）
        #   city.fuel / city.get_light_upkeep()，归一化到 200 回合
        nearest_city_survival = 0.0
        if pos is not None:
            nearest_dist = float("inf")
            for city in game.cities.values():
                if city.team != team:
                    continue
                for cell in city.city_cells:
                    dist = pos.distance_to(cell.pos)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        upkeep = city.get_light_upkeep()
                        nearest_city_survival = min(
                            city.fuel / upkeep / 200.0, 1.0
                        ) if upkeep > 0 else 1.0
        obs[observation_index] = nearest_city_survival
        observation_index += 1

        # 己方城市总 tile 数（1 维，归一化到 max_count=30）
        city_tile_total = len(self.object_nodes["city"]) if "city" in self.object_nodes else 0
        obs[observation_index] = min(city_tile_total / max_count, 1.0)
        observation_index += 1

        # ── 资源量与单位密度（5 维）──────────────────────────────────────────
        # 当前位置的资源量（wood, coal, uranium）
        MAX_WOOD_AMOUNT = 500
        if pos is not None:
            cell = game.map.get_cell_by_pos(pos)
            
            # ch0: wood 资源量（÷ 500）
            if cell.has_resource() and cell.resource.type == Constants.RESOURCE_TYPES.WOOD:
                obs[observation_index] = min(cell.resource.amount / MAX_WOOD_AMOUNT, 1.0)
            else:
                obs[observation_index] = 0.0
            observation_index += 1
            
            # ch1: coal 资源量（已研究才有值，÷ 500）
            if (cell.has_resource() and 
                cell.resource.type == Constants.RESOURCE_TYPES.COAL and
                game.state["teamStates"][team]["researched"]["coal"]):
                obs[observation_index] = min(cell.resource.amount / 500.0, 1.0)
            else:
                obs[observation_index] = 0.0
            observation_index += 1
            
            # ch2: uranium 资源量（已研究才有值，÷ 500）
            if (cell.has_resource() and 
                cell.resource.type == Constants.RESOURCE_TYPES.URANIUM and
                game.state["teamStates"][team]["researched"]["uranium"]):
                obs[observation_index] = min(cell.resource.amount / 500.0, 1.0)
            else:
                obs[observation_index] = 0.0
            observation_index += 1
            
            # ch3: 己方单位密度（3x3 区域内己方单位数 ÷ 9）
            friendly_unit_count = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    check_x = pos.x + dx
                    check_y = pos.y + dy
                    if 0 <= check_x < game.map.width and 0 <= check_y < game.map.height:
                        check_cell = game.map.get_cell(check_x, check_y)
                        for unit_id, unit_obj in check_cell.units.items():
                            if unit_obj.team == team:
                                friendly_unit_count += 1
            obs[observation_index] = min(friendly_unit_count / 9.0, 1.0)
            observation_index += 1
            
            # ch4: 对手单位密度（3x3 区域内对手单位数 ÷ 9）
            opponent_unit_count = 0
            opponent_team = (team + 1) % 2
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    check_x = pos.x + dx
                    check_y = pos.y + dy
                    if 0 <= check_x < game.map.width and 0 <= check_y < game.map.height:
                        check_cell = game.map.get_cell(check_x, check_y)
                        for unit_id, unit_obj in check_cell.units.items():
                            if unit_obj.team == opponent_team:
                                opponent_unit_count += 1
            obs[observation_index] = min(opponent_unit_count / 9.0, 1.0)
            observation_index += 1
        else:
            # 如果 pos 为 None，填充 0
            observation_index += 5

        return obs

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
        """
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

        # ── Heuristic Curriculum ──────────────────────────────────────────────
        # 每局游戏开始时累加游戏局数计数器，用于衰减 heuristic 介入概率。
        # 仅在训练模式下生效；推理模式下 heuristic_prob 固定为 0（完全由 RL 决策）。
        if self.mode == "train":
            self._heuristic_games_played = getattr(self, "_heuristic_games_played", 0) + 1
            # 指数衰减：prob = exp(-games_played / decay_scale)
            #
            # 校准基准（10M steps，每局约 2000 steps → 约 5000 局）：
            #   decay_scale=1100 → 训练结束时 prob ≈ 0.01（推荐，与 10M steps 对齐）
            #   decay_scale=2000 → 训练结束时 prob ≈ 0.08（更保守，heuristic 退出更慢）
            #   decay_scale=500  → 训练进行到 ~30% 时 prob 已降至 0.05（退出过早，不推荐）
            #
            # 若训练 step 数不同，可在训练前覆盖：
            #   player.heuristic_decay_scale = total_steps / steps_per_game / 4.6
            #   （4.6 ≈ -ln(0.01)，使训练结束时 prob 恰好降至 1%）
            decay_scale = getattr(self, "heuristic_decay_scale", 1100)
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
        
        rewards = {}
        
        # Give a reward for unit creation/death. 0.05 reward per unit.
        rewards["rew/r_units"] = (unit_count - self.units_last) * 0.05
        self.units_last = unit_count

        # Give a reward for city creation/death. 0.1 reward per city.
        rewards["rew/r_city_tiles"] = (city_tile_count - self.city_tiles_last) * 0.1
        self.city_tiles_last = city_tile_count

        # Reward collecting fuel
        fuel_collected = game.stats["teamStats"][self.team]["fuelGenerated"]
        rewards["rew/r_fuel_collected"] = ( (fuel_collected - self.fuel_collected_last) / 20000 )
        self.fuel_collected_last = fuel_collected
        
        # Give a reward of 1.0 per city tile alive at the end of the game
        rewards["rew/r_city_tiles_end"] = 0
        if is_game_finished:
            self.is_last_turn = True
            rewards["rew/r_city_tiles_end"] = city_tile_count

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

    def cargo_heuristic(self, game, is_first_turn, is_night=False):
        """
        Heuristic: only active during the night phase (is_night=True).

        Threshold = ceil(unit.get_light_upkeep() * night_turns_remaining)
        expressed as a fuel value. If the unit's current cargo fuel value is
        below this threshold it cannot survive the rest of the night on its
        own, so it is directed toward the nearest friendly city tile.

        Applies to both WORKERs and CARTs.
        """
        if not is_night:
            return

        # How many night turns are left in the current night cycle?
        night_length = GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]
        day_length   = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"]
        turn         = game.state["turn"]
        # Turn within the current day/night cycle (0-indexed)
        cycle_pos    = turn % (day_length + night_length)
        # Turns already spent in the current night
        night_turns_elapsed = max(0, cycle_pos - day_length)
        night_turns_remaining = night_length - night_turns_elapsed  # 1 … 10

        for unit_id, unit in game.state["teamStates"][self.team]["units"].items():
            if not unit.can_act():
                continue

            # Fuel threshold: minimum fuel needed to survive the rest of this night
            fuel_threshold = math.ceil(unit.get_light_upkeep() * night_turns_remaining)

            if unit.get_cargo_fuel_value() < fuel_threshold:
                # Find the nearest friendly city tile
                closest_city_tile = None
                closest_dist = float("inf")
                for city in game.cities.values():
                    if city.team != self.team:
                        continue
                    for cell in city.city_cells:
                        dist = unit.pos.distance_to(cell.pos)
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_city_tile = cell

                if closest_city_tile is not None and closest_dist > 0:
                    direction = unit.pos.direction_to(closest_city_tile.pos)
                    action = MoveAction(
                        team=self.team,
                        unit_id=unit.id,
                        direction=direction,
                    )
                    self.match_controller.take_action(action)
        return

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

        if unit_pct <= unit_threshold:
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

    def worker_collect_heuristic(self, game, is_night=False, cargo_threshold=0.5):
        """
        Heuristic: only active during the day phase (is_night=False).

        If a worker has a collectable resource in ALL 5 directions
        (CENTER + N/W/S/E), it stays put (MoveAction CENTER) to keep
        mining instead of wandering.

        Exception: if the unit's cargo is already filled above `cargo_threshold`
        (as a fraction of max capacity), the heuristic is skipped and RL decides
        what to do next (e.g. build a city or head to deposit).

        "Collectable" means the cell has a resource of a type that the
        team has already researched (wood is always available; coal and
        uranium require research points).

        Args:
            cargo_threshold: fraction of max cargo [0, 1] above which the
                             heuristic is suppressed. Default 0.5 (half full).
        """
        if is_night:
            return

        researched = game.state["teamStates"][self.team]["researched"]

        for unit_id, unit in game.state["teamStates"][self.team]["units"].items():
            if unit.type != Constants.UNIT_TYPES.WORKER:
                continue
            if not unit.can_act():
                continue

            # If cargo is sufficiently full, let RL decide — skip heuristic
            unit_type_key = "WORKER" if unit.type == Constants.UNIT_TYPES.WORKER else "CART"
            max_cargo = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"][unit_type_key]
            cargo_used = max_cargo - unit.get_cargo_space_left()
            if cargo_used / max_cargo >= cargo_threshold:
                continue

            unit_cell = game.map.get_cell_by_pos(unit.pos)
            # The 5 cells a worker can collect from: current + 4 adjacent
            candidate_cells = [unit_cell] + game.map.get_adjacent_cells(unit_cell)

            def is_collectable(cell):
                """True if the cell has a resource the team can currently mine."""
                if not cell.has_resource():
                    return False
                rtype = cell.resource.type
                # Wood is always researchable; coal/uranium need research
                if rtype == Constants.RESOURCE_TYPES.WOOD:
                    return True
                return bool(researched.get(rtype, False))

            # Count how many of the 5 directions have a collectable resource
            collectable_count = sum(1 for c in candidate_cells if is_collectable(c))

            if collectable_count == len(candidate_cells):  # all 5 directions covered
                action = MoveAction(
                    team=self.team,
                    unit_id=unit.id,
                    direction=Constants.DIRECTIONS.CENTER,
                )
                self.match_controller.take_action(action)

    def no_cart_without_worker_heuristic(self, game):
        """
        Heuristic: prevent city tiles from spawning a cart when the team has
        no workers alive.

        If there are zero workers, any city tile that would otherwise produce a
        cart is redirected to spawn a worker instead.  This avoids the degenerate
        state where the team ends up with only carts and no workers to collect
        resources.
        """
        # Count current workers for our team
        worker_count = sum(
            1
            for unit in game.state["teamStates"][self.team]["units"].values()
            if unit.type == Constants.UNIT_TYPES.WORKER
        )

        if worker_count > 0:
            return  # Workers exist — no intervention needed

        # No workers: force every idle city tile to spawn a worker
        for city in game.cities.values():
            if city.team != self.team:
                continue
            for cell in city.city_cells:
                city_tile = cell.city_tile
                if not city_tile.can_act():
                    continue
                action = SpawnWorkerAction(
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
        training (default 500 games).

        Args:
            game ([type]): Game in progress
            is_first_turn (bool): True if it's the first turn of a game.
        """
        is_night = game.is_night()
        prob = getattr(self, "heuristic_prob", 1.0)

        # Each heuristic is independently sampled so they decay at the same rate
        # but don't always activate together (adds stochasticity to the curriculum).
        # self.cargo_heuristic(game, is_first_turn, is_night=is_night)
        # if random.random() < prob:
        #     self.worker_collect_heuristic(game, is_night=is_night)
        if random.random() < prob:
            self.research_heuristic(game)
        if random.random() < prob:
            self.no_cart_without_worker_heuristic(game)
        return
    
    
    

