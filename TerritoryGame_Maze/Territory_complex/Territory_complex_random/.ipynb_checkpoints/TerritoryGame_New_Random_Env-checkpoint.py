import gym
from gym import spaces
import numpy as np
import pygame
import sys
import random

class TerritoryBattleEnvCoopRandom(gym.Env):
    def __init__(self, grid_size=12, max_steps=150):
        super(TerritoryBattleEnvCoopRandom, self).__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0
        self.num_agents = 4

        self.clock = pygame.time.Clock()

        self.agents = [
            {'team': 'red', 'position': (1, 1), 'type': 'attacker', 'direction': 'right', 'is_sharing_vision': False},
            {'team': 'red', 'position': (1, 2), 'type': 'painter', 'direction': 'right', 'is_sharing_vision': False},
            {'team': 'blue', 'position': (self.grid_size - 2, self.grid_size - 2), 'type': 'attacker', 'direction': 'left', 'is_sharing_vision': False},
            {'team': 'blue', 'position': (self.grid_size - 2, self.grid_size - 3), 'type': 'painter', 'direction': 'left', 'is_sharing_vision': False},
        ]

        self.observation_space = spaces.Box(low=0, high=4, shape=(grid_size, grid_size), dtype=np.int32)
        self.action_space = spaces.Discrete(5)

        pygame.init()
        self.window_size = 600
        self.cell_size = self.window_size // grid_size
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Territory Battle")

        try:
            self.red_agent_img = pygame.image.load("IMG_Red.png").convert_alpha()
            self.blue_agent_img = pygame.image.load("IMG_Blue.png").convert_alpha()
            self.red_agent_gray_img = pygame.image.load("IMG_Red_Gray.png").convert_alpha()
            self.blue_agent_gray_img = pygame.image.load("IMG_Blue_Gray.png").convert_alpha()
        except pygame.error as e:
            print(f"Error loading image: {e}")
            pygame.quit()
            sys.exit()

        self.red_agent_img = pygame.transform.scale(self.red_agent_img, (self.cell_size, self.cell_size))
        self.blue_agent_img = pygame.transform.scale(self.blue_agent_img, (self.cell_size, self.cell_size))
        self.red_agent_gray_img = pygame.transform.scale(self.red_agent_gray_img, (self.cell_size, self.cell_size))
        self.blue_agent_gray_img = pygame.transform.scale(self.blue_agent_gray_img, (self.cell_size, self.cell_size))

        self.vision_range = 2

        # 初始化标记Initialize flags
        self.obstacle_positions = set()
        self.resource_positions = set()

        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.trajectories = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.current_step = 0
    
        self.agents = [
            {'team': 'red', 'position': (1, 1), 'type': 'attacker', 'direction': 'right'},
            {'team': 'red', 'position': (1, 2), 'type': 'painter', 'direction': 'right'},
            {'team': 'blue', 'position': (self.grid_size - 2, self.grid_size - 2), 'type': 'attacker', 'direction': 'left'},
            {'team': 'blue', 'position': (self.grid_size - 2, self.grid_size - 3), 'type': 'painter', 'direction': 'left'},
        ]
    
        for agent in self.agents:
            team = 1 if agent['team'] == 'red' else 2
            self.grid[agent['position']] = team
            self.trajectories[agent['position']] = team
    
        # 每回合初始化障碍物和资源点Initialize obstacles and resource points each round
        self._initialize_obstacles_and_resources()
    
        self.red_total_reward = 0
        self.blue_total_reward = 0
    
        return self._get_obs()

    def _initialize_obstacles_and_resources(self):
        """每回合生成障碍物和资源点"""
        self.obstacle_positions.clear()
        self.resource_positions.clear()

        def get_random_position(exclude_positions):
            while True:
                x, y = random.randint(1, self.grid_size - 2), random.randint(1, self.grid_size - 2)
                if (x, y) not in exclude_positions and (x, y) not in [(0, 0), (0, self.grid_size-1), (self.grid_size-1, 0), (self.grid_size-1, self.grid_size-1)]:
                    return (x, y)

        # 生成6个障碍物 Generate 6 obstacles
        for _ in range(6):
            obstacle_pos = get_random_position(self.agents_positions() | self.resource_positions)
            self.obstacle_positions.add(obstacle_pos)
            self.grid[obstacle_pos] = 3

        # 生成2个资源点Generate 2 resource points
        for _ in range(2):
            resource_pos = get_random_position(self.agents_positions() | self.obstacle_positions)
            self.resource_positions.add(resource_pos)
            self.grid[resource_pos] = 4

    def agents_positions(self):
        return {agent['position'] for agent in self.agents}

    def step(self, actions):

        rewards = np.zeros(self.num_agents)
        done = False
    
        old_positions = [agent['position'] for agent in self.agents]
        new_positions = [self._get_new_position(agent, agent['position'], action) for agent, action in zip(self.agents, actions)]
        
        # 检查障碍物并更新新位置Check for obstacles and update the new position
        for i, agent in enumerate(self.agents):
            if new_positions[i] in self.obstacle_positions:  # 如果新位置是障碍物If the new location is an obstacle
                new_positions[i] = old_positions[i]  # 保持原位Stay in place
    
        # 更新代理位置 Update proxy location
        for i, agent in enumerate(self.agents):
            agent['position'] = new_positions[i]
    
            # 检查新位置的色块Check the color patch in its new position
            team_color = 1 if agent['team'] == 'red' else 2
            if self.trajectories[new_positions[i]] == team_color:
                rewards[i] -= 0.1  # 如果移动到自己的色块，减少0.1奖励If you move to your own color block, reduce the reward by 0.1
    #Because this is a floating point number, the final reward will not be an integer, but the error is very small and does not affect the calculation
        # 更新资源点信息和奖励Update resource point information and rewards
        for i, agent in enumerate(self.agents):
            if agent['position'] in self.resource_positions:
                if agent['type'] == 'painter':
                    rewards[i] += 20
                    if agent['team'] == 'red':
                        self.red_total_reward += 20
                        self.blue_total_reward -= 20
                    else:
                        self.blue_total_reward += 20
                        self.red_total_reward -= 20
                    self.grid[agent['position']] = 0  # 资源点被采集后清除# Resource points are cleared after being collected
                    self.resource_positions.remove(agent['position'])  # 从资源点集合中移除# Remove from the resource point collection
                    self._color_around_resource(agent['position'], agent['team'])  # 将周围格子涂色Color the surrounding grid
                else:
                    new_positions[i] = old_positions[i]  # 如果是攻击代理，不能吃资源点，只能走过If it is an attacking agent, it cannot eat resource points, it can only walk through
    
        # 处理攻击逻辑# Processing attack logic
        for i, agent in enumerate(self.agents):
            if agent['type'] == 'attacker':
                target_position = self._get_target_position(agent['position'], agent['direction'])
                for j, enemy_agent in enumerate(self.agents):
                    if enemy_agent['position'] == target_position and enemy_agent['team'] != agent['team']:
                        print(f"Attacker {i} (Team {agent['team']}) attacks Enemy {j} (Team {enemy_agent['team']}) at position {target_position}")
    
                        # 攻击方获得 10 分奖励The attacker gets 10 points reward
                        rewards[i] += 10
                        print(f"Team {agent['team']} (Agent {i}) receives +10 reward. Total reward: {rewards[i]}")
    
                        # 被攻击代理的队伍惩罚 10 分 The team attacked by the proxy will be penalized 10 points
                        for k, team_agent in enumerate(self.agents):
                            if team_agent['team'] == enemy_agent['team']:
                                rewards[k] -= 5
    
                        # 将敌方代理重置到初始位置Reset the enemy agent to its initial position
                        initial_position = (1, 1) if enemy_agent['team'] == 'red' else (self.grid_size - 2, self.grid_size - 2)
    
                        # 清除敌方代理的当前位置Clears the enemy agent's current position
                        self.grid[enemy_agent['position']] = 0
                        # 重置敌方代理的位置到初始位置Reset the enemy agent's position to its initial position
                        enemy_agent['position'] = initial_position
    
                        # 确保初始位置的网格标记为敌方代理的队伍颜色Make sure the grid at the initial position is marked with the enemy agent's team color
                        self.grid[initial_position] = 2 if enemy_agent['team'] == 'blue' else 1
    
                        # 确保敌方代理的位置不会被其他逻辑覆盖Make sure the enemy agent's position is not overwritten by other logic
                        break  # 跳出内层循环，确保只攻击一个敌人Jump out of the inner loop to ensure that only one enemy is attacked
    
        # 清除旧位置Clear old location
        for i, old_position in enumerate(old_positions):
            if 0 <= old_position[0] < self.grid_size and 0 <= old_position[1] < self.grid_size:
                if old_position not in [agent['position'] for agent in self.agents]:
                    self.grid[old_position] = 0
    
        # 更新涂色奖励Updated coloring rewards
        for i, agent in enumerate(self.agents):
            team = 1 if agent['team'] == 'red' else 2
            if self.grid[agent['position']] == 0:
                if agent['type'] == 'painter':
                    rewards[i] += 0.5
                else:
                    rewards[i] += 0.1
                self.grid[agent['position']] = team
                self.trajectories[agent['position']] = team
    
        # 统计红队和蓝队的涂色格子数量Count the number of colored squares of the red team and the blue team
        red_count = np.sum(self.trajectories == 1)
        blue_count = np.sum(self.trajectories == 2)
    
        # 判断是否达到结束条件Determine whether the end condition is met
        if red_count + blue_count == self.grid_size * self.grid_size or self.current_step >= self.max_steps:
            done = True
    
            # 根据涂色数量给与额外奖励Additional rewards based on the number of colors
            if red_count > blue_count:
                for i, agent in enumerate(self.agents):
                    if agent['team'] == 'red':
                        rewards[i] += 20
            elif blue_count > red_count:
                for i, agent in enumerate(self.agents):
                    if agent['team'] == 'blue':
                        rewards[i] += 20
    
        info = {'red_count': red_count, 'blue_count': blue_count}
    
        self.current_step += 1

        # 每回合重置障碍物和资源点Reset obstacles and resource points each round
        if self.current_step % self.max_steps == 0:
            self._initialize_obstacles_and_resources()
    
        return self._get_obs(), rewards, done, info

    def _color_around_resource(self, resource_position, team):
        x, y = resource_position
        team_color = 1 if team == 'red' else 2
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size:
                    if self.grid[x + dx, y + dy] != 3:  # 不涂色障碍物Unpainted obstacles
                        self.grid[x + dx, y + dy] = team_color
                        self.trajectories[x + dx, y + dy] = team_color

    def _get_target_position(self, position, direction):
        x, y = position
        if direction == 'up':
            target_position = (max(0, x - 1), y)
        elif direction == 'down':
            target_position = (min(self.grid_size - 1, x + 1), y)
        elif direction == 'left':
            target_position = (x, max(0, y - 1))
        elif direction == 'right':
            target_position = (x, min(self.grid_size - 1, y + 1))
        else:
            target_position = position

        return target_position

    def render(self):
        self.screen.fill((0, 0, 0))
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # 绘制网格和代理Drawing Meshes and Delegates
                if self.grid[x, y] == 3:  # 绘制障碍物Drawing obstacles
                    pygame.draw.rect(self.screen, (128, 128, 128), pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size))
                elif self.grid[x, y] == 4:  # 绘制资源点Draw resource points
                    pygame.draw.rect(self.screen, (255, 255, 0), pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size))
                elif self.trajectories[x, y] == 1:  # 红队涂色Red Team Coloring
                    pygame.draw.rect(self.screen, (255, 192, 203), pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size))
                elif self.trajectories[x, y] == 2:  # 蓝队涂色Blue Team Coloring
                    pygame.draw.rect(self.screen, (0, 0, 139), pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size))
    
        for agent in self.agents:
            x, y = agent['position']
            if agent['team'] == 'red':
                image = self.red_agent_img
            elif agent['team'] == 'blue':
                image = self.blue_agent_img
    
            direction = agent.get('direction', 'right')
            if direction == 'up':
                image = pygame.transform.rotate(image, 90)
            elif direction == 'down':
                image = pygame.transform.rotate(image, 270)
            elif direction == 'left':
                image = pygame.transform.rotate(image, 180)
            elif direction == 'right':
                image = pygame.transform.rotate(image, 0)
    
            rect = image.get_rect(center=(y * self.cell_size + self.cell_size // 2, x * self.cell_size + self.cell_size // 2))
            self.screen.blit(image, rect.topleft)
    
        pygame.display.flip()
        self.clock.tick(30)

    def _get_obs(self):
        obs = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for agent in self.agents:
            x, y = agent['position']
            x_min = max(0, x - self.vision_range)
            x_max = min(self.grid_size, x + self.vision_range + 1)
            y_min = max(0, y - self.vision_range)
            y_max = min(self.grid_size, y + self.vision_range + 1)
            for i in range(x_min, x_max):
                for j in range(y_min, y_max):
                    obs[i, j] = self.grid[i, j]
        return obs

    def _get_new_position(self, agent, position, action):
        if action == 1:  # Up
            new_position = (max(0, position[0] - 1), position[1])
        elif action == 2:  # Down
            new_position = (min(self.grid_size - 1, position[0] + 1), position[1])
        elif action == 3:  # Left
            new_position = (position[0], max(0, position[1] - 1))
        elif action == 4:  # Right
            new_position = (position[0], min(self.grid_size - 1, position[1] + 1))
        else:
            new_position = position
    
        if new_position != position:
            if action == 1:
                agent['direction'] = 'up'
            elif action == 2:
                agent['direction'] = 'down'
            elif action == 3:
                agent['direction'] = 'left'
            elif action == 4:
                agent['direction'] = 'right'
    
        return new_position

    def close(self):
        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()


