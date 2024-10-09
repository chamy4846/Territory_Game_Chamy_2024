import gym
from gym import spaces
import numpy as np
import pygame
import sys
import random

class TerritoryBattleEnv(gym.Env):
    def __init__(self, grid_size=12, max_steps=200):
        super(TerritoryBattleEnv, self).__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps  # 最大步数
        self.current_step = 0  # 当前步数
        self.num_agents = 4  # 两个队伍，每队两个AI Two teams, two AIs per team
        
        self.clock = pygame.time.Clock()

        # 初始化代理
        self.agents = [
            {'team': 'red', 'position': (1, 1), 'type': 'attacker', 'direction': 'right'},  # 红队1号（攻击型）Red Team 1 (Attacking)
            {'team': 'red', 'position': (1, 2), 'type': 'painter', 'direction': 'right'},  # 红队2号（涂色型）Red Team No. 2 (Colored)
            {'team': 'blue', 'position': (self.grid_size - 2, self.grid_size - 2), 'type': 'attacker', 'direction': 'left'},  # Blue Team 1 (Attacking)
            {'team': 'blue', 'position': (self.grid_size - 2, self.grid_size - 3), 'type': 'painter', 'direction': 'left'},  # Blue Team No. 2 (Colored)
        ]

        self.observation_space = spaces.Box(low=0, high=3, shape=(grid_size, grid_size), dtype=np.int32)
        self.action_space = spaces.Discrete(5)

        # Pygame 初始化
        pygame.init()
        self.window_size = 600  # 增大窗口尺寸
        self.cell_size = self.window_size // grid_size
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Territory Battle")

        # 加载红队和蓝队的代理图片
        # Load the proxy images for the red team and blue team
        try:
            self.red_agent_img = pygame.image.load("IMG_Red.png").convert_alpha()
            self.blue_agent_img = pygame.image.load("IMG_Blue.png").convert_alpha()
            self.red_agent_gray_img = pygame.image.load("IMG_Red_Gray.png").convert_alpha()
            self.blue_agent_gray_img = pygame.image.load("IMG_Blue_Gray.png").convert_alpha()
        except pygame.error as e:
            print(f"Error loading image: {e}")
            pygame.quit()
            sys.exit()

        # 调整图片大小以适应网格单元# Resize the image to fit the grid cells
        self.red_agent_img = pygame.transform.scale(self.red_agent_img, (self.cell_size, self.cell_size))
        self.blue_agent_img = pygame.transform.scale(self.blue_agent_img, (self.cell_size, self.cell_size))
        self.red_agent_gray_img = pygame.transform.scale(self.red_agent_gray_img, (self.cell_size, self.cell_size))
        self.blue_agent_gray_img = pygame.transform.scale(self.blue_agent_gray_img, (self.cell_size, self.cell_size))

        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.trajectories = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.current_step = 0  # 重置当前步数# Reset the current step count
    
        # 如果需要初始化障碍物，可以保留此方法调用
        self._initialize_maze()  # 初始化障碍物地形# Initialize obstacle terrain
    
        # 设置代理的初始位置和类型# Set the initial position and type of the agent
        self.agents = [
            {'team': 'red', 'position': (1, 1), 'type': 'attacker', 'direction': 'right'},
            {'team': 'red', 'position': (1, 2), 'type': 'painter', 'direction': 'right'},
            {'team': 'blue', 'position': (self.grid_size - 2, self.grid_size - 2), 'type': 'attacker', 'direction': 'left'},
            {'team': 'blue', 'position': (self.grid_size - 2, self.grid_size - 3), 'type': 'painter', 'direction': 'left'},
        ]
    
        # 初始化网格中的代理位置# Initialize the proxy position in the grid
        for agent in self.agents:
            team = 1 if agent['team'] == 'red' else 2
            self.grid[agent['position']] = team
            self.trajectories[agent['position']] = team
    
        # 初始化奖励# Initialize rewards
        self.red_total_reward = 0
        self.blue_total_reward = 0
    
        return self._get_obs()

    def _initialize_maze(self):
        # 创建对称的障碍物布局，保证在12x12网格中对称# Create a symmetrical obstacle layout, ensuring symmetry in the 12x12 grid
        obstacles = [
            (4, 4), (4, 5), (4, 6), (5, 4), (6, 4),  # 左上角L形障碍物L-shaped obstacle in the upper left corner
            (7, 7), (7, 6), (7, 5), (6, 7), (5, 7),  # 右下角对称L形障碍物Symmetrical L-shaped obstacle in the lower right corner
            (4, 7), (5, 7), (6, 7), (7, 4), (7, 5),  # 左下角L形障碍物L-shaped obstacle in the lower left corner
            (7, 4), (6, 4), (5, 4), (4, 7), (4, 6)   # 右上角对称L形障碍物
        ]
        
        # 将障碍物放入网格中Place obstacles into the grid
        for obstacle in obstacles:
            self.grid[obstacle] = 3  # 3表示墙壁或不可通行的地形 3 indicates a wall or impassable terrain


    def _get_obs(self):
        return np.copy(self.grid)

    def _get_new_position(self, agent, position, action):
        # 计算新位置 Calculate new position
        if action == 1:  # 上
            new_position = (position[0] - 1, position[1])
        elif action == 2:  # 下
            new_position = (position[0] + 1, position[1])
        elif action == 3:  # 左
            new_position = (position[0], position[1] - 1)
        elif action == 4:  # 右
            new_position = (position[0], position[1] + 1)
        else:  # 不移动
            new_position = position

        return new_position

    def step(self, actions):
        rewards = np.zeros(self.num_agents)
        done = False
        
        old_positions = [agent['position'] for agent in self.agents]
        
        # 跟踪占用的位置 Tracking Occupied Locations
        occupied_positions = {agent['position']: agent['team'] for agent in self.agents}
        
        new_positions = [None] * self.num_agents
        
        # 根据动作计算新位置  Calculate new position based on action
        for i, action in enumerate(actions):
            agent = self.agents[i]
            team = 1 if agent['team'] == 'red' else 2
            old_position = agent['position']
            
            # 确定新的位置 Determine the new location
            new_position = self._get_new_position(agent, old_position, action)
            
            # 检查新位置是否在界内且不是障碍物 Check that the new position is in bounds and not an obstacle
            if 0 <= new_position[0] < self.grid_size and 0 <= new_position[1] < self.grid_size and self.grid[new_position] != 3:
                new_positions[i] = new_position
            else:
                new_positions[i] = old_position  # 如果被阻挡则保持原地 If blocked, stay where you are
    
            # 更新代理的方向 Update the direction of the agent
            if new_position != old_position:
                if action == 1:  # 上
                    agent['direction'] = 'up'
                elif action == 2:  # 下
                    agent['direction'] = 'down'
                elif action == 3:  # 左
                    agent['direction'] = 'left'
                elif action == 4:  # 右
                    agent['direction'] = 'right'
        
        # 处理每个代理的移动 Handle each agent's movement
        for i, new_position in enumerate(new_positions):
            agent = self.agents[i]
            team = 1 if agent['team'] == 'red' else 2
            old_position = agent['position']
    
            # 检查攻击型代理是否要攻击前方敌方代理 Check if the attacking agent wants to attack the enemy agent in front
            if agent['type'] == 'attacker':
                target_position = self._get_target_position(agent['position'], agent['direction'])
                
                # 检查目标位置是否有敌方代理 Check if there is an enemy agent at the target location
                for j, enemy_agent in enumerate(self.agents):
                    if enemy_agent['position'] == target_position and enemy_agent['team'] != agent['team']:
                        # 将敌方代理重置到其初始位置 Reset the enemy agent to its initial position
                        initial_position = (1, 1) if enemy_agent['team'] == 'red' else (self.grid_size - 2, self.grid_size - 2)
                        self.grid[enemy_agent['position']] = 0  # 清除敌方代理的原位置 Clear the original location of the enemy agent
                        enemy_agent['position'] = initial_position  # 重置敌方代理的位置 reset enemy position
                        new_positions[j] = initial_position  # 更新敌方代理的新位置 Update the enemy agent's new position
                        self.grid[initial_position] = 2 if enemy_agent['team'] == 'blue' else 1
                        rewards[i] += 10  # 攻击型代理获得10点奖励 Attacking Agents get 10 points of reward
                        #print(f"Agent {i} (Team {agent['team']}) attacked Agent {j} (Team {enemy_agent['team']}) and gained 10 points")
                        break  # 一旦攻击成功，停止检查其他敌人
    
            # 检查是否移动到空格或敌方涂色的格子Checks whether the move is to an empty space or an enemy-colored space.
            if self.grid[new_positions[i]] == 0 or self.grid[new_positions[i]] != team:
                if agent['type'] == 'painter':
                    rewards[i] += 2  # 为涂色代理添加2点奖励
                else:
                    rewards[i] += 1  # 为非涂色代理添加1点奖励
                self.grid[new_positions[i]] = team  # 涂色
                self.trajectories[new_positions[i]] = team
    
            # 移动代理到新位置
            agent['position'] = new_positions[i]
    
        # 清除旧位置Clear old location
        for i, old_position in enumerate(old_positions):
            if 0 <= old_position[0] < self.grid_size and 0 <= old_position[1] < self.grid_size:
                if old_position not in [agent['position'] for agent in self.agents]:
                    self.grid[old_position] = 0
            
        red_count = np.sum(self.trajectories == 1)
        blue_count = np.sum(self.trajectories == 2)
        
        if red_count + blue_count == self.grid_size * self.grid_size or self.current_step >= self.max_steps:
            done = True
            
            # 最终奖励：谁涂色的格子更多Final reward: Whoever colors more grids
            if red_count > blue_count:
                print(f"Red team wins with {red_count} cells colored")
                for i, agent in enumerate(self.agents):
                    if agent['team'] == 'red':
                        rewards[i] += 20  # 为红队添加奖励
            elif blue_count > red_count:
                print(f"Blue team wins with {blue_count} cells colored")
                for i, agent in enumerate(self.agents):
                    if agent['team'] == 'blue':
                        rewards[i] += 20  # 为蓝队添加奖励
            
            print(f"Final rewards: {rewards}")
    
        info = {'red_count': red_count, 'blue_count': blue_count}
        
        self.current_step += 1
        
        return self._get_obs(), rewards, done, info

    def render(self):
        self.screen.fill((0, 0, 0))
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.grid[x, y] == 3:  # 绘制障碍物
                    pygame.draw.rect(self.screen, (128, 128, 128), pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size))
                elif self.trajectories[x, y] == 1:  # 红队涂色
                    pygame.draw.rect(self.screen, (255, 192, 203), pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size))
                elif self.trajectories[x, y] == 2:  # 蓝队涂色
                    pygame.draw.rect(self.screen, (0, 0, 139), pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size))
    
        # 绘制代理
        for agent in self.agents:
            x, y = agent['position']
            if agent['team'] == 'red':
                image = self.red_agent_img
            elif agent['team'] == 'blue':
                image = self.blue_agent_img
    
            # 根据方向旋转图像 Rotate the image according to the orientation
            direction = agent.get('direction', 'right')
            if direction == 'up':
                image = pygame.transform.rotate(image, 90)
            elif direction == 'down':
                image = pygame.transform.rotate(image, 270)
            elif direction == 'left':
                image = pygame.transform.rotate(image, 180)
            elif direction == 'right':
                image = pygame.transform.rotate(image, 0)
    
            self.screen.blit(image, (y * self.cell_size, x * self.cell_size))
    
        pygame.display.flip()
        
        self.clock.tick(30)



    def _get_target_position(self, position, direction):
        x, y = position
        if direction == 'up':
            return (x - 1, y)
        elif direction == 'down':
            return (x + 1, y)
        elif direction == 'left':
            return (x, y - 1)
        elif direction == 'right':
            return (x, y + 1)
        return position

    

    def close(self):
        pygame.quit()

    def handle_events(self):
        # 确保程序能够响应用户的退出操作并正确结束Ensure that the program responds to the user's exit operation and ends properly
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()

if __name__ == "__main__":
    env = TerritoryBattleEnv(grid_size=20, max_steps=200) 
    done = False

    while not done:
        env.handle_events()
        actions = [env.action_space.sample() for _ in range(env.num_agents)]
        obs, rewards, done, info = env.step(actions)
        env.render()
        pygame.time.wait(100)
        print(f"Actions: {actions}")
        print(f"Positions: {[agent['position'] for agent in env.agents]}")
        print(f"Red Count: {info['red_count']}, Blue Count: {info['blue_count']}")
        print(f"Rewards: {rewards}")
        pygame.time.wait(500)

        if done:
            print(f"Total Red Reward: {env.red_total_reward}")
            print(f"Total Blue Reward: {env.blue_total_reward}")
            obs = env.reset()
    env.close()



    



















        
    











    