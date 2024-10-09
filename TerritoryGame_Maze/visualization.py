import pygame
import numpy as np

class Visualization:
    def __init__(self, env):
        self.env = env
        self.episode = 0

    def update(self, episode):
        """
        更新可视化的内容
        """
        self.episode = episode
        self.env.render()  # 使用 Pygame 的 render 方法渲染当前状态Use Pygame's render method to render the current state
        self.display_episode()

    def display_episode(self):
        """
        在窗口中显示当前 Episode 数字
        """
        font = pygame.font.SysFont(None, 24)
        text = font.render(f'Episode: {self.episode}', True, (255, 255, 255))
        self.env.screen.blit(text, (10, 10))  # 在窗口左上角显示 episode 数字Display episode number in the upper left corner of the window
        pygame.display.flip()

    def show(self):
        """
        Pygame 窗口已经在运行时显示
        """
        running = True
        while running:
            self.env.handle_events()  # 处理 Pygame 事件Handling Pygame Events
            self.env.render()  # 渲染 Pygame 窗口Rendering the Pygame Window
            self.display_episode()
            pygame.time.wait(100)


#初始化Initialization.
#注释Comment
#渲染Rendering



