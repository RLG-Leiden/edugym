import pygame

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

class PyGameVisualizer:
    def __init__(self, env):
        self.env = env
        self.grid_size = 50
        self.width = self.env.size * self.grid_size
        self.height = self.grid_size * 3
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RoadrunnerEnv")
        pygame.init()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 30)

    def draw_grid(self, obs, info):
        for i in range(self.env.size):
            rect = pygame.Rect(i * self.grid_size, 0, self.grid_size, self.grid_size)
            pygame.draw.rect(self.screen, WHITE, rect, 2)
            if i == obs['agent'][0]:
                pygame.draw.rect(self.screen, RED, rect)
                text = self.font.render("A", True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                self.screen.blit(text, text_rect)
            elif i == info['target'][0]:
                pygame.draw.rect(self.screen, WHITE, rect)
                text = self.font.render("T", True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                self.screen.blit(text, text_rect)
            elif i == info['wall'][0]:
                pygame.draw.rect(self.screen, WHITE, rect)
                text = self.font.render("W", True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                self.screen.blit(text, text_rect)
            else:
                pygame.draw.rect(self.screen, WHITE, rect)

    def update(self, observation, info):
        self.screen.fill(BLACK)
        self.draw_grid(observation, info)
        pygame.display.update()
        self.clock.tick(30)

    def run(self):
        self.env.reset()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = self.env.action_space.sample()
            observation, reward, terminated, _, info = self.env.step(action)
            self.update(observation, info)

            if terminated:
                self.env.reset()
