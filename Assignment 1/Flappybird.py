#Denne kode er blevet lavet med inspiration fra https://youtu.be/GiUGVOqqCKg?si=Dw4a0Jg6AyHHzMvO, med lavet om til en class i stedet for og debugged med hjælp fra chatgpt. 
import pygame
from pygame.locals import *
import random
import torch
import numpy as np

SEED = 41

# Python
random.seed(SEED)

# NumPy
np.random.seed(SEED)

class FlappyBirdGame:
    def __init__(self):
        
        self.rendering = False # Set to true if you want to play the game, it is standard to false so it will train faster
        
        if self.rendering:
            pygame.init()
        

        # Screen dimensions
        self.screen_width = 864
        self.screen_height = 936
        self.clock = pygame.time.Clock()
        self.fps = 60

        # Game variables
        self.scroll_speed = 4
        self.game_over = False
        self.pipe_gap = 150
        self.pipe_dist_x = 600
        self.score_game = 0
        self.reward = 0
        self.pass_pipe = False
        self.penalty_applied = False
        self.tick_limited = True
        self.flying = True #False hvis vent med at flyve, TRUE hvis flyv ligså snart den starter
        self.tal = 0
        self.pipe_positions = np.linspace(-150, 150, 15) 
        
        self.reward_alive = 0.1
        self.reward_pipe = 10
        self.penalty_death = -10.0
        
        
        if self.rendering:
           
            # Screen dimensions
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('Crappy Bird')

            # Font and colors
            self.font = pygame.font.SysFont('Bauhaus 93', 60)
            self.white = (255, 255, 255)

            # Images
            self.bg = pygame.image.load('Assignment 1/billeder/bg.png')
            self.ground = pygame.image.load('Assignment 1/billeder/ground.png')
            self.button_img = pygame.image.load('Assignment 1/billeder/game_over.png')
            self.bird_imgs = [pygame.image.load(f'Assignment 1/billeder/bird{num}.png') for num in range(1, 4)]
            self.pipe_img = pygame.image.load('Assignment 1/billeder/pipe_bottom.png')
        else: 
            self.pipe_img = None
         
        #self.bird_imgs = [pygame.image.load(f'billeder/bird{num}.png') for num in range(1, 4)]
        
        # Bird properties
        self.bird_index = 0
        self.bird_counter = 0
        self.bird_vel = 0
        self.bird_clicked = False
        self.bird_rect = (
        self.bird_imgs[0].get_rect(center=(400, self.screen_height // 2))
        if self.rendering
        else pygame.Rect(83, 456, 34, 24)
)
        self.pipes = []

        # Button
        if self.rendering:
            self.button_rect = self.button_img.get_rect(center=(self.screen_width // 2, self.screen_height // 2))

        # actions
        self.actions = [0, 1]  # Do nothing, flap
        
    def close(self):
        if self.rendering:
            pygame.quit()

    def draw_text(self, text, x, y):
        img = self.font.render(text, True, self.white)
        self.screen.blit(img, (x, y))

    def reset_game(self):
        self.pipes.clear()
        self.bird_rect.center = (400, self.screen_height // 2)
        self.bird_vel = 0
        self.flying = True
        self.game_over = False
        self.score = 0
        self.score_game = 0
        self.reward = 0
        self.penalty_applied = False

        return self.get_states()

    def get_states(self):
        distance, pipe_top, pipe_btm = self.get_distance_to_next_pipe()
        return self.bird_rect.y, self.bird_vel, distance, pipe_top, pipe_btm # Bird y, distance to next pipe, gap center
        
    def get_distance_to_next_pipe(self):
        for pipe in self.pipes:
            if pipe["rect"].centerx > self.bird_rect.centerx:
                distance2 = pipe["rect"].left - self.bird_rect.right
                max_dist = 450 
                distance = max(1, min(distance2, max_dist))
                
                #gap_center = pipe["rect"].top - (self.pipe_gap / 2)
                if pipe["pos"] == -1:  # Bottom pipe
                    pipe_top = pipe["rect"].top - self.pipe_gap
                    pipe_btm = pipe["rect"].top 
                return distance, pipe_top, pipe_btm
        return 0, 0, 0  

    def draw_pipes(self):
        for pipe in self.pipes:
            if pipe["pos"] == -1:
                self.screen.blit(self.pipe_img, pipe["rect"])
            else:
                flipped_pipe = pygame.transform.flip(self.pipe_img, False, True)
                self.screen.blit(flipped_pipe, pipe["rect"])

            # #red circle at gap height
            # if pipe["pos"] == -1:  # Bottom pipe
            #     gap_center_top = pipe["rect"].top - self.pipe_gap
            #     gap_center_btm = pipe["rect"].top 
            #     pygame.draw.circle(self.screen, (255, 0, 0), (pipe["rect"].centerx, int(gap_center_top)), 10)
            #     pygame.draw.circle(self.screen, (255, 0, 0), (pipe["rect"].centerx, int(gap_center_btm)), 10)

    def update_pipes(self):
        if not self.game_over:
            for pipe in self.pipes:
                pipe["rect"].x -= self.scroll_speed

            # Remove pipe if off-screen
            self.pipes = [pipe for pipe in self.pipes if pipe["rect"].right > 0]

    def check_collision(self):
        for pipe in self.pipes:
            if self.bird_rect.colliderect(pipe["rect"]):
                return True
        return self.bird_rect.bottom >= 768 or self.bird_rect.top <= 1 
          
    def handle_bird_movement(self):
        if self.flying:
            self.bird_vel += 0.5
            if self.bird_vel > 8:
                self.bird_vel = 8
            if self.bird_rect.bottom < 768:
                self.bird_rect.y += int(self.bird_vel)
            if self.bird_rect.top < 0:
                self.bird_rect.top = 0

        if self.rendering:
            if pygame.mouse.get_pressed()[0] == 1 and not self.bird_clicked:
                self.bird_vel = -10
                self.bird_clicked = True
            if pygame.mouse.get_pressed()[0] == 0:
                self.bird_clicked = False

            # Animate bird
            self.bird_counter += 1
            if self.bird_counter > 5:
                self.bird_counter = 0
                self.bird_index = (self.bird_index + 1) % len(self.bird_imgs)

    def spawn_pipes(self):
        if not self.pipes or self.pipes[-1]["rect"].left < self.pipe_dist_x:
            #pipe_height = random.randint(-150, 150)
            pipe_height = float(np.random.choice(self.pipe_positions))

            if self.rendering:
                self.btm_pipe = {
                    "rect": self.pipe_img.get_rect(
                        topleft=(self.screen_width, int(self.screen_height / 2) + pipe_height + self.pipe_gap / 2)
                    ),
                    "pos": -1,
                    "passed": False,
                }
                self.top_pipe = {
                    "rect": self.pipe_img.get_rect(
                        bottomleft=(self.screen_width, int(self.screen_height / 2) + pipe_height - self.pipe_gap / 2)
                    ),
                    "pos": 1,
                }
            else:
                self.btm_pipe = {
                    "rect": pygame.Rect(
                        self.screen_width, int(self.screen_height / 2) + pipe_height + self.pipe_gap / 2, 52, 800
                    ),
                    "pos": -1,
                    "passed": False,
                }
                self.top_pipe = {
                    "rect": pygame.Rect(
                        self.screen_width, int(self.screen_height / 2) + pipe_height - self.pipe_gap / 2 - 800, 52, 800
                    ),
                    "pos": 1,
              
                }

            self.pipes.append(self.btm_pipe)
            self.pipes.append(self.top_pipe)


    def step(self, action):
        self.done = False
        
        if action == 1 and not self.game_over:  # Flap
            self.bird_vel = -10
        elif action == 0 and not self.game_over:  # Do nothing
            pass

        # Update the environment
        if not self.game_over and self.flying:
            self.reward = self.reward_alive
            self.spawn_pipes()
            self.update_pipes()
            self.handle_bird_movement()

            if self.check_collision():
                self.reward = self.penalty_death
                self.game_over = True
                self.done = True
                
            # Update score and rewards
            if not self.game_over:
                for pipe in self.pipes:
                    if not pipe.get("passed", False) and pipe["rect"].right < self.bird_rect.left:
                        pipe["passed"] = True
                        self.reward += self.reward_pipe
                        self.score_game += 0.5  

        next_state = self.get_states()

        return next_state, self.reward, self.game_over, self.score_game
    
    def run(self):
        running = True
        while running and self.rendering:
            if self.game_over:
                raise Exception("Game over")
            
            if self.rendering and self.tick_limited:
               self.clock.tick(self.fps)

            if self.rendering:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.MOUSEBUTTONDOWN and not self.flying and not self.game_over:
                        self.flying = True
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            self.reset_game()
                        if event.key == pygame.K_t:
                            self.tick_limited = not self.tick_limited
                        if event.key == pygame.K_f:
                            self.rendering = not self.rendering

            if not self.game_over and self.flying == True:
                self.reward += self.reward_alive
                self.spawn_pipes()
                self.update_pipes()
                self.handle_bird_movement()

                
                if self.check_collision():
                    self.reward += self.penalty_death  
                    self.game_over = True

                # Update score and reward
                if not self.game_over:
                    for pipe in self.pipes:
                        if not pipe.get("passed", False) and pipe["rect"].right < self.bird_rect.left:
                            self.reward += self.reward_pipe
                            pipe["passed"] = True
                            if pipe["pos"] == -1:
                                self.score_game += 1

            print(self.get_states(), self.reward, self.game_over, self.score_game)

            if self.rendering:
                self.screen.blit(self.bg, (0, 0))
                self.draw_pipes()
                self.screen.blit(self.bird_imgs[self.bird_index], self.bird_rect)
                self.screen.blit(self.ground, (0, 768))
                self.draw_text(str(self.score_game), self.screen_width // 2, 30)

                if self.game_over:
                    self.screen.blit(self.button_img, self.button_rect.topleft)
                    
                    if pygame.mouse.get_pressed()[0] == 1 and self.button_rect.collidepoint(pygame.mouse.get_pos()):
                        self.reset_game()

                pygame.display.update()

        pygame.quit()

if __name__ == "__main__":
    game = FlappyBirdGame()
    game.run()

