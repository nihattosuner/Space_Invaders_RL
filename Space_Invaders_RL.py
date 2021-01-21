# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 22:34:57 2020

@author: Nihat TOSUNER
"""

import pygame
import random
import numpy as np
import os 
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
FPS=60

model_filename = "my_model_trial.h5"
save_model = True
load_previous_model = True
test=False #False yaparsam train edecek True yaparsam Train edilmiş dosyayı test edecek
# --- Classes

screen_width = 300
screen_height = 480
all_sprites_list = pygame.sprite.Group()
bullet_list = pygame.sprite.Group()


class Block(pygame.sprite.Sprite):
    """ This class represents the block. """

    def __init__(self, color):
        # Call the parent class (Sprite) constructor
        super().__init__()

        self.image = pygame.Surface([20, 20])
        self.image.fill(color)


        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(280)
        self.rect.y = random.randrange(200)
    def update(self):
        
        self.rect.y += 1
        if self.rect.y > screen_height:
            self.rect.y=0
            self.rect.x = random.randrange(280)
            
    def getCoordinates(self):
        return (self.rect.x, self.rect.y)   


class Bullet(pygame.sprite.Sprite):
    """ This class represents the bullet . """

    def __init__(self):
        # Call the parent class (Sprite) constructor
        super().__init__()

        self.image = pygame.Surface([4, 10])
        self.image.fill(BLACK)

        self.rect = self.image.get_rect()

    def update(self):
        """ Move the bullet. """
        self.rect.y -= 3
    def getCoordinates(self):
        return (self.rect.x, self.rect.y)   

class Player(pygame.sprite.Sprite):
    """ This class represents the Player. """

    def __init__(self):
        """ Set up the player on creation. """
        # Call the parent class (Sprite) constructor
        super().__init__()
        
        self.last_shot = pygame.time.get_ticks()
        self.shoot_delay = 250
        self.image = pygame.Surface([20, 20])
        self.image.fill(RED)

        self.rect = self.image.get_rect()
        self.rect.y = 460

    def update(self,action):
        self.speedx = 0  # update'e girdiğinde ilk 0 olsun.
        self.power = 1
        
        keystate = pygame.key.get_pressed()  # klavyeden müdehale olduğunda değişecek bu yüzden pygamein keybord özelliğini alıyorum.

        if keystate[pygame.K_LEFT]or action == 0:
            self.speedx = -4  # sol seçilirse -4 piksel gitsin.
        elif keystate[pygame.K_RIGHT] or action == 1:
            self.speedx = 4  # sağ seçilirse 4 piksel gitsin.    
        elif keystate[pygame.K_SPACE]or action == 2:
            self.shoot()        
        else:
            self.speedx = 0  # hiçbir şey seçilmezse olduğu yerde kalsın.

        
        
        
        if self.rect.right > screen_width:
            self.rect.right = screen_width
        if self.rect.left < 0:
            self.rect.left = 0
        
        
        self.rect.x += self.speedx  # hareket ettiğinde x yönünü uupdate etmem için.
    
    
    def shoot(self):
        now = pygame.time.get_ticks()
        if now - self.last_shot > self.shoot_delay:
            self.last_shot = now
            if self.power == 1:
                bullet = Bullet()
                bullet.rect.x = self.rect.x+10
                bullet.rect.y = self.rect.y
                # Add the bullet to the lists
                all_sprites_list.add(bullet)
                bullet_list.add(bullet)

    def getCoordinates(self):
        return (self.rect.x, self.rect.y)   
epsilon = 1
class DQLAgent:
    def __init__(self):
        # parameter / hyperparameter
        self.state_size = 32 # distance [(playerx-m1x),(playery-m1y),(playerx-m2x),(playery-m2y)]
        self.action_size = 4 # right, left, no move,shoot
        
        self.gamma = 0.99
        self.learning_rate = 0.0001
        
          # explore
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.memory = deque(maxlen = 12000)
        
        self.model = self.build_model()
        
        
    def build_model(self):
        # neural network for deep q learning
        model = Sequential()
        model.add(Dense(128, input_dim = self.state_size, activation = "relu"))
        model.add(Dense(64,   activation = "relu"))
        model.add(Dense(32, activation = "relu"))
        model.add(Dense(16,   activation = "relu"))
        model.add(Dense(self.action_size,activation = "linear"))
        model.compile(loss = "mse", optimizer = Adam(lr = self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        # storage
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        state = np.array(state)
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        # training
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory,batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.array(state)
            next_state = np.array(next_state)
            if done:
                target = reward 
            else:
                target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])
            train_target = self.model.predict(state)
            train_target[0][action] = target
            self.model.fit(state,train_target, verbose = 0)
            
    def adaptiveEGreedy(self):
        global epsilon
        if epsilon > self.epsilon_min:
            epsilon *= self.epsilon_decay
    
    def SaveModel(self):
        if save_model:
            #Save model
            self.model.save_weights(model_filename) 
            print("Model Saved.")

# --- Create the window
agent=DQLAgent()
if load_previous_model:
    dir_path = os.path.realpath(".")
    fn = dir_path + "/"+model_filename
    print("filepath ", fn)
    if  os.path.isfile(fn):
        print("Model Yükleniyor...")
        agent.model.load_weights(model_filename)
        print("Model Yüklendi.")
class Env(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.block_list = pygame.sprite.Group()
#        self.player = Player()
#        all_sprites_list.add(self.player)

#        self.b1=Block(BLACK)
#        self.b2=Block(BLACK)
#        self.b3=Block(BLACK)
#        self.b4=Block(BLACK)
#        self.b5=Block(BLACK)
#        self.b6=Block(BLACK)
#        self.b7=Block(BLACK)
#        self.b8=Block(BLACK)
#        self.b9=Block(BLACK)
#        self.b10=Block(BLACK)
#        self.block_list.add(self.b1)
#        self.block_list.add(self.b2)
#        self.block_list.add(self.b3)
#        self.block_list.add(self.b4)
#        self.block_list.add(self.b5)
#        self.block_list.add(self.b6)
#        self.block_list.add(self.b7)
#        self.block_list.add(self.b8)
#        self.block_list.add(self.b9)
#        self.block_list.add(self.b10)
#        all_sprites_list.add(self.b1)
#        all_sprites_list.add(self.b2)
#        all_sprites_list.add(self.b3)
#        all_sprites_list.add(self.b4)
#        all_sprites_list.add(self.b5)
#        all_sprites_list.add(self.b6)
#        all_sprites_list.add(self.b7)
#        all_sprites_list.add(self.b8)
#        all_sprites_list.add(self.b9)
#        all_sprites_list.add(self.b10)

        
        
        
        
        self.total_reward = 0
        self.done = False
        self.agent = DQLAgent()
        
    def findDistance(self, a, b):
        d = a-b
        return d
    
        
    
    def step(self, action):
        state_list = []
        
        # update
        self.player.update(action)
        self.block_list.update()
        bullet_list.update()
        
        
        # get coordinate
        next_player_state = self.player.getCoordinates()
        next_m1_state = self.b1.getCoordinates()
        next_m2_state = self.b2.getCoordinates()
        next_m3_state = self.b3.getCoordinates()
        next_m4_state = self.b4.getCoordinates()
        next_m5_state = self.b5.getCoordinates()
        next_m6_state = self.b6.getCoordinates()
        next_m7_state = self.b7.getCoordinates()
        next_m8_state = self.b8.getCoordinates()
        next_m9_state = self.b9.getCoordinates()
        next_m10_state = self.b10.getCoordinates()
        next_m11_state = self.b11.getCoordinates()
        next_m12_state = self.b12.getCoordinates()
        next_m13_state = self.b13.getCoordinates()
        next_m14_state = self.b14.getCoordinates()
        next_m15_state = self.b15.getCoordinates()

        

        
        # find distance
        state_list.append(next_player_state[0])
        state_list.append(next_player_state[1])
        state_list.append(next_m1_state[0])
        state_list.append(next_m1_state[1])
        state_list.append(next_m2_state[0])
        state_list.append(next_m2_state[1])
        state_list.append(next_m3_state[0])
        state_list.append(next_m3_state[1])
        state_list.append(next_m4_state[0])
        state_list.append(next_m4_state[1])
        state_list.append(next_m5_state[0])
        state_list.append(next_m5_state[1])
        state_list.append(next_m6_state[0])
        state_list.append(next_m6_state[1])
        state_list.append(next_m7_state[0])
        state_list.append(next_m7_state[1])
        state_list.append(next_m8_state[0])
        state_list.append(next_m8_state[1])
        state_list.append(next_m9_state[0])
        state_list.append(next_m9_state[1])
        state_list.append(next_m10_state[0])
        state_list.append(next_m10_state[1])
        state_list.append(next_m11_state[0])
        state_list.append(next_m11_state[1])
        state_list.append(next_m12_state[0])
        state_list.append(next_m12_state[1])
        state_list.append(next_m13_state[0])
        state_list.append(next_m13_state[1])
        state_list.append(next_m14_state[0])
        state_list.append(next_m14_state[1])
        state_list.append(next_m15_state[0])
        state_list.append(next_m15_state[1])
#        state_list.append(self.findDistance(next_player_state[0],next_m1_state[0]))
#        state_list.append(self.findDistance(next_player_state[1],next_m1_state[1]))
#        state_list.append(self.findDistance(next_player_state[0],next_m2_state[0]))
#        state_list.append(self.findDistance(next_player_state[1],next_m2_state[1]))
#        state_list.append(self.findDistance(next_player_state[0],next_m3_state[0]))
#        state_list.append(self.findDistance(next_player_state[1],next_m3_state[1]))
#        state_list.append(self.findDistance(next_player_state[0],next_m4_state[0]))
#        state_list.append(self.findDistance(next_player_state[1],next_m4_state[1]))
#        state_list.append(self.findDistance(next_player_state[0],next_m5_state[0]))
#        state_list.append(self.findDistance(next_player_state[1],next_m5_state[1]))
#        state_list.append(self.findDistance(next_player_state[0],next_m6_state[0]))
#        state_list.append(self.findDistance(next_player_state[1],next_m6_state[1]))
#        state_list.append(self.findDistance(next_player_state[0],next_m7_state[0]))
#        state_list.append(self.findDistance(next_player_state[1],next_m7_state[1]))
#        state_list.append(self.findDistance(next_player_state[0],next_m8_state[0]))
#        state_list.append(self.findDistance(next_player_state[1],next_m8_state[1]))
#        state_list.append(self.findDistance(next_player_state[0],next_m9_state[0]))
#        state_list.append(self.findDistance(next_player_state[1],next_m9_state[1]))
#        state_list.append(self.findDistance(next_player_state[0],next_m10_state[0]))
#        state_list.append(self.findDistance(next_player_state[1],next_m10_state[1]))
#        state_list.append(self.findDistance(next_player_state[0],next_m11_state[0]))
#        state_list.append(self.findDistance(next_player_state[1],next_m11_state[1]))
#        state_list.append(self.findDistance(next_player_state[0],next_m12_state[0]))
#        state_list.append(self.findDistance(next_player_state[1],next_m12_state[1]))
#        state_list.append(self.findDistance(next_player_state[0],next_m13_state[0]))
#        state_list.append(self.findDistance(next_player_state[1],next_m13_state[1]))
#        state_list.append(self.findDistance(next_player_state[0],next_m14_state[0]))
#        state_list.append(self.findDistance(next_player_state[1],next_m14_state[1]))
#        state_list.append(self.findDistance(next_player_state[0],next_m15_state[0]))
#        state_list.append(self.findDistance(next_player_state[1],next_m15_state[1]))
        
        
        return [state_list]
         
    # reset
    def initialStates(self):
        super().__init__()
        self.block_list = pygame.sprite.Group()
        self.player = Player()
        all_sprites_list.add(self.player)
        self.b1=Block(BLUE)
        self.b2=Block(BLUE)
        self.b3=Block(BLUE)
        self.b4=Block(BLUE)
        self.b5=Block(BLUE)
        self.b6=Block(BLUE)
        self.b7=Block(BLUE)
        self.b8=Block(BLUE)
        self.b9=Block(BLUE)
        self.b10=Block(BLUE)
        self.b11=Block(BLUE)
        self.b12=Block(BLUE)
        self.b13=Block(BLUE)
        self.b14=Block(BLUE)
        self.b15=Block(BLUE)
        self.block_list.add(self.b1)
        self.block_list.add(self.b2)
        self.block_list.add(self.b3)
        self.block_list.add(self.b4)
        self.block_list.add(self.b5)
        self.block_list.add(self.b6)
        self.block_list.add(self.b7)
        self.block_list.add(self.b8)
        self.block_list.add(self.b9)
        self.block_list.add(self.b10)
        self.block_list.add(self.b11)
        self.block_list.add(self.b12)
        self.block_list.add(self.b13)
        self.block_list.add(self.b14)
        self.block_list.add(self.b15)
        all_sprites_list.add(self.b1)
        all_sprites_list.add(self.b2)
        all_sprites_list.add(self.b3)
        all_sprites_list.add(self.b4)
        all_sprites_list.add(self.b5)
        all_sprites_list.add(self.b6)
        all_sprites_list.add(self.b7)
        all_sprites_list.add(self.b8)
        all_sprites_list.add(self.b9)
        all_sprites_list.add(self.b10)
        all_sprites_list.add(self.b11)
        all_sprites_list.add(self.b12)
        all_sprites_list.add(self.b13)
        all_sprites_list.add(self.b14)
        all_sprites_list.add(self.b15)
        
        
        self.total_reward = 0
        self.done = False
    
        state_list = []
        
        # get coordinate
        player_state = self.player.getCoordinates()
        m1_state = self.b1.getCoordinates()
        m2_state = self.b2.getCoordinates()
        m3_state = self.b3.getCoordinates()
        m4_state = self.b4.getCoordinates()
        m5_state = self.b5.getCoordinates()
        m6_state = self.b6.getCoordinates()
        m7_state = self.b7.getCoordinates()
        m8_state = self.b8.getCoordinates()
        m9_state = self.b9.getCoordinates()
        m10_state = self.b10.getCoordinates()
        m11_state = self.b11.getCoordinates()
        m12_state = self.b12.getCoordinates()
        m13_state = self.b13.getCoordinates()
        m14_state = self.b14.getCoordinates()
        m15_state = self.b15.getCoordinates()

        

        
        # find distance
        state_list.append(player_state[0])
        state_list.append(player_state[1])
        state_list.append(m1_state[0])
        state_list.append(m1_state[1])
        state_list.append(m2_state[0])
        state_list.append(m2_state[1])
        state_list.append(m3_state[0])
        state_list.append(m3_state[1])
        state_list.append(m4_state[0])
        state_list.append(m4_state[1])
        state_list.append(m5_state[0])
        state_list.append(m5_state[1])
        state_list.append(m6_state[0])
        state_list.append(m6_state[1])
        state_list.append(m7_state[0])
        state_list.append(m7_state[1])
        state_list.append(m8_state[0])
        state_list.append(m8_state[1])
        state_list.append(m9_state[0])
        state_list.append(m9_state[1])
        state_list.append(m10_state[0])
        state_list.append(m10_state[1])
        state_list.append(m11_state[0])
        state_list.append(m11_state[1])
        state_list.append(m12_state[0])
        state_list.append(m12_state[1])
        state_list.append(m13_state[0])
        state_list.append(m13_state[1])
        state_list.append(m14_state[0])
        state_list.append(m14_state[1])
        state_list.append(m15_state[0])
        state_list.append(m15_state[1])
#        state_list.append(self.findDistance(player_state[0],m1_state[0]))
#        state_list.append(self.findDistance(player_state[1],m1_state[1]))
#        state_list.append(self.findDistance(player_state[0],m2_state[0]))
#        state_list.append(self.findDistance(player_state[1],m2_state[1]))
#        state_list.append(self.findDistance(player_state[0],m3_state[0]))
#        state_list.append(self.findDistance(player_state[1],m3_state[1]))
#        state_list.append(self.findDistance(player_state[0],m4_state[0]))
#        state_list.append(self.findDistance(player_state[1],m4_state[1]))
#        state_list.append(self.findDistance(player_state[0],m5_state[0]))
#        state_list.append(self.findDistance(player_state[1],m5_state[1]))
#        state_list.append(self.findDistance(player_state[0],m6_state[0]))
#        state_list.append(self.findDistance(player_state[1],m6_state[1]))
#        state_list.append(self.findDistance(player_state[0],m7_state[0]))
#        state_list.append(self.findDistance(player_state[1],m7_state[1]))
#        state_list.append(self.findDistance(player_state[0],m8_state[0]))
#        state_list.append(self.findDistance(player_state[1],m8_state[1]))
#        state_list.append(self.findDistance(player_state[0],m9_state[0]))
#        state_list.append(self.findDistance(player_state[1],m9_state[1]))
#        state_list.append(self.findDistance(player_state[0],m10_state[0]))
#        state_list.append(self.findDistance(player_state[1],m10_state[1]))
#        state_list.append(self.findDistance(player_state[0],m11_state[0]))
#        state_list.append(self.findDistance(player_state[1],m11_state[1]))
#        state_list.append(self.findDistance(player_state[0],m12_state[0]))
#        state_list.append(self.findDistance(player_state[1],m12_state[1]))
#        state_list.append(self.findDistance(player_state[0],m13_state[0]))
#        state_list.append(self.findDistance(player_state[1],m13_state[1]))
#        state_list.append(self.findDistance(player_state[0],m14_state[0]))
#        state_list.append(self.findDistance(player_state[1],m14_state[1]))
#        state_list.append(self.findDistance(player_state[0],m15_state[0]))
#        state_list.append(self.findDistance(player_state[1],m15_state[1]))
        
        return [state_list]
        
    def run(self):
        # game loop
        state = self.initialStates()
        

            
        self.score=0
        batch_size = 24
        done = False
        while not done:
            
            # keep loop running at the right speed
            clock.tick(FPS) 
            # process input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done= True
            # update
            action = self.agent.act(state)
            next_state = self.step(action)
            for i in next_state: #her mavi kutucuk asağıya vurulmadan inerse -200
                for t in range(32):
                    if t%2 == 1:
                        if i[t] == 0:
                            self.score -= 100
                            
                            

            self.total_reward += self.score
                
            for block in self.block_list:
        
                # See if it hit a block
                hit_list = pygame.sprite.spritecollide(self.player, self.block_list, True)
                
                if hit_list:
                    self.score -= 200
                    all_sprites_list.empty()
                    self.block_list.empty()
                    bullet_list.empty()
                    #print(self.score)
                    done=True
        
            
            # Calculate mechanics for each bullet
            for bullet in bullet_list:
                
        
                # See if it hit a block
                block_hit_list = pygame.sprite.spritecollide(bullet, self.block_list, True)

                    


                    
                
        
                # For each block hit, remove the bullet and add to the score
                for block in block_hit_list:
                    bullet_list.remove(bullet)
                    all_sprites_list.remove(bullet)
                    self.score += 30
                    #print(self.score)
        
                # Remove the bullet if it flies up off the screen
                if bullet.rect.y < 0:
                    self.score -= 50 #her mermi sıkısı -1
                    #print(self.score)
                    bullet_list.remove(bullet)
                    all_sprites_list.remove(bullet)
            if not self.block_list:
                all_sprites_list.empty()
                self.block_list.empty()
                bullet_list.empty()
                #print(self.score)
                done=True            
            # storage
            self.agent.remember(state, action,self.score, next_state, self.done)
            
            # update state
            state = next_state
            
            # training
            self.agent.replay(batch_size)
            
            
            
            # draw / render(show)
            screen.fill(WHITE)
            all_sprites_list.draw(screen)
            # after drawing flip display
            pygame.display.flip()
        # epsilon greedy
        self.agent.adaptiveEGreedy()
        print(self.score)
    
        pygame.quit()  
# Initialize Pygame
if __name__ == "__main__":
    agent=DQLAgent()
    if test:
        deneme=True
        epsilon=0
        print(" Model Test Ediliyor.")
    else:
        deneme=False
        print(" Model Eğitiliyor")
                
    env = Env()
    liste = []
    t = 0
    while deneme:
        t += 1
        print("Episode: ",t)
        liste.append(env.total_reward)
                
        # initialize pygame and create window
        pygame.init()
        screen = pygame.display.set_mode((screen_width,screen_height))
        pygame.display.set_caption("RL Game")
        clock = pygame.time.Clock()
        
        env.run()

    while not deneme:
        t += 1
        print("Episode: ",t)
        liste.append(env.total_reward)
                
        # initialize pygame and create window
        pygame.init()
        screen = pygame.display.set_mode((screen_width,screen_height))
        pygame.display.set_caption("RL Game")
        clock = pygame.time.Clock()
        if t%10==0:
            agent.SaveModel()
        
        env.run()

        





