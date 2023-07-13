import torch 
import random
import numpy as np
from collections import deque
import snake_game as sg
import model as m
from plot import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness factor
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() // essentially a sliding window
        self.model = m.QNet()
        self.target_model = type(self.model)()
        self.target_model.load_state_dict(torch.load('./model/model7.pth'))
        # self.target_model.load_state_dict(self.model.state_dict())
        self.trainer = m.QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = sg.Point(head.x-20, head.y)
        point_r = sg.Point(head.x+20, head.y)
        point_u = sg.Point(head.x, head.y-20)
        point_d = sg.Point(head.x, head.y+20)

        dir_l = game.direction == sg.Direction.LEFT
        dir_r = game.direction == sg.Direction.RIGHT
        dir_u = game.direction == sg.Direction.UP
        dir_d = game.direction == sg.Direction.DOWN
    
        state = [
            # danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # new direction
            dir_r, 
            dir_l, 
            dir_u, 
            dir_d,

            # food location
            game.food.x < game.head.x, #food left
            game.food.x > game.head.x, #food right
            game.food.y < game.head.y, #food up
            game.food.y > game.head.y, #food down
            # round((abs(game.food.x - game.head.x) + abs(game.food.y - game.head.y))/1120, 1), # screen WxH = 640x480 so 640+480 = 1120
            abs(game.food.x - game.head.x) + abs(game.food.y - game.head.y) < 100, 
            abs(game.food.x - game.head.x) + abs(game.food.y - game.head.y) > 200
        ]

        return np.array(state, dtype=float)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    
    def get_action(self, state):
        # random moves: exploration/exploitation tradeoff
        self.epsilon = 0 - self.n_games
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1

        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.target_model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move



def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = sg.SnakeGameAI()
    epochs = 0

    while True:
        #get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)


        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game: ', agent.n_games, 'Score: ', score, 'Record: ', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            epochs += 1 # indicator that will help us update the target model.

        # if epochs % 10 == 0:
        #     agent.target_model.load_state_dict(agent.model.state_dict())
        


if __name__ == "__main__":
    train()

