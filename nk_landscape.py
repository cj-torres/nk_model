import hashlib
import random
import numpy as np
rng = np.random.default_rng(1234)

def get_k_tuples(ls, k):
    assert k <= len(ls), "Returned k-tuples must have length less than input list"
    return [tuple(ls[i+j] for j in range(k)) for i in range(len(ls) - k + 1)]


def tuplize(word, k):
    '''
    :param word:
    :param k:
    :return:
    '''
    assert " " not in word, "Spaces are not valid characters and must be used as padding only"
    assert not any(char.isdigit() for char in word), "The string contains a number"
    padded_word = k * " " + word
    tuples = [(i, c) for i, c in enumerate(padded_word)]
    k_tuples = get_k_tuples(tuples, k)
    return k_tuples


class NKLandscape:
    def __init__(self, seed, k, alphabet):
        self.master_seed = seed
        self.k = k
        self.alphabet = alphabet

    def get_value(self, word):
        k_tuples = tuplize(word, self.k)
        # Convert the tuple to a string representation
        fitness = 0.0

        # for normalizing landscape
        num_tuples = len(k_tuples)

        for k_tuple in k_tuples:
            tuple_str = ''.join([f'{char}_{index}' for char, index in k_tuple])

            # Get a hash of the string
            hash_val = hashlib.md5((tuple_str + str(self.master_seed)).encode()).hexdigest()

            # Use the hash to seed a random number generator
            seed_val = int(hash_val, 16)

            # Use the seed value to get a random number
            fitness = fitness + self._seeded_random(seed_val)/num_tuples

        return fitness

    def _seeded_random(self, seed):
        random.seed(seed)
        return random.random()


class ActionAgent:
    def __init__(self, landscape, learning_rate):
        self.landscape = landscape
        self.memory = {a: self.landscape.get_value(a) for a in self.landscape.alphabet + [""]}  # store visited nodes and their values
        self.actions = [self.change_char, self.remove_char, self.add_char]
        self.action_rewards = 0.0
        self.logits = np.array([0.0, 0.0, 0.0])  # start with equal probabilities
        self.learning_rate = learning_rate

    def action_probs(self):
        return np.exp(self.logits)/np.exp(self.logits).sum()

    def get_known_string(self):
        return rng.choice(list(self.memory.keys()))

    def explore(self):
        rand_string = self.get_known_string()

        # decide an action based on probabilities
        action_idx = rng.choice(3, p=self.action_probs())
        action = self.actions[action_idx]

        new_str = action(rand_string)
        if new_str not in self.memory:
            reward = self.landscape.get_value(new_str)
            self.memory[new_str] = reward
            # Here, you can implement a learning mechanism to adjust probs
            # based on the outcomes (e.g., using reinforcement learning techniques)
            self.adjust_probabilities(reward, action_idx)
            self.adjust_expected_rewards(reward)

        else:
            self.adjust_probabilities(0, action_idx)
            self.adjust_expected_rewards(0)

        return new_str, self.memory[new_str]

    def change_char(self, s):
        if len(s) > 0:
            idx = random.randint(0, len(s) - 1)
            old_char = s[idx]
            while True:
                new_char = random.choice(self.landscape.alphabet)
                if new_char != old_char:
                    return s[:idx] + new_char + s[idx + 1:]
        else:
            return s

    def remove_char(self, s):
        if len(s) > 0:
            return s[:-1]
        else:
            return s

    def add_char(self, s):
        new_char = random.choice(self.landscape.alphabet)
        return s + new_char

    def adjust_expected_rewards(self, reward):
        '''
        :param reward:
        :return:
        '''
        self.action_rewards = self.action_rewards + self.learning_rate*(reward-self.action_rewards)

    def adjust_probabilities(self, reward, action):
        '''
        :param reward:
        :param action:
        :return:
        '''
        for i, logit in enumerate(self.logits):
            if action == i:
                self.logits[i] = logit + self.learning_rate*(reward-self.action_rewards)*(1 - self.action_probs()[i])
            else:
                self.logits[i] = logit - self.learning_rate*(reward-self.action_rewards)*self.action_probs()[i]


class ActionWordAgent:
    def __init__(self, landscape, learning_rate):
        self.landscape = landscape
        self.memory = {a: self.landscape.get_value(a) for a in self.landscape.alphabet}  # store visited nodes and their values
        self.word_probs = {k: 0 for k in self.memory.keys()}
        self.actions = [self.change_char, self.remove_char, self.add_char]
        self.action_rewards = 0.0
        self.logits = np.array([0.0, 0.0, 0.0])  # start with equal probabilities
        self.learning_rate = learning_rate

    def action_probs(self):
        return np.exp(self.logits)/np.exp(self.logits).sum()

    def get_word_probs(self):
        word_tuples = [(k, v) for k, v in self.word_probs.items()]
        words = [w[0] for w in word_tuples]
        logits = np.array([w[1] for w in word_tuples])
        return words, np.exp(logits)/np.exp(logits).sum()

    def get_known_string(self):
        return random.choice(list(self.memory.keys()))

    def explore(self):
        words, probs = self.get_word_probs()
        rand_string = rng.choice(words, p=probs)

        # decide an action based on probabilities
        action_idx = rng.choice(3, p=self.action_probs())
        action = self.actions[action_idx]

        new_str = action(rand_string)
        if new_str not in self.memory:
            reward = self.landscape.get_value(new_str)
            self.memory[new_str] = reward
            self.word_probs[new_str] = self.word_probs[rand_string]
            # Here, you can implement a learning mechanism to adjust probs
            # based on the outcomes (e.g., using reinforcement learning techniques)
            self.adjust_probabilities(reward, action_idx)
            self.adjust_word_probabilities(reward, rand_string)
            self.adjust_expected_rewards(reward)

        else:
            self.adjust_probabilities(0, action_idx)
            self.adjust_word_probabilities(0, rand_string)
            self.adjust_expected_rewards(0)

        return new_str, self.memory[new_str]

    def change_char(self, s):
        if len(s) > 0:
            idx = random.randint(0, len(s) - 1)
            old_char = s[idx]
            while True:
                new_char = random.choice(self.landscape.alphabet)
                if new_char != old_char:
                    return s[:idx] + new_char + s[idx + 1:]
        else:
            return s

    def remove_char(self, s):
        if len(s) > 0:
            return s[:-1]
        else:
            return s

    def add_char(self, s):
        new_char = random.choice(self.landscape.alphabet)
        return s + new_char

    def adjust_expected_rewards(self, reward):
        '''
        :param reward:
        :return:
        '''
        self.action_rewards = self.action_rewards + self.learning_rate*(reward-self.action_rewards)

    def adjust_probabilities(self, reward, action):
        '''
        :param reward:
        :param action:
        :return:
        '''
        for i, logit in enumerate(self.logits):
            if action == i:
                self.logits[i] = logit + self.learning_rate*(reward-self.action_rewards)*(1 - self.action_probs()[i])
            else:
                self.logits[i] = logit - self.learning_rate*(reward-self.action_rewards)*self.action_probs()[i]

    def adjust_word_probabilities(self, reward, chosen_word):
        '''
        :param reward:
        :param chosen_word:
        :return:
        '''
        prob_dict = dict(zip(*self.get_word_probs()))
        for word, logit in self.word_probs.items():
            if word == chosen_word:
                self.word_probs[word] = logit + self.learning_rate*(reward-self.action_rewards)*(1 - prob_dict[word])
            else:
                self.word_probs[word] = logit - self.learning_rate*(reward-self.action_rewards)*prob_dict[word]


import csv
filename = "nk_run_3.csv"

with open(filename, "w", newline='') as data_file:
    writer_data = csv.writer(data_file)
    writer_data.writerow(["trial number", "step", "expected reward", "prob 1", "prob 2", "prob 3"])
    for i in range(1):
        landscape = NKLandscape(i,4,["a", "b", "c", "d", "e"])
        agent = ActionWordAgent(landscape, .01)

        for j in range(50001):
            agent.explore()
            if j % 10 == 0:
                writer_data.writerow([i, j, agent.action_rewards, *agent.action_probs().tolist()])

