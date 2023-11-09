

import random
import argparse
import codecs
import os
import numpy

# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        # Given the basename, open the transition and emission files that correspond to the basename.
        with open(f'{basename}.trans', 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3:
                    state_from, state_to, prob = parts
                    prob = float(prob)
                    if state_from not in self.transitions:
                        self.transitions[state_from] = {}
                    self.transitions[state_from][state_to] = prob
        
        # Load emissions
        with open(f'{basename}.emit', 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3:
                    state, symbol, prob = parts
                    prob = float(prob)
                    if state not in self.emissions:
                        self.emissions[state] = {}
                    self.emissions[state][symbol] = prob

        # Validation to make sure the data was loaded
        if not self.transitions or not self.emissions:
            raise ValueError("Transition or emission probabilities not loaded properly.")



   ## you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
        if not self.transitions or not self.emissions:
            raise ValueError("The model has not been loaded with transition and emission probabilities.")

        # Start with the initial state ('#')
        current_state = '#'
        observation = ''

        for _ in range(n):
            # Select the next state using the transition probabilities of the current state
            states, weights = zip(*self.transitions[current_state].items())
            current_state = random.choices(states, weights)[0]

            # Select the emission using the emission probabilities of the current state
            if current_state in self.emissions:
                emissions, emission_weights = zip(*self.emissions[current_state].items())
                observation += random.choices(emissions, emission_weights)[0]
            else:
                # If there are no emissions for the current state, raise an error or handle appropriately
                raise ValueError(f"No emissions found for the current state: {current_state}")

        return observation


    def forward(self, observations):
        num_states = len(self.transitions) - 1  # Exclude the start state
        num_time_steps = len(observations)

        # Initialize the matrix M with zeros for states, but we need num_time_steps + 1 rows
        M = [[0.0 for _ in range(num_states)] for _ in range(num_time_steps + 1)]
        state_indices = {state: idx for idx, state in enumerate(self.transitions) if state != '#'}
        states = list(state_indices.keys())

        # Set up the initial probabilities from the start state
        for s in states:
            state_idx = state_indices[s]
            M[0][state_idx] = self.transitions['#'].get(s, 0) * self.emissions[s].get(observations[0], 0)

        # Propagate forward
        for t in range(1, num_time_steps):
            for s in states:
                state_idx = state_indices[s]
                sum_prob = 0
                for s2 in states:
                    prev_state_idx = state_indices[s2]
                    sum_prob += M[t - 1][prev_state_idx] * self.transitions[s2].get(s, 0) * self.emissions[s].get(observations[t], 0)
                M[t][state_idx] = sum_prob

        # The final state probabilities (not necessary for the forward algorithm)
        final_probs = [M[num_time_steps - 1][state_idx] for state_idx in range(num_states)]
        final_state = states[final_probs.index(max(final_probs))]

        return final_state, M

    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
        num_states = len(self.transitions) - 1  # Exclude the start state
        num_time_steps = len(observations)
        
        # Initialize the matrices M (for probabilities) and backpointers
        M = [[0.0 for _ in range(num_states)] for _ in range(num_time_steps)]
        backpointers = [[0 for _ in range(num_states)] for _ in range(num_time_steps)]

        # State index mapping
        state_indices = {state: idx for idx, state in enumerate(self.transitions) if state != '#'}
        states = list(state_indices.keys())

        # Initialize the first column of M with the initial probabilities
        for s in states:
            state_idx = state_indices[s]
            M[0][state_idx] = self.transitions['#'].get(s, 0) * self.emissions[s].get(observations[0], 0)
            backpointers[0][state_idx] = 0  # Start state backpointer is arbitrary

        # Fill in the Viterbi matrix and backpointers
        for t in range(1, num_time_steps):
            for s in states:
                state_idx = state_indices[s]
                max_prob, max_state = max(
                    ((M[t-1][prev_state_idx] * self.transitions[prev_state].get(s, 0) * self.emissions[s].get(observations[t], 0), prev_state_idx)
                     for prev_state, prev_state_idx in state_indices.items()),
                    key=lambda x: x[0]
                )
                M[t][state_idx] = max_prob
                backpointers[t][state_idx] = max_state

        # Reconstruct the most likely state path
        best_path = []
        # Start with the state that has the highest probability at the last time step
        best_state = max(range(num_states), key=lambda idx: M[num_time_steps-1][idx])
        best_path.append(states[best_state])

        # Follow the backpointers to find the best path
        for t in range(num_time_steps-1, 0, -1):
            best_state = backpointers[t][best_state]
            best_path.insert(0, states[best_state])

        return best_path



model = HMM()
model.load('two_english')
observations = model.generate(10)

num_states = len(model.transitions) - 1
num_time_steps = len(observations)
M = [[0.0 for _ in range(num_states)] for _ in range(num_time_steps)]

print(M)







# print(observations)
# viterbi_path = model.viterbi(observations)
# print(viterbi_path)


# final_state, forward_matrix = model.forward(observations)
# print(final_state)

