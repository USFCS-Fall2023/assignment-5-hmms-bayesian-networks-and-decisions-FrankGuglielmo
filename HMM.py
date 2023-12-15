

import random
import argparse
import codecs
import os
import numpy as np

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
        #Keep a list of all the possible states
        self.states = []

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
                    self.transitions[state_from][state_to] = float(prob)
                    if state_from != '#' and state_from not in self.states:
                        self.states.append(state_from)
        
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
        stateseq = []
        observations = []

        for _ in range(n):
            # Determine the next state using the transition probabilities of the current state
            next_state = np.random.choice(
                list(self.transitions[current_state].keys()),
                p=list(self.transitions[current_state].values())
            )
            stateseq.append(next_state)

            # Determine the next output based on the next state's emission probabilities
            next_output = np.random.choice(
                list(self.emissions[next_state].keys()),
                p=list(self.emissions[next_state].values())
            )
            observations.append(next_output)

            # Update the current state
            current_state = next_state

        return Observation(stateseq, observations)


    def forward(self, observations):
        M = np.zeros((len(self.states), len(observations)))  # Use self.states to define the size of the forward matrix
        start_probs = self.transitions['#']

        # Initialize the forward matrix with the start probabilities
        for i, state in enumerate(self.states):
            if observations[0] in self.emissions[state]:  # Check if the first observation is possible for the state
                M[i, 0] = start_probs.get(state, 0) * self.emissions[state].get(observations[0], 0)

        # Iterate over the rest of the observations
        for t in range(1, len(observations)):
            for s_to_idx, s_to in enumerate(self.states):
                for s_from_idx, s_from in enumerate(self.states):
                    if observations[t] in self.emissions[s_to]:  # Check if the observation is possible for the state
                        M[s_to_idx, t] += (M[s_from_idx, t-1] * self.transitions[s_from].get(s_to, 0) * self.emissions[s_to].get(observations[t], 0))

        # Probability of the observation sequence is the sum of the final column
        prob_obs_sequence = np.sum(M[:, -1])
        return prob_obs_sequence

    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
        num_states = len(self.transitions) - 1  # Exclude the start state
        num_time_steps = len(observation)
        
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


def test_load(basename):
    model = HMM()
    model.load(basename)
    print(model.transitions)
    print(model.emissions)

def test_generate(basename):
    model = HMM()
    model.load(basename)
    print(model.generate(10))

def test_forward(basename):
    model = HMM()
    model.load(basename)
    
    #For every line in ambiguous_sents, split the line into words and make them a list
    with open('ambiguous_sents.obs', 'r') as file:
        observations = [line.strip().split() for line in file]
        for observation in observations:
            sequence = model.forward(observation)
            print("Forward probability of observation sequence: ", observation, " is ", sequence)

def test_viterbi(basename):
    model = HMM()
    model.load(basename)
    observations = model.generate(10)
    print(model.viterbi(observations))

test_load('two_english')
test_generate('two_english')
test_forward('partofspeech.browntags.trained')
# test_viterbi('two_english')

