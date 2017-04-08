"""Implementation of methods for part of speech tagging using HMMs."""

import numpy as np
from scipy.misc import logsumexp

import collections
from collections import Counter, defaultdict

def logp_default_dict(*dict_params):
    return defaultdict(
            lambda: float('-inf'),  # Missing keys are -inf.
            *dict_params)

def normalize_as_logp(counts):
    log_total = np.log(sum(counts.itervalues()))
    return logp_default_dict(
            {k:(np.log(counts[k]) - log_total) for k in counts})

class HMM(object):

    def __init__(self):
        # c(y_0): counts for y_0
        # string -> int
        self.initial_counts = Counter()

        # c(yy'): counts for (y_{i-1}, y_i)
        # string (y_{i-1}) -> string (y_i) -> int
        self.transition_counts = defaultdict(lambda: Counter())

        # c(x|y): counts for tag, word
        # string (tag) -> string (word) -> int
        self.emission_counts = defaultdict(lambda: Counter())

        # Log probabilities, computed by compute_logprobs()
        # Types are same as above, but with float values.
        self.initial = logp_default_dict()
        self.transition = defaultdict(logp_default_dict)
        self.emission = defaultdict(logp_default_dict)


    def update_counts(self, tagged_sentence):
        """Accumulate counts of initial states, transitions, and emissions.

        Updates self.initial_counts, self.transition_counts, and self.emission_counts, as defined
        in the constructor above.

        These types are defined as defaultdicts and counters, so you don't need
        to initialize anything. Just do:
            self.initial_counts[tag] += 1
        or
            self.transition_counts[tag][tag] += 1
        as appropriate.

        Args:
            tagged_sentence: list((string, string)) list of (word, tag) tuples
        """

        for i, (w, t) in enumerate(tagged_sentence):
            if i == 0:
                # Sequence start count
                self.initial_counts[t] += 1
            else:
                # Transition from last state
                t_1 = tagged_sentence[i-1][1]
                self.transition_counts[t_1][t] += 1
            # Emission counts
            self.emission_counts[t][w] += 1

    def compute_logprobs(self):
        """Compute log-probabilities.

        Compute log-probabilities from the counts. Remember that self.transition
        and self.emission should be nested dicts, with the first key being the
        tag that the inner dict is conditioned on.

        Hint: use the normalize_as_logp() function, and keep this simple!
        You may want to refer back to Assignment 2, Part 1.
        """
        # Initial.
        self.initial = normalize_as_logp(self.initial_counts)

        # Transition.
        self.transition = defaultdict(
                logp_default_dict,
                {k: normalize_as_logp(self.transition_counts[k])
                    for k in self.transition_counts})

        # Emission.
        self.emission = defaultdict(
                logp_default_dict,
                {k: normalize_as_logp(self.emission_counts[k])
                    for k in self.emission_counts})

        # Compute set of all known POS tags.
        self.tagset = set(self.emission.keys())

    ##
    # Forward-backward inference
    def forward(self, sentence):
        """Run the Forward algorithm to compute alpha.

        We'll implement alpha as a dict, where the keys are tuples (i,tag) and
        the values are log-probabilities.
        So alpha[(3,'N')]  in the code is equal to log(alpha(3,'N')) in the
        equations in the writeup.

        Your alpha table should have entries for each i = 0, 1, ..., N and each 
        possible tag in self.tagset.

        Args:
            sentence: list(string) sequence to tag

        Returns:
            alpha: dict((int, string) -> float) forward beliefs, as
                log-probabilities.
        """
        alpha = dict()
        #### YOUR CODE HERE ####
        # Iterate through the sentence from left to right.
        for i, w in enumerate(sentence):
            for t in self.tagset:
                if i == 0:
                    alpha[(0, t)] = self.initial[t] + self.emission[t][w]
                else:
                    alpha[(i, t)] = logsumexp([self.emission[t][w] + self.transition[t_1][t] + alpha[(i-1, t_1)]
                        for t_1 in self.tagset])

        # Hint:  if you fail the unit tests, print out your alpha here
        #        and check it manually against the tests.
        # print alpha

        #### END(YOUR CODE) ####
        return alpha

    def backward(self, sentence):
        """Run the Backward algorithm to compute beta.

        We'll implement beta as a dict, where the keys are tuples (i,tag) and
        the values are log-probabilities.
        So beta[(3,'N')]  in the code is equal to log(beta(3,'N')) in the
        equations in the writeup.

        Args:
            sentence: list(string) sequence to tag

        Returns:
            beta: dict((int, string) -> float) backward beliefs, as
                log-probabilities.
        """
        beta = dict()
        N = len(sentence)
        for t in self.tagset:
            beta[(N - 1,t)] = 0.0  # np.log(1.0)

        # This will count down n-1, n-2, ..., 1, 0
        for i, w in reversed(list(enumerate(sentence[1:]))):
            pass
            for t_1 in self.tagset:
                sum_terms = [self.emission[t][w] + self.transition[t_1][t] + beta[(i + 1, t)]
                        for t in self.tagset]
                beta[(i, t_1)] = logsumexp(sum_terms)
        
        return beta

    def forward_backward(self, sentence):
        """Determine POS tags according to forward-backward.

        Hint: use your functions for forward and backward.

        Args:
            sentence: list(string) sequence to tag

        Returns:
            list(string), the most likely POS tag for each word, as determined
            by forward-backward.
        """
        tags = []
        alpha = self.forward(sentence)
        beta = self.backward(sentence)

        # For each position...
        for i in xrange(len(sentence)):
            # ... compute the score for each possible tag...
            candidates = [(alpha[(i,t)] + beta[(i,t)], t) for t in self.tagset]
            # ... pick the one that scores highest.
            tags.append(max(candidates)[1])

        return tags


    def build_viterbi_delta(self, sentence):
        """Determine POS tags using the Viterbi algorithm.

        Hint:
          This code is nearly identical to the forward algorithm, except "max"
          takes the place of a summation.
          Note that in addition, you will need to keep backpointers (similar
          to best_cuts_with_trace in assignment 4) so that the sequence can be
          recovered.

        Args:
          sentence: list(string) sequence to tag

        Returns:
          delta: the "delta" table from Viterbi.
          bp: backpointers to determine which sequence generated that score.
        """

        # Viterbi table. Similar to alpha from the Forward algorithm, this is a
        # map of (i, tag) -> log probability
        delta = dict()
        # Backpointers, map of (i, tag) -> previous tag
        bp = dict()

        #### YOUR CODE HERE ####
        for i, w in enumerate(sentence):
            for t in self.tagset:
                if i == 0:
                    delta[(0, t)] = self.initial[t] + self.emission[t][w]
                    bp[(0,t)] = None
                    prev_tag = t
                else:
                    delta[(i, t)] = logsumexp([self.emission[t][w] + max([self.transition[t_1][t] + delta[(i-1, t_1)]
                                            for t_1 in self.tagset])])
                    bp[(i,t)] = prev_tag
                    prev_tag = t
                    

        #### END(YOUR CODE) ####
        print bp

        return delta, bp

    def viterbi(self, sentence):
        """Determine POS tags using the Viterbi algorithm.

        Args:
            sentence: list(string) sequence to tag

        Returns:
            list(string), the most likely sequence of POS tags for the
            sentence.
        """
        # Build DP table.
        delta, bp = self.build_viterbi_delta(sentence)

        # Find the sequence that scores best.
        n = len(sentence)
        end_score, end_tag = max([(delta[(n-1,t)], t) for t in self.tagset])

        # Follow backpointers to obtain sequence with the highest score.
        tags = [end_tag]
        for i in reversed(range(0, n-1)):
            tags.append(bp[(i + 1,tags[-1])])
        return list(reversed(tags))
