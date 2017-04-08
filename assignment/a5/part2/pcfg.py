# An elegant weapon, for a more civilized age
from __future__ import division

import numpy as np
import nltk
from collections import Counter, defaultdict

class PCFG(object):
    """Simple implementation of a Probabilistic Context-Free Grammar.

    This uses NLTK data structures, and is similar to nltk.grammar.PCFG but
    with more efficient lookups and the added advantage of you, the student,
    getting to write key parts of the implementation!
    """

    def __init__(self):
        # Map of nltk.grammar.Production -> int
        self.production_counts = Counter()
        # Map of nltk.grammar.Nonterminal -> int
        self.lhs_counts = Counter()

        # See compute_scores
        self.scored_productions = None
        # See build_index
        self.parsing_index = None

    def top_productions(self, n=10):
        return self.production_counts.most_common(n)

    def top_lhs(self, n=10):
        return self.lhs_counts.most_common(n)

    def update_counts(self, parsed_sentence):
        """Accumulate counts of productions from a single sentence.

        Updates self.production_counts and self.lhs_counts, incrementing counts
        by 1 for each production seen and each lhs seen.

        Args:
            parsed_sentence: nltk.tree.Tree

        Returns:
            None
        """
        pass
        #### YOUR CODE HERE ####
	for production in parsed_sentence.productions():
	    lhs = production.lhs()
	    self.production_counts[production] += 1
	    self.lhs_counts[lhs] += 1


        #### END(YOUR CODE) ####

    def compute_scores(self):
        """Compute log-probabilities.

        Populate self.scored_productions, which has the same keys as
        self.production_counts but where the values are the log probabilities
        log(p) = log(numerator) - log(denominator), according to the equation
        in the notebook.
        """
        # Map of nltk.grammar.Production -> float
        self.scored_productions = dict()

        #### YOUR CODE HERE ####
	for key in self.production_counts:
	    numerator = np.log(self.production_counts[key])
	    denominator = np.log(self.lhs_counts[key.lhs()])
	    self.scored_productions[key] = numerator - denominator


        #### END(YOUR CODE) ####

    def build_index(self):
        """Index productions by RHS, for use in bottom-up parsing.

        This should be run after compute_scores()
        """
        # Map of tuple(nltk.grammar.Nonterminal) ->
        #                  list((nltk.grammar.Nonterminal, double))
        # Maps from RHS to (LHS, score)
        self.parsing_index = defaultdict(list)
        for production in self.scored_productions:
            score = self.scored_productions[production]
            l = self.parsing_index[production.rhs()]
            l.append((production.lhs(), score))










