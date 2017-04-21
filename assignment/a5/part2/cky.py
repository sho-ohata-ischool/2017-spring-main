
import collections
import itertools

import numpy as np
import nltk
from nltk.tree import Tree, ProbabilisticTree

import pcfg

def make_chart():
    dummy_tree_factory = lambda: ProbabilisticTree('', [], logprob=-np.inf)
    cell_factory = lambda: collections.defaultdict(dummy_tree_factory)
    return collections.defaultdict(cell_factory)

def ordered_spans(n):
    """Generate all spans, sorted bottom-up.

    Returns a list all spans [i,j) where 0 <= i < j <= n, sorted by the length
    of the span (j - i).

    For example, ordered_spans(4) would return spans:
        [0, 1) [1, 2) [2, 3) [3, 4)
        [0, 2) [1, 3) [2, 4)
        [0, 3) [1, 4)
        [0, 4)
    """
    key_fn = lambda (a, b): b - a
    return sorted([(i,j) for i in xrange(n)
                         for j in xrange(i+1, n+1)], key=key_fn)


def CKY_apply_preterminal_rules(words, grammar, chart):
    """Populate the bottom row of the CKY chart.

       Specifically, apply preterminal unary rules to go from word to preterminal.

       Hint:  use grammar.parsing_index[(word,)] to enumerate available unary rules and their
              corresponding scores.
       Hint:  A `chart` is a two level structure.  The first key is a tuple representing the span.
              the second key is a part of speech that can be produced by that span.
              Finally, the value is a ProbabilisticTree containing the score of the best way to create
              that part of speech.  As with A4 best_cuts_with_trace, it also maintains some book keeping
              to know how to create it.  Concretely...

              chart[(i, i+1)][pos_tag] = ProbabilisticTree(pos_tag, [word], logprob=score)

       Args:
         - words: sequence of words to parse
         - grammar: the grammar to parse with
         - chart: the chart to populate

       Returns: False if a preterminal could not be found in the grammar for a word.
                True otherwise.
    """
    #### YOUR CODE HERE ####
    # Handle preterminal rules A -> a
    # For the ith token, you should populate cell (i,i+1).
    for i, word in enumerate(words):
	if (word,) not in grammar.parsing_index:
	    return False

	else:
	    for x in range(len(grammar.parsing_index[(word,)])):
	        pos_tag, score = grammar.parsing_index[(word,)][x]
    	        chart[(i, i+1)][pos_tag] = ProbabilisticTree(pos_tag, [word], logprob=score)

    #### END(YOUR CODE) ####
    return True

def CKY_apply_binary_rules(N, grammar, chart):
    """Populate the remainder of the chart, assuming the bottom row is complete.

       Iterating throught the chart from the bottom up, apply all available
       binary rules at each position in the chart.  Each cell of the chart should
       enumerate the heads that can be produced there and the score corresponding
       to their most efficient construction.

       Hint: grammar.parsing_index[(B, C)] will return a list of binary
             production rules of the form A -> B, C along with their score.

       Args:
         - N: the number of words
         - grammar: the grammar to use to parse
         - chart: the chart to populate, see CKY_apply_preterminal_rules for a detailed description.
    """
    #### YOUR CODE HERE ####
    # Iterate through the chart, handling nonterminal rules A -> B C
    # Use the ordered_spans function to get a list of spans from the bottom up.
    for (i, j) in ordered_spans(N):
        for split in xrange(i+1, j):
            # Consider all possible A -> B C
	    for rule, value in grammar.parsing_index.iteritems():
	        B = chart[(i, split)][rule]
		C = chart[(split, j)][rule]
		for pos_tag, score in value:
		    #B = chart[(i, split)][rule]
		    #C = chart[(split, j)][rule]
		    sum_score = score + B.logprob() + C.logprob()

		    if pos_tag not in chart[(i,j)]:
			chart[(i,j)][pos_tag] = ProbabilisticTree(pos_tag, [B, C], score=sum_score)
			#chart[(i,j)][pos_tag] = ProbabilisticTree(pos_tag, [B, C], score=sum_score)

                    else:
		        if sum_score > chart[(i,j)][pos_tag].logprob():
                            #chart[(i,j)][pos_tag] = ProbabilisticTree(pos_tag, [B, C], score=sum_score)
			    chart[(i,j)][pos_tag] = ProbabilisticTree(pos_tag, [B, C], score=sum_score)

    print set(chart[(0,2)].keys())
    #### END(YOUR CODE) ####

def CKY(words, grammar, target_type=None):
    """Run the CKY chart-parsing algorithm with the given grammar.

    Given a sequence of words and a weighted context-free grammar, finds the
    most likely derivation Tree.

    Args:
        words: list(string) sentence to parse
        grammar: (pcfg.PCFG) weighted CFG
        target_type: (string OR nltk.grammar.Nonterminal) if specified, will
            return the highest scoring derivation of that type (e.g. 'S'). If
            None, will return the highest scoring derivation of any type.

    Returns:
        (nltk.tree.ProbabilisticTree): optimal derivation
    """
    assert(isinstance(grammar, pcfg.PCFG))
    assert(grammar.parsing_index is not None)

    # The chart is a map from span -> symbol -> derivation
    # Formally: tuple(int, int) -> nltk.grammar.Nonterminal
    #                                  -> nltk.tree.ProbabilisticTree
    # The defaultdict machinery (see Assignment 2, Part 1) will populate cells
    # with dummy entries so you don't need to handle the special case of
    # an empty cell in your inner loop.
    # (See make_chart earlier in this file if you want to see this in action.)
    chart = make_chart()

    N = len(words)

    # Words -> preterminals via unary rules.
    # (i.e. populate the bottom row of the chart)
    if not CKY_apply_preterminal_rules(words, grammar, chart):
        return (None, None)

    # Populate the rest of the chart from binary rules.
    CKY_apply_binary_rules(N, grammar, chart)

    # Verify we were able to produce something in the cell spanning the
    # entire sentence.  If we can't, the sentence isn't parseable under
    # our grammar.
    if len(chart[(0,N)]) == 0:
        return (None, None)

    # Final parse tree is just the best-scoring derivation in the top cell.
    # (If a specific target_type is requested - often 'S' - return that parse
    # tree even if there is a higher scoring parse available.)
    if target_type is None:
        return max(chart[(0,N)].values(), key=lambda t: t.logprob())
    else:
        return chart[(0,N)][nltk.grammar.Nonterminal(target_type)]

