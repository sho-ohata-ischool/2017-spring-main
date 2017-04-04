import cky

import collections
import unittest

class MockGrammar(object):
    def __init__(self, parsing_index):
        self.parsing_index = collections.defaultdict(list)
        self.parsing_index.update(parsing_index)

class TestParsing(unittest.TestCase):

    def test_failing_rule_application(self):
        chart = cky.make_chart()
        grammar = MockGrammar({
            ('the',): [('DT', -4)],
            ('potato',): [('N', -3), ('V', -20)],})

        self.assertFalse(cky.CKY_apply_preterminal_rules(
                'the rock'.split(), grammar, chart))

    def test_rule_application(self):
        chart = cky.make_chart()
        grammar = MockGrammar({
            ('the',): [('DT', -4)],
            ('potato',): [('N', -3), ('V', -20)],
            ('DT', 'N'): [('NP', -1), ('VP', -300)],
            ('DT', 'V'): [('VP', -2)]})

        # Verify preterminal rule application.
        self.assertTrue(cky.CKY_apply_preterminal_rules(
                'the potato'.split(), grammar, chart))

        self.assertSetEqual(set([(0, 1), (1, 2)]), set(chart.keys()))
        self.assertSetEqual(set(['DT']), set(chart[(0, 1)].keys()))
        self.assertEqual(chart[(0, 1)]['DT'].label(), 'DT')
        self.assertEqual(chart[(0, 1)]['DT'].logprob(), -4)
        self.assertEqual(len(chart[(0, 1)]['DT'].leaves()), 1)
        self.assertEqual(chart[(0, 1)]['DT'].leaves()[0], 'the')

        self.assertSetEqual(set(['N', 'V']), set(chart[(1, 2)].keys()))
        self.assertEqual(chart[(1, 2)]['N'].label(), 'N')
        self.assertEqual(chart[(1, 2)]['N'].logprob(), -3)
        self.assertEqual(len(chart[(1, 2)]['N'].leaves()), 1)
        self.assertEqual(chart[(1, 2)]['N'].leaves()[0], 'potato')

        self.assertEqual(chart[(1, 2)]['V'].label(), 'V')
        self.assertEqual(chart[(1, 2)]['V'].logprob(), -20)
        self.assertEqual(len(chart[(1, 2)]['V'].leaves()), 1)
        self.assertEqual(chart[(1, 2)]['V'].leaves()[0], 'potato')

        # Verify binary rule application.
        cky.CKY_apply_binary_rules(2, grammar, chart)
        self.assertSetEqual(set([(0, 1), (1, 2), (0, 2)]), set(chart.keys()))
        self.assertEqual(set(chart[(0, 2)].keys()), set(['NP', 'VP']))
        self.assertEqual(chart[(0, 2)]['NP'].label(), 'NP')
        self.assertEqual(chart[(0, 2)]['NP'].logprob(), -8)
        self.assertEqual(chart[(0, 2)]['VP'].label(), 'VP')
        self.assertEqual(chart[(0, 2)]['VP'].logprob(), -26)
        self.assertEqual(chart[(0, 2)]['VP'][1].label(), 'V')


if __name__ == '__main__':
    unittest.main()
