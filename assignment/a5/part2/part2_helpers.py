
import nltk
from nltk.tree import Tree
import types

def verify_ptb_install():
    # Download PTB metadata
    assert(nltk.download('ptb'))

    import hashlib
    from nltk.corpus import ptb
    # Be sure we have the category list
    assert('news' in ptb.categories())

    m = hashlib.md5()  # NOT SECURE!
    m.update(','.join(ptb.fileids()))
    if m.hexdigest() == 'e3b49c6df5529560b2945e6a4715f9b0':
        print 'Penn Treebank succesfully installed!'
        return True
    else:
        print 'Error installing Penn Treebank (hash mismatch).'
        print 'It may still work - try loading it in NLTK.'
        return False

##
# Tree preprocessing functions
def get_np_real_child(node):
    """Find the real child, skipping over injected 'NP-*' cross-reference
    nodes.

    Args:
        node: (nltk.tree.Tree) nonterminal node

    Returns:
        list(nltk.tree.Tree) list of processed child nodes
    """
    if type(node) == types.UnicodeType:
        return [node]
    if 'NONE' in node.label():
        return []

    real_children = []
    if node.label().startswith('NP-'):
        for child in node:
            real_children.extend(get_np_real_child(child))
    else:
        real_children.append(node)
    return map(copy_strip_np_sbj, real_children)

def copy_strip_np_sbj(tree):
    """Make a copy of a Tree, stripping NP-* cross-reference nodes.

    Args:
        tree: (nltk.tree.Tree) nonterminal node

    Returns:
        nltk.tree.Tree, a new tree with cross-reference nodes removed.
    """
    children = []
    for child in tree:
        children.extend(get_np_real_child(child))
    return Tree(tree.label(), children)

