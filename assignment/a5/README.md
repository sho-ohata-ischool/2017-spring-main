# Assignment 5: Part of Speech and Parsing

## Due: April 19, 11:59p Pacific

This assignment consists of two parts:

* [Part of Speech](part1/Part-of-Speech.ipynb)
* [Parsing](part2/CKY.ipynb)

### The Penn Treebank (optional)

NLTK includes the `treebank` corpus, which is an abbreviated sample (3900 sentences) of the full (73k sentence) corpus.

The full corpus is available through Berkeley for research and academic purposes. We've included a copy in `ptb.zip` in this directory, along with a script that will install it to the proper directory for NLTK to access. Run as:
```
./install_ptb.sh
```
If it installs successfully, you can substitute `nltk.corpus.ptb` for `nltk.corpus.treebank`, and most functions should work normally - but with access to much more data. See [NLTK - Parsed Corpora](http://www.nltk.org/howto/corpus.html#parsed-corpora) for more information.

## Submission instructions

As always, you should commit your changes often with `git add` and `git commit`. As this is a new assignment, there may be bugfixes and corrections - so keep your eye on Piazza and GitHub for any updates.

Please submit by running the submit script:
```
./assignment/submit.sh -u your-github-username -a 5
```
You can view your work in your usual submission repo on the `a5-submit` branch.
