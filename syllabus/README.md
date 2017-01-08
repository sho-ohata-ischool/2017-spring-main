# DATASCI W266: Natural Language Processing

[Course Overview](#course-overview)  
[Grading](#grading)  
[Final Project](#final-project)  
[Course Resources](#course-resources)  
[Schedule and Readings](#schedule-and-readings)  

## Course Overview
Understanding language is fundamental to human interaction. Our brains have evolved language-specific circuitry that helps us learn it very quickly; however, this also means that we have great difficulty explaining how exactly meaning arises from sounds and symbols. This course is a broad introduction to linguistic phenomena and our attempts to analyze them with machine learning. We will cover a wide range of concepts with a focus on practical applications such as information extraction, machine translation, sentiment analysis, and summarization.
   
**Prerequisite:**
[MIDS 207 (Machine Learning)](https://www.ischool.berkeley.edu/courses/datasci/207)

**Live Sessions:**
* Monday 6:30p - 8p Pacific ([James Kunz](mailto:jkunz@ischool.berkeley.edu))
* Tuesday 6:30p - 8p Pacific ([Ian Tenney](mailto:iftenney@ischool.berkeley.edu))
* Friday 4p - 5:30p Pacific ([James Kunz](mailto:jkunz@ischool.berkeley.edu))

**Office Hours:**
* TBD - please enter your preferences in the [poll](http://doodle.com/poll/679ybxc3w96umfpu)

**Course Assistant:**
* [Drew Plant](mailto:drewplant@berkeley.edu)

**Async Instructors:**
* [Dan Gillick](mailto:dgillick@gmail.com)
* [Kuzman Ganchev](mailto:kuzman.ganchev@gmail.com)

**Contacts and resources:**
* Course website: [GitHub datasci-w266/2017-spring-main](../../../)
* [Piazza](http://piazza.com/berkeley/spring2017/datasciw266) - we'll use this for Q&A, and this will be the fastest way to reach the course staff. Note that you can post anonymously, and/or make posts visible only to instructors for private questions.
* Email list for course staff: mids-nlp-instructors@googlegroups.com

## Grading
### Breakdown

Your grade will be determined as follows:
* **Participation**: 10%
* **Assignments**: 50%
* **Final Project**: 40%

There will be a number of smaller [assignments](../../../tree/master/assignment/) throughout the term for you to
exercise what you learned in async and live sessions. Some assignments may be more difficult than others, and will be weighted accordingly.

Participation will be graded holistically, based on live session attendance and participation as well as participation on Piazza. (Don’t stress about this part.)

### Late Day Policy

We recognize that sometimes things happen in life outside the course, especially in MIDS where we all have full time jobs and family responsibilities to attend to. To help with these situations, we are giving you **5 "late days"** to use throughout the term as you see fit.  Each late day gives you a 24 hour (or any part thereof) extension to any deliverable in the course **except** the final project presentation or report. (UC Berkeley needs grades submitted very shortly after the end of classes.)

Once you run out of late days, each 24 hour period (or any part thereof) results in a **10 percentage point deduction** on that deliverable's grade.

You can use a maximum of 2 late days on any single deliverable, and we will not be accepting any submissions more than 48 hours past the original due-date. (We want to be more flexible here, but your fellow students also want their graded assignments back promptly!)

We don't anticipate granting extensions beyond these policies.  Plan your time accordingly!

### More serious issues

If you run into a more serious issue that will affect your ability to complete the course, please contact the instructors and MIDS student services.  A word of warning though: in previous sections, we have had students ask for INC grades because their lives were otherwise busy.  Mostly we have declined, opting instead for the student to complete the course to the best of their ability and have a grade assigned based on that work.  (MIDS prefers to avoid giving INCs, as they have been abused in the past.)

## Final Project
*See the [Final Project Guidelines](../../../tree/master/project/)*

## Course Resources
We are not using any particular textbook for this course. We’ll list some relevant readings each week. Here are some general resources:
* [Jurafsky and Martin: Speech and Language Processing](http://www.cs.colorado.edu/~martin/slp.html)
* [NLTK Book](http://www.nltk.org/book/) accompanies NLTK (Natural Language ToolKit) and includes useful, practical descriptions (with python code) of basic concepts.

We’ll be posting materials to the course [GitHub repo](../../../).

*Note:* this is a new class, and the syllabus below might be subject to change. We'll be sure to announce anything major on [Piazza](http://piazza.com/berkeley/spring2017/datasciw266).

## Code References

The course will be taught in Python, and we'll be making heavy use of NumPy, TensorFlow, and Jupyter (IPython) notebooks. We'll also be using Git for distributing and submitting materials. If you want to brush up on any of these, we recommend:
* **Git tutorials:** [Introduction / Cheat Sheet](https://git-scm.com/docs/gittutorial), or [interactive tutorial](https://try.github.io/)
* **Python / NumPy:** Stanford's CS231n has [an excellent tutorial](http://cs231n.github.io/python-numpy-tutorial/)
* **TensorFlow:** We'll go over the basics of TensorFlow in [Assignment 1](../../../tree/master/assignment/a1/). You can also check out the [tutorials](https://www.tensorflow.org/get_started/) on the TensorFlow website, but these can be somewhat confusing if you're not familiar with the underlying models.


## Misc. Deep Learning and NLP References
A few useful papers that don’t fit under a particular week. All optional, but interesting!
* (optional) [Chris Olah’s blog](http://colah.github.io/)
* (optional) [Natural Language Processing (almost) from Scratch (Collobert and Weston, 2011)](https://arxiv.org/pdf/1103.0398v1.pdf)
* (optional) [GloVe: Global Vectors for Word Representation (Pennington, Socher, and Manning, 2014)](http://nlp.stanford.edu/pubs/glove.pdf)

---

## Schedule and Readings

We'll update the table below with assignments as they become available, as well as additional materials throughout the quarter. Keep an eye on GitHub for updates!

<table>
<tr>
<th></th>
<th>Subject</th>
<th>Topics</th>
<th>Materials</th>
</tr>
<tr><!--- Week 1 -->
  <td><strong>Week&nbsp;1</strong><br>(Jan.&nbsp;8&nbsp;-&nbsp;14)</td>
  <td>Introduction</td>
  <td><ul>
    <li>Overview of NLP applications
    <li>Ambiguity in language
    <li>General concepts
  </ul></td>
  <td><ul>
  <li>Skim: <a href="http://www.nltk.org/book/ch01.html" target="_blank">NLTK book chapter 1 (python and basics)</a>
  <li>Skim: <a href="http://www.nltk.org/book_1ed/ch02.html" target="_blank">NLTK book chapter 2 (data resources)</a>
  <li>Read: <a href="https://www.technologyreview.com/s/602094/ais-language-problem/" target="_blank">AI’s Language Problem (Technology Review)</a>
  <li><em>Optional:</em> <a href="http://www.newyorker.com/magazine/2007/04/16/the-interpreter-2" target="_blank">The Interpreter (New Yorker)</a>
  <li><em>Optional:</em> <a href="https://www.uio.no/studier/emner/hf/ikos/EXFAC03-AAS/h05/larestoff/linguistics/Chapter%204.(H05).pdf" target="_blank">Introduction to Linguistic Typology</a>
  </ul></td>
</tr>
<tr><!--- Week 1 Assignment -->
  <td><strong>Assignment&nbsp;0</strong><br><em>due&nbsp;Jan.&nbsp;15</em></td>
  <td><strong>Course Set-up</strong></td>
  <td><ul>
    <li>GitHub
    <li>Google Cloud
  </ul></td>
  <td><em>Link to-be-released</em></td>
</tr>
<tr><!--- Week 2 -->
  <td><strong>Week&nbsp;2</strong><br>(Jan.&nbsp;15&nbsp;-&nbsp;21)</td>
  <td>Language Modeling I</td>
  <td><ul>
    <li>LM applications
    <li>N-gram models
    <li>Smoothing methods
    <li>Text generation
  </ul></td>
  <td><ul>
  <li>Skim: <a href="http://www.cs.berkeley.edu/~klein/cs294-5/chen_goodman.pdf" target="_blank">Chen and Goodman Survey</a>
  <li>Skim: <a href="http://arxiv.org/pdf/1312.3005.pdf" target="_blank">1 Billion Word Benchmark</a>
  <li><em>Optional:</em> <a href="http://norvig.com/ngrams/ch14.pdf" target="_blank">Natural Language Corpus Data (Peter Norvig)</a>
  </ul></td>
</tr>
<tr><!--- Week 2 Assignment -->
  <td><strong>Assignment&nbsp;1</strong><br><em>due&nbsp;Jan.&nbsp;22</em></td>
  <td><strong>Background and TensorFlow</strong></td>
  <td><ul>
    <li>Dynamic Programming
    <li>Information Theory
    <li>TensorFlow tutorial
  </ul></td>
  <td><em>Link to-be-released</em></td>
</tr>
<tr><!--- Week 3 -->
  <td><strong>Week&nbsp;3</strong><br>(Jan.&nbsp;22&nbsp;-&nbsp;28)</td>
  <td>Clusters and Distributions</td>
  <td><ul>
  <li>Representations of meaning
  <li>Word classes
  <li>Word vectors via co-occurrence counts
  <li>Word vectors via prediction (word2vec)
  </ul></td>
  <td><ul>
  <li>Read: <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.13.9919&rep=rep1&type=pdf" target="_blank">Brown Clustering</a> (Brown et al. 1992)
  <li>Read: <a href="http://arxiv.org/pdf/1301.3781.pdf" target="_blank">CBOW and SkipGram</a> (Mikolov et al. 2013)
  <li><em>Optional:</em> <a href="http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/" target="_blank">Deep Learning, NLP, and Representations</a> (Chris Olah's blog)
  <li><em>Optional:</em> <a href="https://www.tensorflow.org/versions/master/tutorials/word2vec/index.html" target="_blank">Tensorflow Word2Vec Tutorial</a> (just the parts on <a href="https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/examples/tutorials/word2vec/word2vec_basic.py" target="_blank">word2vec_basic.py</a> - don’t bother with the “Optimizing the Implementation” part or anything in C++)
  <li><em>Optional:</em> <a href="https://www.technologyreview.com/s/602025/how-vector-space-mathematics-reveals-the-hidden-sexism-in-language/" target="_blank">How Vector Space Mathematics Reveals the Hidden Sexism in Language</a> (and the <a href="http://arxiv.org/abs/1607.06520)" target="_blank">original paper</a>)
  </ul></td>
</tr>
<tr><!--- Week 4 -->
  <td><strong>Week&nbsp;4</strong><br>(Jan.&nbsp;29&nbsp;-&nbsp;Feb.&nbsp;4)</td>
  <td>Language Modeling II</td>
  <td><ul>
  <li>Neural Net LMs
  <li>Word embeddings
  <li>Hierarchical softmax
  <li>State of the art: Recurrent Neural Nets
  </ul></td>
  <td><ul>
  <li>Read: <a href="http://machinelearning.wustl.edu/mlpapers/paper_files/BengioDVJ03.pdf" target="_blank">A Neural Probabilistic Language Model</a> (Bengio et al. 2003)
  <li>Read or skim: <a href="http://neuralnetworksanddeeplearning.com/chap2.html" target="_blank">How the backpropagation algorithm works</a>
  <li><em>Optional:</em> <a href="http://colah.github.io/posts/2015-08-Understanding-LSTMs/" target="_blank">Understanding LSTM Networks</a> (Chris Olah's blog)
  <li>Optional (skim): <a href="https://www.tensorflow.org/versions/master/tutorials/recurrent/index.html#recurrent-neural-networks" target="_blank">Tensorflow LSTM Language Model Tutorial</a>
  <li>Optional / fun: <a href="http://playground.tensorflow.org/" target="_blank">Tensorflow Playground</a>
  </ul></td>
</tr>
<tr><!--- Week 5 -->
  <td><strong>Week&nbsp;5</strong><br>(Feb.&nbsp;5&nbsp;-&nbsp;11)</td>
  <td>Basics of Text Processing</td>
  <td><ul>
  <li>Edit distance for strings
  <li>Tokenization
  <li>Sentence splitting
  </ul></td>
  <td><ul>
  <li>Skim: <a href="http://www.nltk.org/book_1ed/ch03.html" target="_blank">NLTK book chapter 3</a> (processing raw text)
  <li>Skim: <a href="http://norvig.com/ngrams/ch14.pdf" target="_blank">Natural Language Corpus Data</a> (Peter Norvig) <em>(if you didn't read in Week 2)</em>
  <li>Read: <a href="http://www.dgillick.com/resource/sbd_naacl_2009.pdf" target="_blank">Sentence Boundary Detection and the Problem with the U.S.</a>
  </ul></td>
</tr>
<tr><!--- Week 6 -->
  <td><strong>Week&nbsp;6</strong><br>(Feb.&nbsp;12&nbsp;-&nbsp;18)</td>
  <td>Information Retrieval</td>
  <td><ul>
  <li>Building a Search Index
  <li>Ranking
  <li>TF-IDF
  <li>Click signals
  </ul></td>
  <td><ul>
  <li>Read: <a href="http://static.googleusercontent.com/media/research.google.com/en//archive/googlecluster-ieee.pdf" target="_blank">Web Search for a Planet</a> (Google)
  <li>Read: <a href="http://infolab.stanford.edu/~backrub/google.html" target="_blank">The Anatomy of a Large-Scale Hypertextual Web Search Engine</a>
  <li>Skim: <a href="http://nlp.stanford.edu/IR-book/pdf/irbookprint.pdf" target="_blank">"An Introduction to Information Retrieval", sections 6.2 and 6.3</a>
  <li><em>Optional:</em> <a href="http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf" target="_blank">PageRank</a> (Page, et al. 1999)
  </ul></td>
</tr>
<tr><!--- Week 7 -->
  <td><strong>Week&nbsp;7</strong><br>(Feb.&nbsp;19&nbsp;-&nbsp;25)</td>
  <td>Part-of-Speech Tagging I</td>
  <td><ul>
  <li>Tag sets
  <li>Most frequent tag baseline
  <li>HMM/CRF models
  </ul></td>
  <td><ul>
  <li>Read: <a href="http://www.nltk.org/book_1ed/ch05.html" target="_blank">NLTK book chapter 5: Categorizing and Tagging Words</a>
  </ul></td>
</tr>
<tr><!--- Week 8 -->
  <td><strong>Week&nbsp;8</strong><br>(Feb.&nbsp;26&nbsp;-&nbsp;Mar.&nbsp;4)</td>
  <td>Part-of-Speech Tagging II</td>
  <td><ul>
  <li>Feature engineering
  <li>Leveraging unlabeled data
  <li>Low resource languages
  </ul></td>
  <td><ul>
  <li>Read: <a href="https://arxiv.org/pdf/1104.2086.pdf" target="_blank">A Universal Part-of-Speech Tagset</a>
  <li>Read: <a href="http://nlp.stanford.edu/pubs/CICLing2011-manning-tagging.pdf" target="_blank">Part-of-Speech Tagging from 97% to 100%: Is It Time for Some Linguistics?</a>
  </ul></td>
</tr>
<tr><!--- Week 9 -->
  <td><strong>Week&nbsp;9</strong><br>(Mar.&nbsp;5&nbsp;-&nbsp;11)</td>
  <td>Dependency Parsing</td>
  <td><ul>
  <li>Dependency trees
  <li>Transition-based parsing: Arc&#8209;standard, Arc&#8209;eager
  <li>Graph based parsing: Eisner Algorithm, Chu&#8209;Liu&#8209;Edmonds
  </ul></td>
  <td><ul>
  <li>Read: <a href="http://www.nltk.org/book_1ed/ch08.html" target="_blank">NLTK book chapter 8 (analyzing sentence structure)</a>
  </ul></td>
</tr>
<tr><!--- Week 10 -->
  <td><strong>Week&nbsp;10</strong><br>(Mar.&nbsp;12&nbsp;-&nbsp;18)</td>
  <td>Constituency Parsing</td>
  <td><ul>
  <li>Context-free grammars (CFGs)
  <li>CYK algorithm
  <li>Probabilistic CFGs
  <li>Lexicalized grammars, split-merge, and EM
  </ul></td>
  <td><em>Readings TBA</em></td>
  <!-- <td><ul>        -->
  <!-- <li>Placeholder -->
  <!-- </ul></td>      -->
</tr>
<tr><!--- Week 11 -->
  <td><strong>Week&nbsp;11</strong><br>(Mar.&nbsp;19&nbsp;-&nbsp;25)</td>
  <td>Entities</td>
  <td><ul>
  <li>From syntax to semantics
  <li>Named Entity Recognition
  <li>Coreference Resolution
  </ul></td>
  <td><ul>
  <li><a href="http://www.nltk.org/book_1ed/ch07.html" target="_blank">NLTK Book Chapter 7 (Extracting Information from Text)</a>
  </ul></td>
</tr>
<tr><!--- Spring Break -->
  <td><strong>Spring Break</strong><br>(Mar.&nbsp;26&nbsp;-&nbsp;Apr.&nbsp;1)</td>
  <td><em>(no class)</em></td>
</tr>
<tr><!--- Week 12 -->
  <td><strong>Week&nbsp;12</strong><br>(Apr.&nbsp;2&nbsp;-&nbsp;8)</td>
  <td>Machine Translation I</td>
  <td><ul>
  <li>Word-based translation models
  <li>IBM Models 1 and 2
  <li>HMM Models
  <li>Evaluation
  </ul></td>
  <td><ul>
  <li><a href="http://www.isi.edu/natural-language/mt/wkbk.rtf" target="_blank">Statistical MT Handbook by Kevin Knight</a>
  </ul></td>
</tr>
<tr><!--- Week 13 -->
  <td><strong>Week&nbsp;13</strong><br>(Apr.&nbsp;9&nbsp;-&nbsp;15)</td>
  <td>Machine Translation II</td>
  <td><ul>
  <li>Phrase-based translation
  <li>Neural MT via sequence-to-sequence models
  <li>Attention-based models
  </ul></td>
  <td><ul>
  <li><a href="http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf" target="_blank">Sequence to Sequence Learning with Neural Networks</a>
  <li><a href="https://arxiv.org/pdf/1409.0473.pdf" target="_blank">Neural Machine Translation by Jointly Learning to Align and Translate</a>
  <li><em>Optional:</em> <a href="https://arxiv.org/abs/1609.08144" target="_blank">Google’s Neural Machine Translation System</a>
  <li><em>Optional:</em> <a href="http://distill.pub/2016/augmented-rnns/#attentional-interfaces" target="_blank">Attention and Augmented Recurrent Neural Networks</a> (section on “Attentional Interfaces” has an awesome visualization of an MT example, showing alignments)
  </ul></td>
</tr>
<tr><!--- Week 14 -->
  <td><strong>Week&nbsp;14</strong><br>(Apr.&nbsp;16&nbsp;-&nbsp;22)</td>
  <td>Summarization</td>
  <td><ul>
  <li>Single- vs. multi-document summarization
  <li>Maximum marginal relevance (MMR) algorithm
  <li>Formulation of a summarization objective
  <li>Integer linear programming (ILP) for optimal solutions
  <li>Evaluation of summaries
  </ul></td>
  <td><em>Readings TBA</em></td>
  <!-- <td><ul>        -->
  <!-- <li>Placeholder -->
  <!-- </ul></td>      -->
</tr>
<tr><!--- Week 15 -->
  <td><strong>Week&nbsp;15</strong><br>(Apr.&nbsp;23&nbsp;-&nbsp;29)</td>
  <td>Sentiment Analysis</td>
  <td><ul>
  <li>Sentiment lexicons
  <li>Aggregated sentiment applications
  <li>Convolutional neural networks (CNNs)
  </ul></td>
  <td><em>Readings TBA</em></td>
  <!-- <td><ul>        -->
  <!-- <li>Placeholder -->
  <!-- </ul></td>      -->
</tr>
</table>

