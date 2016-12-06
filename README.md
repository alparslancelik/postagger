# postagger
Part-of-speech tagger using Hidden Markov Model

The command for training (and tuning) the POS tagger is:
```java
java build_tagger sents.train sents.devt model_file
```

The file model_file contains the statistics gathered from the training (and tuning) process, which include the POS tag transition probabilities and the word emission probabilities (and other tuned parameters).

The test file consists of a list of sentences (without POS tags), one sentence per line. A sample test file is provided (sents.test).
The command to test on this test file and generate an output file is:   

```java
java run_tagger sents.test model_file sents.out
```
The output file has the same format as the POS-tagged training file. A sample output file is also provided (sents.out).

**Dataset**  
Penn Treebank tag set is used.  
sents.train ->  A training set of POS-tagged sentences  
sents.devt  ->  A separate development set of sentences for tuning the POS tagger  
