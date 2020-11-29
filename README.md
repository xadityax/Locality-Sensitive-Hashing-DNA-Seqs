### Locality-Sensitive-Hashing-DNA-Seqs
Implementing Locality Sensitive Hashing for DNA Sequences.

### Preprocessing
DNA sequence dataset from Kaggle is used. Classes of the DNA sequences are seperated from the data and only the sequence is used.

### Shingling
Size of shingles is taken as input. 5-10 recommended.

### Minhashing
Random permutations of pseudo indices are used to generate signatures for sequences (documents). Number of permutations is taken as input.

### LSH
Number of bands is taken as input and sequences in the same band are hashed into buckets where two sequences from the same band have high probability of going into the same bucket if they are similar.

### GUI
The GUI is built with Tkinter in python. 

### How to run
Please enter number of bands, number of permutations and size of shingles before giving corpus directory input to start LSH.
