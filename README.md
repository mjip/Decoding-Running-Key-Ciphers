# Decoding Running Key Ciphers: Natural Language Processing Final Project
Implementation of https://www.cs.mcgill.ca/~mpayne/docs/Griffing-Viterbi.pdf, 
comparing Viterbi decoding and unsmoothed/smoothed n-gram methods in decoding running
key ciphers with English keystreams and plaintexts. Uses the Brown and Project Gutenberg corpora.

## Usage
`python3 FinalProject.py [--path p] [--ngrams n] [--train tr] [--test te] [--length l]`
where p is the path to the folder for the source texts, tr and te are the training and testing corpora respectively,
n is the number of ngrams to train up until and l is the minimum length of the ciphertext. The defaults are docs/, 2, 
brown0000.txt, brown0000.txt, and 100 in order. 

Project by Victor Redko and Marie Payne  
