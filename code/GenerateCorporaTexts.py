#!/usr/bin/python3

import nltk

def main():
	# Convert Gutenberg, Brown corpora into txts with each sentence on separate lines
	guten_files = nltk.corpus.gutenberg.fileids()
	gsentences = []
	for gfile in guten_files:
		sentences_lists = nltk.corpus.gutenberg.sents(gfile)
		sentences = [' '.join(s) for s in sentences_lists]
		gsentences.extend(sentences)

	with open('../docs/gutenberg.txt','w') as f:
		for gs in gsentences:
			f.write(gs + '\n')

	brown_files = nltk.corpus.brown.fileids()
	bsentences = []
	for bfile in brown_files:
		sentences_lists = nltk.corpus.brown.sents(bfile)
		sentences = [' '.join(s) for s in sentences_lists]
		bsentences.extend(sentences)

	with open('../docs/brown.txt','w') as f:
		for bs in bsentences:
			f.write(bs + '\n')

	print("yeehaw")

if __name__ == '__main__':
	main()
