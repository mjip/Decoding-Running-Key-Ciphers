#!/usr/bin/python3

import random
import sys
import time
import string
import re
import argparse

from math import inf
from math import log
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

def accuracy(correct, guess):
    """ Evaluates how accurate the output of the testing method is.

    Checks whether each letter or vector of the guess matches the corresponding
    letter or vector of the correct sentence and returns how many are right, as
    a percent.

    Args:
        correct: string or vector that is the correct sentence.
        guess: string or vector that is the output of the testing method.

    Returns:
        The accuracy of the guess based on what the correct answer is.

        acc: float representing what percentage of vectors or characters in the
            guess match the character at the same index in the correct sentence.
    """

    acc = 0.
    if len(correct) == 0: return 0
    for n in range(len(correct)):
        if correct[n] == guess[n]: acc += 1.

    return acc / len(correct)

def devectorize(vector, n_grams):
    """ Transforms inputs from vectors to sentences.

    Goes through the vector and replaces each entry with its associated n-gram.
    For n > 1, the first vector is put in full and the rest have only their last
    letter put in.

    Args:
        vector: vector representing the sentence.
        n_grams: n-grams from which to create the sentence.

    Returns:
        plaintext: the plaintext sentence.
    """

    plaintext = [n_grams[vector[0]]]
    for n in range(1, len(vector)):
        plaintext.append(n_grams[vector[n]][-1])

    plaintext = ''.join(plaintext)
    return plaintext

def generate_ngrams(alphabet, limit):
    """ Generates all possible n-grams from 1 to the limit.

    Goes through the alphabet in a loop to generate a list of lists of n-grams
    from 1 to the limit.

    Args:
        alphabet: list of the symbols from which the n-grams will be generated.
        limit: the maximum length of the n-grams generated.

    Returns:
        n_grams: a list of lists of n-grams up to the limit.
    """

    print('Generating 1-grams')
    n_grams = [alphabet]
    for n in range(1, limit):
        print('Generating %d-grams' % (n + 1))
        new_grams = []
        for gram in n_grams[-1]:
            for letter in alphabet:
                new_grams.append('%s%s' % (gram, letter))

        n_grams.append(new_grams)

    print('Done generating n-grams\n')
    return n_grams

def get_hmms(sentences, n_grams, smoothing=True):
    """ Creates the hidden Markov models for all n-grams.

    Going through input sentences, the frequency of each n-gram is counted, as
    well as the n-gram which follows. These frequencies are smoothed with add-1
    smoothing by default, but may be unsmoothed.

    Args:
        sentences: list of sentences on which to train the HMMs.
        n_grams: list of lists of n-grams for which to make HMMs.
        smoothing: boolean for whether to smooth with add-1 smoothing or not.

    Returns:
        hmms: a list of dicts which represent the HMMs and their initial and
        transition probabilities, as well as the frequencies of the n-grams.
    """

    initial = 0.
    hmm_type = 'unsmoothed'

    if smoothing:
        initial = 1.
        hmm_type = 'smoothed'

    inits = [{} for grams in n_grams]
    inits_counts = [initial*len(grams) for grams in n_grams]
    transitions = [{} for grams in n_grams]
    trans_counts = [{} for grams in n_grams]
    freqs = [{} for grams in n_grams]
    freqs_counts = [initial*len(grams) for grams in n_grams]

    for n in range(len(n_grams)):
        print('Generating %s Markov model for %d-grams' % (hmm_type, (n+1)))
        for gram in n_grams[n]:
            inits[n].update({gram : initial})
            freqs[n].update({gram : initial})
            transitions[n].update({gram : {}})
            next_grams = get_next_gram(gram, n_grams[n])
            trans_counts[n].update({gram : initial * len(next_grams)})

            for next_gram in next_grams:
                transitions[n][gram].update({next_gram : initial})

    for sentence in sentences:
        if len(sentence) < len(n_grams): continue

        for n in range(len(n_grams)):
            start = sentence[:n+1]
            inits[n][start] += 1.
            inits_counts[n] += 1.

        for i in range(len(sentence)):
            for n in range(len(n_grams)):
                if i+n+1 < len(sentence):
                    gram = sentence[i:i+n+1]
                    freqs[n][gram] += 1.
                    freqs_counts[n] += 1.
                    
                    if i+n+2 < len(sentence) and n > 0:
                        next_gram = sentence[i+1:i+n+2]
                        transitions[n][gram][next_gram] += 1.
                        trans_counts[n][gram] += 1.

    for n in range(len(n_grams)):
        for gram in inits[n].keys():
            prob = inits[n][gram] / inits_counts[n]
            if prob == 0:
                inits[n][gram] = -1 * inf
            else:
                inits[n][gram] = log(prob)

        for gram in freqs[n].keys():
            prob = freqs[n][gram] / freqs_counts[n]
            if prob == 0:
                freqs[n][gram] = -1 * inf
            else:
                freqs[n][gram] = log(prob)

        for gram in transitions[n].keys():
            for next_gram in transitions[n][gram].keys():
                if trans_counts[n][gram] == 0:
                    trans_counts[n][gram] = 1
                prob = transitions[n][gram][next_gram] / trans_counts[n][gram]
                if prob == 0:
                    transitions[n][gram][next_gram] = -1 * inf
                else:
                    transitions[n][gram][next_gram] = log(prob)

    transitions[0] = {letter : inits[0].copy() for letter in inits[0].keys()}
    print('Done generating %s Markov models\n' % hmm_type)

    return [[inits[n], transitions[n], freqs[n]] for n in range(len(n_grams))]

def get_next_gram(stem, grams):
    """ Yields every n-gram that can follow the stem n-gram.

    Args:
        stem: n-gram string for which we want possible following n-grams.
        grams: all n-grams of the same length as the stem.

    Returns:
        next_grams: list of n-grams which may follow the stem. That is, an
            n-gram whose first n-1 letters match the last n-1 letters of the
            stem.
    """

    next_grams = []
    for gram in grams:
        if gram[:len(gram)-1] == stem[1:]:
            next_grams.append(gram)

    return next_grams

def print_to_file(guesses, correct, formatting, filename, viterbi=False):
    """ Prints the guesses to a .txt file for further examination.

    For each guess, the sentence is formatted as the original correct sentence
    was, then appended to a string in the form guess -- correct. These guesses
    are then printed to file.

    Args:
        guesses: list of tuple strings representing the decoding guesses.
        correct: the correct sentences.
        formatting: the dict that has how each correct sentence is formatted.
        filename: the name of the file to print to.
        viterbi: boolean for whether this is for the results of Viterbi or not.
    """

    intro_message = ('Each line represents the output guess of decoding using '
                     'the method indicated by the filename.\nEach line is of '
                     'the form:')

    form = '\n[correct]\n[message guess]\n[key guess]\n\n'
    if not viterbi:
        form = '\n[correct]\n[guess]\n\n'

    to_print = [intro_message, form]
    if not viterbi:
        for n in range(len(guesses)):
            format_to_follow = correct[n]
            message = list(guesses[n])
            for m in range(len(format_to_follow)):
                letter = format_to_follow[m]
                if not letter.isalpha():
                    message.insert(m, letter)

            message = ''.join(message)
            line = '%s\n%s\n\n' % (correct[n], message)
            to_print.append(line)

    else:
        for n in range(len(guesses)):
            format_to_follow = correct[n]
            message_guess = list(guesses[n][0])
            key_guess = list(guesses[n][1])
            for m in range(len(format_to_follow)):
                letter = format_to_follow[m]
                if not letter.isalpha():
                    message_guess.insert(m, letter)
                    key_guess.insert(m, letter)

            message_guess = ''.join(message_guess)
            key_guess = ''.join(key_guess)
            line = '%s\n%s\n%s\n\n' % (correct[n], message_guess,
                                       key_guess)
            to_print.append(line)

    filename += '.txt'
    file = open('cache/' + filename, 'w+')
    file.write(''.join(to_print))
    file.close()

def vectorize(sentence, n_grams, n):
    """ Transforms a sentence into a vector representing itself in n-gram form.

    Runs through the sentence and for each n-gram that appears, appends the
    index of that n-gram to the vector for the sentence, then returns that
    vector.

    Args:
        sentence: string to vectorize.
        n_grams: list of n-grams to create vectors out of.
        n: the length of each n-gram.

    Returns:
        vector: list which is the vector representation of the sentence.
    """

    vector = []
    for i in range(len(sentence)-n+1):
        sentence_gram = sentence[i:i+n]
        vector.append(n_grams.index(sentence_gram))

    return vector

def viterbi(sentence, markov_models):
    """ Runs through the Viterbi algorithm for the largest order Markov model.

    Initializes one or more pre-trellises which run Viterbi for the beginning of
    the sentence, and then the trellis for the Markov model in question is
    qcalculated using transition probabilities from the pre-trellises for
    initial probabilities. Then returns the most likely (message, key) pair,
    with the first being arbitrarily selected as the message.

    Args:
        sentence: the sentence on which Viterbi is run.
        markov_models: the Markov models which Viterbi uses for probabilities.

    Returns:
        message_key: tuple that contains the message and key. The message is the
            first and the key is the second. The real message may be the key,
            but the first is arbitrarily chosen as the message.
    """
    
    if len(markov_models) == 1:
        return viterbi_unigram(sentence, markov_models[0])

    alphabet = list('abcdefghijklmnopqrstuvwxyz')
    trellis = [{}]
    
    # Initializes the trellis for the start of the sentence with the
    # "pre-trellises", plus the start of the trellis for the n-grams in
    # question.
    for n in range(len(markov_models)):
        states = list(markov_models[n][0].keys())
        trans = markov_models[n][1]
        freqs = markov_models[n][2]

        if n == 0:
            init = markov_models[n][0]
            for state in states:
                key = alphabet[(alphabet.index(sentence[0]) \
                                - alphabet.index(state)) % 26]
                emit = freqs[state] + freqs[key]
                trellis[0].update({state : {'prob' : init[state] + emit,
                                            'prev' : None, 'key' : key}})

        else:
            trellis.append({})
            for state in states:
                key = ''
                for n in range(len(state)):
                    letter = alphabet[(alphabet.index(sentence[n]) \
                                      - alphabet.index(state[n])) % 26]
                    key = '%s%s' % (key, letter)

                emit = freqs[state] + freqs[key]
                prev_trans = markov_models[n-1][1]
                prev_states = list(prev_trans.keys())
                if n > 1:
                    prev_states = [gram for gram in prev_states if gram == state[:-1]]
                max_trans = prev_trans[prev_states[0]][state[1:]]
                previous = prev_states[0]
                for prev in prev_states:
                    if prev_trans[prev][state[1:]] > max_trans:
                        max_trans = prev_trans[prev][state[1:]]
                        previous = prev

                trellis[n].update({state : {'prob' : max_trans + emit,
                                            'prev' : previous, 'key' : key}})

    states = list(markov_models[-1][0].keys())
    trans = markov_models[-1][1]
    freqs = markov_models[-1][2]

    # Filling the rest of the trellis.
    num_grams = len(trellis) - 1
    for i in range(1, len(sentence)-len(markov_models)+1):
        trellis.append({})
        for state in states:
            key = ''
            for n in range(len(state)):
                letter = alphabet[(alphabet.index(sentence[i+n]) \
                                   - alphabet.index(state[n])) % 26]
                key = '%s%s' % (key, letter)

            emit = freqs[state] + freqs[key]
            prev_states = [gram for gram in states if gram[1:] == state[:-1]]
            best_transition = trellis[num_grams+i-1][prev_states[0]]['prob'] \
                              + trans[prev_states[0]][state]
            previous = prev_states[0]
            for prev_state in prev_states[1:]:
                trans_prob = trellis[num_grams+i-1][prev_state]['prob'] \
                             + trans[prev_state][state]
                if trans_prob > best_transition:
                    best_transition = trans_prob
                    previous = prev_state

            max_trans_prob = best_transition + emit
            trellis[num_grams+i].update({state : {'prob' : max_trans_prob,
                                        'prev' : previous, 'key' : key}})

        pct = int(round((100. * float(i)) / \
                        (len(sentence) - len(markov_models) + 1)))
        sys.stdout.write('\rPercent of sentence decoded: {}%'.format(pct))
        sys.stdout.flush()

    sys.stdout.write('\r                                     ')
    sys.stdout.flush()

    # Finds the best message and key pair from the highest probabilities.
    best_message = []
    best_key = []
    max_prob = max(state['prob'] for state in trellis[-1].values())
    prev_state = None

    #Finding the most probable final state.
    for state in trellis[-1].keys():
        if trellis[-1][state]['prob'] == max_prob:
            best_message.append(state[-1])
            best_key.append(trellis[-1][state]['key'][-1])
            prev_state = state
            break

    # Walking along previous states to build the best probability sentence.
    for i in range(len(trellis) - 2, -1, -1):
        best_message.insert(0, trellis[i+1][prev_state]['prev'][-1])
        prev_state = trellis[i+1][prev_state]['prev']
        best_key.insert(0, trellis[i][prev_state]['key'][-1])

    message_key = (''.join(best_message), ''.join(best_key))

    return message_key

def viterbi_unigram(sentence, markov_model):
    """ Runs through the Viterbi algorithm for a unigram Markov model.

    Args:
        sentence: string on which the algorithm is run.
        markov_model: unigram Markov model to run Viterbi with.

    Returns:
        message_key: tuple that contains the message and key. The message is the
            first and the key is the second. The real message may be the key,
            but the first is arbitrarily chosen as the message.
    """

    states = list(markov_model[0].keys())
    init = markov_model[0]
    trans = markov_model[1]
    freqs = markov_model[2]
    alphabet = list('abcdefghijklmnopqrstuvwxyz')
    trellis = [{}]

    # Initialize the Viterbi trellis.
    for state in states:
        key = alphabet[(alphabet.index(sentence[0]) \
                        - alphabet.index(state)) % 26]
        emit = freqs[state] + freqs[key]
        trellis[0].update({state : {'prob' : init[state] + emit,
                                           'prev' : None, 'key' : key}})

    # Fill the rest of the trellis based on the sentence.
    for i in range(1, len(sentence)):
        trellis.append({})
        for state in states:
            best_transition = trellis[i-1][states[0]]['prob'] \
                              + trans[states[0]][state]
            previous = states[0]
            for prev_state in states[1:]:
                trans_prob = trellis[i-1][prev_state]['prob'] \
                             + trans[prev_state][state]
                if trans_prob > best_transition:
                    best_transition = trans_prob
                    previous = prev_state

            key = alphabet[(alphabet.index(sentence[i]) \
                            - alphabet.index(state)) % 26]
            emit = freqs[state] + freqs[key]
            max_trans_prob = best_transition + emit
            trellis[i].update({state : {'prob' : max_trans_prob,
                                        'prev' : previous, 'key' : key}})

    # Finds the best message and key pair from the highest probabilities.
    best_message = []
    best_key = []
    max_prob = max(state['prob'] for state in trellis[-1].values())
    prev_state = None

    #Finding the most probable final state.
    for state in trellis[-1].keys():
        if trellis[-1][state]['prob'] == max_prob:
            best_message.append(state)
            best_key.append(trellis[-1][state]['key'])
            prev_state = state
            break

    # Walking along previous states to build the best probability sentence.
    for i in range(len(trellis) - 2, -1, -1):
        best_message.insert(0, trellis[i+1][prev_state]['prev'])
        prev_state = trellis[i+1][prev_state]['prev']
        best_key.insert(0, trellis[i][prev_state]['key'])

    message_key = (''.join(best_message), ''.join(best_key))

    return message_key

def preprocess(filename, length):
    """ Reads in sentences from a file and cleans them before converting into the
    plaintexts/ciphertexts to test and train on.

    It first reads in sentences and splits into plaintext and keys. It also creates a
    dict that keeps track of what each sentence initially looks like before it
    is stripped (sent to lowercase and removal of spaces and punctuation) for
    later formatting purposes.

    Args:
        filename: the path to the file that contains the sentences to be preprocessed.
        length: the minimum sentence length for each sentence

    Returns:
        (plaintext, ciphertext, orig_sentences_dict)
        The plaintext and ciphertext lists created from the preprocessed data and the
        dict mapping the scrubbed sentences to their raw form.
    """
    with open(filename,'r') as f:                                      
        sentences = f.readlines()

    orig_sentences = {}
    punctuated_sentences = [''] * len(sentences)
    whitespace_sentences = []
    for n in range(len(sentences)):
        try:
            stripped = ''.join(sentences[n].strip('.\n').split(' ')).lower()
        except:
            break
        while len(stripped) < length:
            try: stripped += ''.join(sentences[n+1].strip('.\n').split(' ')).lower()
            except: break
            del sentences[n+1]
        orig_sentences.update({stripped : sentences[n].strip('\n')})
        punctuated_sentences[n] = stripped

    punctuation_table = str.maketrans(dict.fromkeys(string.punctuation))
    whitespace_sentences = [s.translate(punctuation_table) for s in punctuated_sentences]
    sentences = [re.sub(r'\s+', '', s) for s in whitespace_sentences if s != '']
    sentences = [re.sub(r'\\xla', '', s) for s in whitespace_sentences]
    sentences = [''.join([i for i in s if i.isalpha()]) for s in sentences]
    sentences = [''.join([i for i in s if not i.isdigit()]) for s in sentences]
    sentences = [s for s in sentences if s != '']

    sentences = set(sentences)
    plaintext = [sentences.pop() for n in range(int(len(sentences) / 2))]
    running_key = list(sentences)

    alphabet = list('abcdefghijklmnopqrstuvwxyz')
    ciphertext = []

    # Generating ciphertext from sentences and running keys.
    for n in range(len(plaintext)):
        plain = plaintext[n]
        key = running_key[n]

        while len(plain) > len(key): key += key

        if len(plain) < len(key):
            key = key[0:len(plain)]

        crypt = []
        for x in range(len(plain)):
            plain_char = alphabet.index(plain[x])
            key_char = alphabet.index(key[x])
            crypt.append(alphabet[(plain_char + key_char) % 26])

        ciphertext.append(''.join(crypt))

    return (plaintext, ciphertext, orig_sentences)

def main():
    # ----------------------------------------------------------------------------
    # Parses commandline arguments that potentially specify the training/testing
    # corpus paths and the number of ngrams to evaluate on.
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to where training/testing corpus are", default="../docs/")
    parser.add_argument("--train", help="training text filename", default="brown0000.txt")
    parser.add_argument("--test", help="testing text filename", default="brown0000.txt")
    parser.add_argument("--ngrams", help="value of n for ngrams to use", default=2)
    parser.add_argument("--length", help="minimum plain/ciphertext length", default=100)
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Preprocess training/testing data specified
    same_corpora = False
    if args.train == args.test:
        plaintext, ciphertext, orig_sentences = preprocess(args.path + args.train, args.length)
        same_corpora = True
    else:
        plain_test, cipher_test, orig_sentences_test = preprocess(args.path + args.test, args.length)
        plain_train, cipher_train, orig_sentences_train = preprocess(args.path + args.train, args.length)

    # ---------------------------------------------------------------------------

    # Splitting into training and testing sets. Default partition is 80/20 if same.

    if same_corpora:
        train_len = int((len(plaintext) * 4) / 5)
        plain_train = plaintext[:train_len]
        plain_test = plaintext[train_len:]
        cipher_train = ciphertext[:train_len]
        cipher_test = ciphertext[train_len:]

    orig_test = {}
    # Generating all n-grams up to the specified length.
    alphabet = list('abcdefghijklmnopqrstuvwxyz')
    n_grams = generate_ngrams(alphabet, args.ngrams)

    # Creating baseline for comparison, where each letter is just guessed.
    baseline_acc = 0.
    baseline_guesses = []
    for n in range(len(cipher_test)):
        sentence = cipher_test[n]
        baseline_guess = ''
        for letter in sentence:
            letter_guess = random.randint(0, 25)
            baseline_guess = ''.join([baseline_guess, alphabet[letter_guess]])
        baseline_guesses.append(baseline_guess)

        baseline_acc += accuracy(plain_test[n], baseline_guess)

    filename = 'baseline'
    print_to_file(baseline_guesses, plain_test, orig_test, filename)
    
    baseline_acc /= len(cipher_test)
    print('Accuracy of baseline: %.5f' % baseline_acc)

    # Turning each sentence into vectors of n-grams.
    plain_train_vec = [[] for n in range(len(n_grams))]
    cipher_train_vec = [[] for n in range(len(n_grams))]
    plain_test_vec = [[] for n in range(len(n_grams))]
    cipher_test_vec = [[] for n in range(len(n_grams))]

    print('\nGetting n-gram vectors from sentences')
    for n in range(len(n_grams)):
        for m in range(len(plain_train)):
            plain_train_vec[n].extend(vectorize(plain_train[m],
                                                n_grams[n], n+1))
            cipher_train_vec[n].extend(vectorize(cipher_train[m],
                                                 n_grams[n], n+1))

        for m in range(len(plain_test)):
            plain_test_vec[n].extend(vectorize(plain_test[m],
                                               n_grams[n], n+1))
            cipher_test_vec[n].extend(vectorize(cipher_test[m],
                                                n_grams[n], n+1))

    # Testing multinomial Naive Bayes.
    print('\nTesting multinomial Naive Bayes')
    naive_bayes = MultinomialNB()
    for n in range(len(plain_train_vec)):
        cipher = np.array(cipher_train_vec[n]).reshape(-1, 1)
        plain = np.array(plain_train_vec[n]).reshape(-1, 1).ravel()
        naive_bayes.fit(cipher, plain)

        test = np.array(cipher_test_vec[n]).reshape(-1, 1)
        predict = naive_bayes.predict(test).tolist()
        predictions = []

        nb_accuracy = 0.
        for sentence in plain_test:
            sentence_length = len(sentence)
            prediction = predict[:sentence_length-n]

            str_predict = devectorize(prediction, n_grams[n])
            predictions.append(str_predict)
            nb_accuracy += accuracy(sentence, str_predict)

        nb_accuracy /= len(plain_test)
        print('Accuracy of %d-gram Naive Bayes: %.5f' % ((n+1), nb_accuracy))

        filename = 'naive_bayes_%d-grams' % (n+1)
        #print_to_file(predictions, plain_test, orig_test, filename)

    # Testing logistic regression.
    print('\nTesting logistic regression')
    log_reg = LogisticRegression()
    for n in range(len(plain_train_vec)):
        cipher = np.array(cipher_train_vec[n]).reshape(-1, 1)
        plain = np.array(plain_train_vec[n]).reshape(-1, 1).ravel()
        log_reg.fit(cipher, plain)

        test = np.array(cipher_test_vec[n]).reshape(-1, 1)
        predict = log_reg.predict(test).tolist()
        predictions = []

        lr_accuracy = 0.
        for sentence in plain_test:
            sentence_length = len(sentence)
            prediction = predict[:sentence_length-n]

            str_predict = devectorize(prediction, n_grams[n])
            predictions.append(str_predict)
            lr_accuracy += accuracy(sentence, str_predict)

        lr_accuracy /= len(plain_test)
        print('Accuracy of %d-gram logistic regression: %.5f' % ((n+1), lr_accuracy))

        filename = 'logistic_regression_%d-grams' % (n+1)
        #print_to_file(predictions, plain_test, orig_test, filename)

    # Testing support vector machine.
    print('\nTesting support vector machine')
    svm = LinearSVC()
    for n in range(len(plain_train_vec)):
        cipher = np.array(cipher_train_vec[n]).reshape(-1, 1)
        plain = np.array(plain_train_vec[n]).reshape(-1, 1).ravel()
        svm.fit(cipher, plain)

        test = np.array(cipher_test_vec[n]).reshape(-1, 1)
        predict = svm.predict(test).tolist()
        predictions = []

        svm_accuracy = 0.
        for sentence in plain_test:
            sentence_length = len(sentence)
            prediction = predict[:sentence_length-n]

            str_predict = devectorize(prediction, n_grams[n])
            predictions.append(str_predict)
            svm_accuracy += accuracy(sentence, str_predict)

        svm_accuracy /= len(plain_test)
        print('Accuracy of %d-gram support vector machine: %.5f' % ((n+1), svm_accuracy))

        filename = 'svm_%d-grams' % (n+1)
        #print_to_file(predictions, plain_test, orig_test, filename)

    print()

    # Creating the hidden Markov models.
    hmms_smoothed = get_hmms(plain_train, n_grams)
    hmms_unsmoothed = get_hmms(plain_train, n_grams, False)

    # Testing the hidden Markov models with Viterbi.
    for n in range(len(hmms_smoothed)):
        print('Running Viterbi on %d-gram Markov models' % (n+1))
        smoothed_acc = 0.
        smoothed_guesses = []
        unsmoothed_acc = 0.
        unsmoothed_guesses = []
        for m in range(len(cipher_test)):
            sentence = cipher_test[m]
            smoothed_guess = viterbi(sentence, hmms_smoothed[:n+1])
            smoothed_guesses.append(smoothed_guess)
            smoothed_acc += accuracy(plain_test[m], smoothed_guess[0])

            unsmoothed_guess = viterbi(sentence, hmms_unsmoothed[:m+1])
            unsmoothed_guesses.append(unsmoothed_guess)
            unsmoothed_acc += accuracy(plain_test[m], unsmoothed_guess[0])

        sys.stdout.write('\r')
        sys.stdout.flush()

        filename = 'smoothed_HMM_%d-grams' % (n+1)
        print_to_file(smoothed_guesses, plain_test, orig_test, filename, True)
        
        smoothed_acc /= len(cipher_test)
        print('Accuracy of smoothed %d-gram HMM: %.5f' % ((n+1), smoothed_acc))

        filename = 'unsmoothed_HMM_%d-grams' % (n+1)
        print_to_file(unsmoothed_guesses, plain_test, orig_test, filename, True)
        
        unsmoothed_acc /= len(cipher_test)
        print('Accuracy of unsmoothed %d-gram HMM: %.5f' % ((n+1), unsmoothed_acc))

        print()

if __name__ == '__main__':
    main()
