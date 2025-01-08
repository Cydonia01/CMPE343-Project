import numpy as np
import math

words = np.array([]) # numpy array to store all words
next_word_count = {} # dictionary to store next word count

file = open("sentences.txt") # open file

# read file line by line
for line in file:
    sentence = line.strip().split()
    words = np.append(words, sentence)

    ## counte the number of next words for each word
    for i in range(len(sentence) - 1):
        if next_word_count.get(sentence[i]) is None:
            next_word_count[sentence[i]] = {}
        next_word_count[sentence[i]][sentence[i+1]] = next_word_count[sentence[i]].get(sentence[i+1], 0) + 1

file.close() # close file

# calculate probability of next word (normalizing to make the sum of probabilities 1)
for key in next_word_count:
    total = sum(next_word_count[key].values())
    for next_word in next_word_count[key]:
        next_word_count[key][next_word] /= total

def generate_sentence():
    prob = 1
    current_word = "<|start|>"
    generated_sentence = [] # list to store generated sentence
    while current_word != "<|end|>":
        generated_sentence.append(current_word)

        # take the next words and their probabilites
        next_word_candidates = list(next_word_count[current_word].keys())
        next_word_probabilities = list(next_word_count[current_word].values())

        # choose the next word based on the probabilities
        current_word = np.random.choice(next_word_candidates, p=next_word_probabilities)
        prob *= next_word_count[generated_sentence[-1]][current_word]
        
    
    prob *= next_word_count[generated_sentence[-1]]["<|end|>"]
    generated_sentence.append("<|end|>")
    # print("Probability of the generated sentence: ", math.log10(prob))
    
    return " ".join(generated_sentence)
# generate 5 sentences
for i in range(5):
    print(generate_sentence())