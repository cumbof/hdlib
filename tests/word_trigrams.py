
import nltk
import nltk.corpus

from hdlib.space import Space, Vector
from hdlib.arithmetic import bind, bundle, permute


# I am trying to encode sequences of three words "quick brown fox"
# I am desiring that the output of "quick brown" be "fox"

# I am permuting the first word twice, the second word once and leaving the third word as is
# and then binding those together to encode the sequences of three words:
#
# ppWord1 * pWord2 * Word3
#
# bind( 
#     permute(space.get(trigram[0]), 2),  #ppWord1
#     bind( 
#             permute(space.get(trigram[1]), 1),  #pWord2
#             space.get(trigram[2]) #Word3
#         ) 
# ) 

# To get the output I am binding the permutes of the first two words in a sequence with the space:
#
# ppWord1 * pWord2 * Space = Word3
# 
# GUESS_PES = bind( 
#                 permute(space.get(trailingword1), 2), 
#                 bind( 
#                         permute(space.get(trailingword2)), 
#                         bundling
#                     ) 
#             )

# It works for a small set of three word sequences, but when using a large corpus it fails
#
# Can you help me understand if this is the best way to go about encoding three word sequences
# with the goal of inputting words 1 and 2, and getting word 3 as output
# 
# one, why does this work at small scale, is it just a fluke? Or am I encoding sequences correctly
# 
# two, why does it fail when scaling up, I have the dimensionality set to 10,000 dimensions, 
# if you have any idea why this idea of word trigrams might fail at scale



# Split sentences into trigrams using sliding window
def split_on_window(sequence="I love food and I like drink", limit=4):
    results = []
    split_sequence = sequence.split()
    iteration_length = len(split_sequence) - (limit - 1)
    max_window_indicies = range(iteration_length)
    for index in max_window_indicies:
        results.append(split_sequence[index:index + limit])
    return results



# using sentences from nltk.corpus
sentences = nltk.corpus.brown.sents()
seen = {}
trigrams = []

# using an list of test sentences
sentencestest = [["the", "quick", "brown", "fox", "jumps", "over"],["mary", "had", "a", "little", "lamb"],["all", "quiet", "on", "the", "western", "front"],["the", "small", "fox", "ate", "beans"]]



# the problem lies here, this works on a small set of sentences in sentencetest
# ie: you get out "fox" when you type the two words "quick brown"

# when generalizing to the browns sentences, it no longer works
# uncomment the line below to try out on browns sentences

# for sentence in sentences: # uncomment this for browns sentences
for sentence in sentencestest: # uncomment this for sentencetest

    # only include word that starts with alphanumeric
    words = [word.lower() for word in sentence if word[0].isalpha()]

    # split words into threes with a sliding window
    trigramset = split_on_window(" ".join(words), 3)

    for tx in trigramset:
        trigrams.append(tx)

    # keep track of words seen
    for tx in trigramset:
        for word in tx:
            try:
                seen[word].append('seen')
            except KeyError as _:
                seen[word] = []
                seen[word].append('seen')

print(trigrams)
# print(seen)

# Initialize vectors space
space = Space(10000)

# Define features and country information
# names = [
#   "USA", "WASHINGTON DC", "DOLLAR", # United States of America
#   "MEX", "MEXICO CITY", "PESO"  # Mexico
# ]

# Build a random bipolar vector for each word
# Add vectors to the space
space.bulkInsert(list(seen.keys()))

print('Binding')
bindings = []
for trigram in trigrams:
    # print(trigram)
    bindings.append( bind( 
                            permute(space.get(trigram[0]), 2), 
                            bind( 
                                    permute(space.get(trigram[1]), 1), 
                                    space.get(trigram[2]) 
                                ) 
                        ) 
                    ) # Bind NAM with USA



# Encode USA information in a single vector
# USTATES = [(NAM * USA) + (CAP * WDC) + (MON * DOL)]
# USTATES_NAM = bind(space.get("MEX"), space.get("USA")) # Bind NAM with USA
# USTATES_CAP = bind(space.get("MEXICO CITY"), space.get("WASHINGTON DC")) # Bind CAP with WDC
# USTATES_MON = bind(space.get("PESO"), space.get("DOLLAR")) # Bind MON with DOL
# USTATES = bundle(bundle(USTATES_NAM, USTATES_CAP), USTATES_MON) # Bundle USTATES_NAM, USTATES_CAP, and USTATES_MON
print('Bundling')

first = True
for binding in bindings:
  if first:
    bundling = binding
    first = False
  else:
    bundling = bundle(binding, bundling)
    

# F_UM = USTATES * MEXICO
#      = [(USA * MEX) + (WDC * MXC) + (DOL * PES) + noise]
#F_UM = bind(USTATES, MEXICO)

# DOL * F_UM = DOL * [(USA * MEX) + (WDC * MXC) + (DOL * PES) + noise]
#            = [(DOL * USA * MEX) + (DOL * WDC * MXC) + (DOL * DOL * PES) + (DOL * noise)]
#            = [noise1 + noise2 + PES + noise3]
#            = [PES + noise4]
#            ≈ PES

while(True):

    query = input("Query string (two words minimum): ")
    querywords = query.split()
    trailingword1 = querywords[-2]
    trailingword2 = querywords[-1]

    print(trailingword1)
    print(trailingword2)

    try:
        GUESS_PES = bind( 
                            permute(space.get(trailingword1), 2), 
                            bind( 
                                    permute(space.get(trailingword2)), 
                                    bundling
                                ) 
                        )
        print(space.find(GUESS_PES))
    except TypeError as _:
        print('symbol not seen, please type atleast two words')
