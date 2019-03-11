import nltk
from nltk.corpus import words, brown
import enchant

word_dictionary = enchant.Dict("en_US")
word_dictionary.add('2012')
word_dictionary.add('2008')
word_dictionary.add('2016')
word_dictionary.add('obama')
word_dictionary.add('romney')
word_dictionary.add('potus')
word_dictionary.add('america')



"""
word_dictionary = list(set(words.words()))
word_dictionary.append(unicode('2012'))
word_dictionary.append(unicode('2008'))
word_dictionary.append(unicode('2016'))
word_dictionary.append(unicode('obama'))
word_dictionary.append(unicode('romney'))
word_dictionary.append(unicode('potus'))
word_dictionary.append(unicode('america'))
"""



def split_hashtag_to_words_all_possibilities(hashtag):
	all_possibilities = []
	split_posibility = []
	for i in reversed(range(len(hashtag)+1)):
		if len(hashtag[:i]) > 0 and word_dictionary.check(hashtag[:i]) == True:
			split_posibility.append(True)
		else:
			split_posibility.append(False)
	
	possible_split_positions = [i for i, x in enumerate(split_posibility) if x == True]
	
	for split_pos in possible_split_positions:
		split_words = []
		word_1, word_2 = hashtag[:len(hashtag)-split_pos], hashtag[len(hashtag)-split_pos:]
		
		if len(word_2) > 0 and word_dictionary.check(word_2) == True:
			split_words.append(word_1)
			split_words.append(word_2)
			all_possibilities.append(split_words)

			another_round = split_hashtag_to_words_all_possibilities(word_2)
				
			if len(another_round) > 0:
				all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in zip([word_1]*len(another_round), another_round)]
		else:
			another_round = split_hashtag_to_words_all_possibilities(word_2)
			
			if len(another_round) > 0:
				all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in zip([word_1]*len(another_round), another_round)]
	return all_possibilities



def split_hashtag(x):
    #if the entire hashtag is a an acutal word just return it
    if word_dictionary.check(x) == True:
        return x
    a = split_hashtag_to_words_all_possibilities(x) # all possibilities
    if not a: #if it can't parse just return it
        return unicode('') #return unicode('#+'+x)
    argC = {} #otherwise return the longest set of n parsed words
    for i in range(0,len(a)):
        argC[len(a[i])] = i
    
    return unicode(" ".join(a[argC[min(argC.keys())]]))

"""   
import re
tw = unicode('This is a tweet #ihaveyourback #420')
hashtags = re.findall(r"#(\w+)", tw)
for mwords in hashtags:
    tw = re.sub('#'+ mwords,split_hashtag(mwords),tw)
print tw
"""
