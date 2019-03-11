#computes accuracy of sentiment analysis using a recurrent neural network (GRU) 


import sys,os,string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15, 5)
pd.set_option('display.max_colwidth', 300)

def get_data(data,mode = 'train'):
    """ get and clean the data """
    data = data.iloc[1:]
    data['text'] = data['text'].values.astype('unicode')
    # remove rows with mixed sentiment
    data = data[data['sentiment'] < 2]
    data.index = range(len(data))
    
    return data
import re

emoticon_dictionary = {':)':' smileyface ','(:':' smileyface ','XD': ' happyface ',':D': ' smileyface ','>.<':' smileyface ',':-)':' smileyface ',';)':' winkface ',';D':' winkface ',':\'(':' cryingface '}

emoticons = [':\)','\(:','XD',':D','>\.<',':-\)',';\)',';D',':\'\(']

emoticon_pattern = re.compile(r'(' + '\s*|\s*'.join(emoticons) + r')')

# convert emoticons to words
def emoticon_converter(x):
    x = emoticon_pattern.sub(lambda i : emoticon_dictionary[i.group().replace(' ','')],x)   
    return x
    
from hashTagSplit import *

def separate_hashtag(x):
    x = x.split()
    temp = []
    for i,word in enumerate(x):
        if '#' in word:
            if any(w.isupper() for w in word):
                temp += re.findall('[A-Z][^A-Z]*',word)
            else:
                if len(word) > 1:
                    temp += [split_hashtag(word[1:])]
        else:
            temp.append(word)
    
    return ' '.join(temp)
    
# remove punctuations
punc = ['\:','\;','\?','\$','\.','\(','\)','\=','\%','\-','\>','\<','\,','\"','\\','\&','\+']
cond_1 = re.compile('|'.join(punc))
# remove tags
tags = ['<a>','</a>','<e>','</e>']
cond_2 = re.compile("|".join(tags))

def preprocess(data):
    """ preprocess the data"""
     # remove users
    data = data.apply(lambda x : re.sub(r'\@\s?\w+','',x))
    # remove hypertext 
    data = data.apply(lambda x : re.sub(r'http://\S+','',x))
    # remove tags
    data = data.apply(lambda x : re.sub(cond_2,'',x))
    # remove punctuations
    data = data.apply(lambda x : re.sub(cond_1,'',x))
    # remove digits
    data = data.apply(lambda x : re.sub(r'[0-9]+','',x))
    # convert to ascii
    data = data.apply(lambda x: x.encode('utf-8'))
    printable = set(string.printable)
    for i in range(len(data)):
        data[i] = filter(lambda x: x in printable, data[i])
    
    return data
    
import nltk

from nltk.corpus import stopwords

manual_stopwords_list = ['RT','MT'] #retweet, modified tweet
stopwords_list = stopwords.words('english') + manual_stopwords_list


# stopwords list based on pos tags
remove_tags_nltkpos = ['IN','DT','PRP','CC']

#IN = Preposition or subordinating conjunction (on,after,etc.)
#DT = Determiner, (the,a,every)
#PRP = Persoanl Pronoun (I,he,she)
#CC = Coordinating conjunction (and,but,or)


def pos_tag_filter(x):
    x = x.split()
    s = nltk.pos_tag(x)
    for i,(_,tag) in enumerate(s):
        if tag in remove_tags_nltkpos:
            x[i] = ''
    return ' '.join(x)
    
# stemming
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer

class WordTokenizer(object):
    def __init__(self,stemmer='porter'):
        self.stemmer = stemmer
        if stemmer == 'wordnet':
            self.wnl = WordNetLemmatizer()
        if stemmer == 'porter':
            self.wnl = PorterStemmer()
        if stemmer == 'snowball':
            self.wnl = SnowballStemmer('english')
    def __call__(self,doc):
        if self.stemmer == 'wordnet':
            return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        else:
            return [self.wnl.stem(t) for t in word_tokenize(doc)]
                    
GLOVE_FILE = 'glove.twitter.27B.200d.txt'
EMBEDDING_DIM = 200 #size of word vector 

embeddings_index = {}
f = open(GLOVE_FILE)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC,libsvm,SVC
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import TruncatedSVD


from sklearn.metrics import classification_report,accuracy_score,f1_score,precision_score,recall_score
from sklearn.cross_validation import cross_val_score,cross_val_predict,KFold,StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support as score

def get_X_y(data):
    return data['text'],data['sentiment'].astype(int)
    

# create a pipeline

def model_pipeline(X,WordTokenizer,text_vector = None, svd_transform = None,mode = 'train'):

    if mode == 'train':
        text_vector = Pipeline([('vect', CountVectorizer(tokenizer = WordTokenizer('wordnet'),stop_words = [],ngram_range = (1,2),max_features=10000)),
                    ('tfidf',TfidfTransformer())])
        svd_transform = TruncatedSVD(n_components = 1000,n_iter = 5)
        # transform the data
        X = text_vector.fit_transform(X)
        X_reduced = svd_transform.fit_transform(X)
        return X,X_reduced,text_vector,svd_transform
    else:
        X = text_vector.transform(X)
        X_reduced = svd_transform.transform(X)
        return X,X_reduced  
        


def classifier_predict(clf,X):
    return clf.predict_proba(X)  
    
import keras
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.preprocessing.text import Tokenizer
from keras import optimizers
from keras import regularizers

MAX_SEQUENCE_LENGTH = 30 #max number of sentences in a message
MAX_NB_WORDS = 20000 #cap vocabulary
TOKENIZER = 'keras' #or use nltk
STEMMER = 'wordnet'

def get_Ytrue_Ypred(model,x,y):
    #Y matrix is [1,0,0] for class 0, [0,1,0] for class 1, [0,0,1] for class -1
    convert_to_label ={0:0,1:1,2:-1}
    model_predictions = model.predict(x)
    y_pred = np.zeros(len(y))
    y_true = np.zeros(len(y))

    for i in range(len(y)):
        y_pred[i] = convert_to_label[np.argmax(model_predictions[i])]
        y_true[i] = convert_to_label[np.argmax(y[i])]

    return y_true,y_pred
    
class weighted_categorical_crossentropy(object):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        loss = weighted_categorical_crossentropy(weights).loss
        model.compile(loss=loss,optimizer='adam')
    """
    
    def __init__(self,weights):
        self.weights = K.variable(weights)
        
    def loss(self,y_true, y_pred):
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= y_pred.sum(axis=-1, keepdims=True)
        # clip
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        # calc
        loss = y_true*K.log(y_pred)*self.weights
        loss =-K.sum(loss,-1)
        return loss
        
def kerasprocess_data(texts,labels = None,tokenizer = None,mode = 'train'):
    if mode == 'train':
        tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(texts)
        word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(texts) #list of lists, basically replaces each word with number

    #pad the data 
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    if mode == 'train':
        #prepare embedding matrix
        num_words = len(word_index)+1
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
    
        return data,labels,embedding_matrix,tokenizer

    return data
    
def GRU_train(data,labels,embedding_matrix,data_name='Obama'):
    labels = keras.utils.np_utils.to_categorical(labels,nb_classes=3)
    
    if data_name == 'Obama':
        clf = obama_build_model(embedding_matrix,3)
    else:
        clf = romney_build_model(embedding_matrix,3)
    clf.fit(data,labels, nb_epoch=50, batch_size=64,verbose=0)
    return clf
    

def GRU_predict(clf,data):
    predict_probs = clf.predict(data)
    # keras predicts probabilites on 0,1,-1 should be -1,0,1
    predict_probs[:,[0,1,2]] = predict_probs[:,[2,0,1]]
    return predict_probs

def obama_build_model(embedding_matrix,labels_len):
    np.random.seed(1)
    num_words = embedding_matrix.shape[0]
    l2 = regularizers.l2(0.01)
    l22 = regularizers.l2(0.01)
    model = Sequential()
    embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=0)
    model.add(embedding_layer)
    model.add(GRU(10,return_sequences=False,dropout_W=0.6,dropout_U=0.5))
    weights = np.array([1,2,1]) #index 0 for class 0, index 1 for class 1, index 2 for class -1
    mloss = weighted_categorical_crossentropy(weights).loss
    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.8, nesterov=True)
    model.add(Dense(labels_len, activation='softmax'))
    model.compile(loss=mloss, optimizer='rmsprop')
    
    return model
    

def obama_fullcommonpipeline(filename,mode = 'train'):

    obama_data = pd.read_excel(filename,names = ['date','time','text','sentiment'],parse_cols = 4,sheetname = 'Obama')

    obama_data = get_data(obama_data,mode)
    obama_data['text'] = obama_data['text'].apply(emoticon_converter)
    obama_data['text'] = obama_data['text'].apply(separate_hashtag)
    obama_data['text'] = preprocess(obama_data['text'])
    obama_data['text'] = obama_data['text'].apply(pos_tag_filter)
    obama_data['text'] = obama_data['text'].apply(lambda x : x.lower())
    return obama_data


def obama_fulltrainpipeline(trainfilename):
    print "obama training/test"
    np.random.seed(1)
    obama_data = obama_fullcommonpipeline(trainfilename)
    X,y = get_X_y(obama_data)
    X,X_reduced,text_vector,svd_transform = model_pipeline(X,WordTokenizer)
    texts = obama_data['text']
    labels = np.array(obama_data['sentiment'])
    data,labels,embedding_matrix,tokenizer = kerasprocess_data(texts,labels)
    
    avg_acc = []
    avg_f1 = []
    f_pos = []
    f_neg = []
    precision_pos = []
    precision_neg = []
    recall_pos = []
    recall_neg = []
  
    kf = KFold(n=len(data),n_folds=10)
    for train,test in kf:
        np.random.seed(1)
        x_train, x_val, y_train, y_val = X[train], X[test], y[train], y[test]
        x_trainr, x_valr = X_reduced[train], X_reduced[test]
        x_traing, x_valg, y_traing, y_valg = data[train], data[test], labels[train], labels[test]

        gru_clf = GRU_train(x_traing,y_traing,embedding_matrix)
        
        gru_pred = GRU_predict(gru_clf,x_valg)
        y_pred = np.argmax(gru_pred,axis = 1) - 1
        y_true = y_val
        
        avg_acc.append(accuracy_score(y_true,y_pred))
        avg_f1.append(f1_score(y_true,y_pred,average='macro'))      
        print classification_report(y_true,y_pred)
        precision, recall, fscore, support = score(y_true, y_pred)
        f_pos.append(fscore[2])
        f_neg.append(fscore[0])
        precision_pos.append(fscore[2])
        precision_neg.append(fscore[0])
        recall_pos.append(fscore[2])
        recall_neg.append(fscore[0])
    

    #print classification_report(y_true,y_pred)
    print 'Average f1-score = ', np.mean(np.array(avg_f1))
    print 'Overall Accuracy = ',100.0*np.mean(np.array(avg_acc)),'%'
    print 'positive f1-score = ', np.mean(np.array(f_pos))
    print 'negative f1-score = ', np.mean(np.array(f_neg))
    print 'positive precision = ', np.mean(np.array(precision_pos))
    print 'negative precision = ', np.mean(np.array(precision_neg))
    print 'positive recall = ', np.mean(np.array(recall_pos))
    print 'negative recall = ', np.mean(np.array(recall_neg))
    
    
    

    
trainfilename = 'training-Obama-Romney-tweets.xlsx'

obama_fulltrainpipeline(trainfilename)
