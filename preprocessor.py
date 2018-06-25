# libraries for dataset preparation, feature engineering, model training 
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas, xgboost, numpy, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

def split_dataset(features,labels):

    # split the dataset into training and validation datasets 
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(features, labels)

    # label encode the target variable 
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)
    
    return train_x, valid_x, train_y, valid_y

def get_count_vec_features(features,train_x,valid_x):
    
    # create a count vectorizer object 
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(features)

    # transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(train_x)
    xvalid_count =  count_vect.transform(valid_x)
    
    return xtrain_count,xvalid_count

def get_tf_idf_features(features,train_x,valid_x):
    
    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(features)
    xtrain_tfidf =  tfidf_vect.transform(train_x)
    xvalid_tfidf =  tfidf_vect.transform(valid_x)
    
    # ngram level tf-idf 
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram.fit(features)
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

    # characters level tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram_chars.fit(features)
    xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
    xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)
    
    return xtrain_tfidf,xvalid_tfidf,xtrain_tfidf_ngram,xvalid_tfidf_ngram,xtrain_tfidf_ngram_chars,xvalid_tfidf_ngram_chars

def get_word_embedding_features(train_x,valid_x):
    
    # load the pre-trained word-embedding vectors 
    embeddings_index = {}
    for i, line in enumerate(open('data/wiki-news-300d-1M.vec')):
        values = line.split()
        embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

    # create a tokenizer 
    token = text.Tokenizer()
    token.fit_on_texts(trainDF['text'])
    word_index = token.word_index

    # convert text to sequence of tokens and pad them to ensure equal length vectors 
    train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
    valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

    # create token-embedding mapping
    embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

def get_lda_features(features,xtrain_count):
    
     # create a count vectorizer object 
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(features)
    
    # train a LDA Model
    lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)
    X_topics = lda_model.fit_transform(xtrain_count)
    topic_word = lda_model.components_ 
    vocab = count_vect.get_feature_names()

    # view the topic models
    n_top_words = 10
    topic_summaries = []
    for i, topic_dist in enumerate(topic_word):
        topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n_top_words+1):-1]
        topic_summaries.append(' '.join(topic_words))
    
    return topic_summaries