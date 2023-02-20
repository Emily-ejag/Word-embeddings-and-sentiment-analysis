# IA4 Group 37: David Smerkous, Emily Arteaga, Anita Ruangrotsakun
#!pip3 install nltk  # ensure you have nltk
import numpy as np
import os
import re
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import nltk
import re
from functools import partial
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Loads GloVe embeddings from a designated file location. 
#
# Invoked via:
# ge = GloVe_Embedder(path_to_embeddings)
#
# Embed single word via:
# embed = ge.embed_str(word)
#
# Embed a list of words via:
# embeds = ge.embed_list(word_list)
#
# Find nearest neighbors via:
# ge.find_k_nearest(word, k)
#
# Save vocabulary to file via:
# ge.save_to_file(path_to_file)

class GloVe_Embedder:
    def __init__(self, path):
        self.embedding_dict = {}
        self.embedding_array = []
        self.unk_emb = 0

        if not os.path.isfile(path):
            print("Error: could not find file", path)

        # Adapted from https://stackoverflow.com/questions/37793118/load-pretrained-GloVe-vectors-in-python
        with open(path,'r') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                self.embedding_dict[word] = embedding
                self.embedding_array.append(embedding.tolist())
        self.embedding_array = np.array(self.embedding_array)
        self.embedding_dim = len(self.embedding_array[0])
        self.vocab_size = len(self.embedding_array)
        self.unk_emb = np.zeros(self.embedding_dim)

    # Check if the provided embedding is the unknown embedding.
    def is_unk_embed(self, embed):
        return np.sum((embed - self.unk_emb) ** 2) < 1e-7
    
    # Check if the provided string is in the vocabulary.
    def token_in_vocab(self, x):
        if x in self.embedding_dict and not self.is_unk_embed(self.embedding_dict[x]):
            return True
        return False

    # Returns the embedding for a single string and prints a warning if
    # the string is unknown to the vocabulary.
    # 
    # If indicate_unk is set to True, the return type will be a tuple of 
    # (numpy array, bool) with the bool indicating whether the returned 
    # embedding is the unknown embedding.
    #
    # If warn_unk is set to False, the method will no longer print warnings
    # when used on unknown strings.
    def embed_str(self, x, indicate_unk = False, warn_unk = True):
        if self.token_in_vocab(x):
            if indicate_unk:
                return (self.embedding_dict[x], False)
            else:
                return self.embedding_dict[x]
        else:
            if warn_unk:
                    print("Warning: provided word is not part of the vocabulary!")
            if indicate_unk:
                return (self.unk_emb, True)
            else:
                return self.unk_emb

    # Returns an array containing the embeddings of each vocabulary token in the provided list.
    #
    # If include_unk is set to False, the returned list will not include any unknown embeddings.
    def embed_list(self, x, include_unk = True):
        if include_unk:
            embeds = [self.embed_str(word, warn_unk = False).tolist() for word in x]
        else:
            embeds_with_unk = [self.embed_str(word, indicate_unk=True, warn_unk = False) for word in x]
            embeds = [e[0].tolist() for e in embeds_with_unk if not e[1]]
            if len(embeds) == 0:
                print("No known words in input:" + str(x))
                embeds = [self.unk_emb.tolist()]
        return np.array(embeds)
    
    # Finds the vocab words associated with the k nearest embeddings of the provided word. 
    # Can also accept an embedding vector in place of a string word.
    # Return type is a nested list where each entry is a word in the vocab followed by its 
    # distance from whatever word was provided as an argument.
    def find_k_nearest(self, word, k, warn_about_unks = True):
        if type(word) == str:
            word_embedding, is_unk = self.embed_str(word, indicate_unk = True)
        else:
            word_embedding = word
            is_unk = False
        if is_unk and warn_about_unks:
            print("Warning: provided word is not part of the vocabulary!")

        all_distances = np.sum((self.embedding_array - word_embedding) ** 2, axis = 1) ** 0.5
        distance_vocab_index = [[w, round(d, 5)] for w,d,i in zip(self.embedding_dict.keys(), all_distances, range(len(all_distances)))]
        distance_vocab_index = sorted(distance_vocab_index, key = lambda x: x[1], reverse = False)
        return distance_vocab_index[:k]

    def save_to_file(self, path):
        with open(path, 'w') as f:
            for k in self.embedding_dict.keys():
                embedding_str = " ".join([str(round(s, 5)) for s in self.embedding_dict[k].tolist()])
                string = k + " " + embedding_str
                f.write(string + "\n")
######## End of GloVe embedding skeleton  #######
#
#
#
#
#
#
#
################## Start of Part 1 ###############            
def part1():
    ge = GloVe_Embedder("GloVe_Embedder_data.txt")
    data = []

    # Part 1a: build dataset
    # Returned list includes the seed word as well, so collecting top 30 most similar words
    class_labels = []
    true_labels = []
    seeds = ['flight', 'good', 'terrible', 'help', 'late']
    for word in seeds:
        result = ge.find_k_nearest(word, 30)
        data += result
        for _ in range(30):
            class_labels.append(word)
            true_labels.append(seeds.index(word))

    word_list = [word[0] for word in data]
    print('Word List: (', len(word_list), ') words', word_list)
    

    # Part 1b: try PCA
    embeds = ge.embed_list(word_list)
    pca = PCA(n_components=2)
    pca_reduction = pca.fit_transform(embeds)

    df = pd.DataFrame(data = pca_reduction, columns = ['pca1', 'pca2'])
    plot_df = pd.concat([df, pd.DataFrame(class_labels, columns=['class'])], axis = 1)
    print(plot_df.head())

    colors = ["navy", "turquoise", "darkorange", "green", "pink"]
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    for target, color in zip(seeds, colors):
        indicesToKeep = plot_df['class'] == target
        ax.scatter(plot_df.loc[indicesToKeep, 'pca1']
                , plot_df.loc[indicesToKeep, 'pca2']
                , c = color
                , s = 50)
    ax.legend(seeds)
    ax.grid()
    plt.savefig("pca_scatterplot.png")
    

    # Part 1c: t-SNE
    for perplexity in [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100]:
        tsne_reduction = TSNE(n_components=2, init='random', perplexity=perplexity).fit_transform(embeds)
        df = pd.DataFrame(data = tsne_reduction, columns = ['tsne1', 'tsne2'])
        plot_df = pd.concat([df, pd.DataFrame(class_labels, columns=['class'])], axis = 1)
        print(plot_df.head())

        colors = ["navy", "turquoise", "darkorange", "green", "pink"]
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title('TSNE with perplexity=' + str(perplexity), fontsize = 20)
        for target, color in zip(seeds, colors):
            indicesToKeep = plot_df['class'] == target
            ax.scatter(plot_df.loc[indicesToKeep, 'tsne1']
                    , plot_df.loc[indicesToKeep, 'tsne2']
                    , c = color
                    , s = 50)
        ax.legend(seeds)
        ax.grid()
        plt.savefig("tsne_scatterplot_perplexity" + str(perplexity) + ".png")


    # Part 1d: k-means
    inertias = []
    mi_scores = []
    rand_scores = []
    purities = []

    def purity_score(y_true, y_pred, n):    
        # shift through each cluster
        purity = 0.0
        for k in range(n):
            # find labeled cluster index
            cind = np.where(y_pred == k)[0]  # all embed indx with specified cluster num
            true_v = y_true[cind]  # get associated class labels for true

            # get most common label
            val, count = np.unique(true_v, return_counts=True)
            count_argmax = count.argmax()
            vcount = count[count_argmax]
            vmax = val[count_argmax]
            # print('Found correct', vcount, 'out of', len(cind), 'in cluster', vmax)

            # now add to purity (which is essentially current accuracy of cluster label)
            purity += float(vcount)

        # final normalized purity score (closer to 1.0 the better)
        return purity / float(len(y_pred))


    for n in range(2, 21):
        kmeans = KMeans(n_clusters=n).fit(embeds)
        inertias.append(kmeans.inertia_)

        pred_labels = kmeans.predict(embeds)
        mi_score = normalized_mutual_info_score(true_labels, pred_labels)
        mi_scores.append(mi_score)

        rand_score = adjusted_rand_score(true_labels, pred_labels)
        rand_scores.append(rand_score)

        # calculate purity (our implementation without extra sklearn imports)
        purity = purity_score(np.array(true_labels, np.int32), pred_labels, n)
        print('Final purity', purity, 'for k =', n)
        purities.append(purity)

    plt.plot(list(range(2, 21)), inertias, "ro-")
    plt.title("K-means Inertia vs Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia value")
    plt.savefig("kmeans_inertia.png")
    plt.close()

    plt.plot(list(range(2, 21)), mi_scores, "bo-")
    plt.title("Normalized Mutual Information Score vs Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Normalized Mutual Information Score")
    plt.savefig("kmeans_mutual_info.png")
    plt.close()

    plt.plot(list(range(2, 21)), rand_scores, "go-")
    plt.title("Adjusted Rand Index Score vs Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Adjusted Rand Index Score")
    plt.savefig("kmeans_rand_index.png")
    plt.close()

    plt.plot(list(range(2, 21)), purities, "go-")
    plt.title("Purity Score vs Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Purity Score")
    plt.savefig("kmeans_purity.png")
    plt.close()
    
# run part 1
part1()
######### END of part 1 ############################################################################################################
#
#
#
#
#
#
#
#
#
#
######### Part 2 ###################################################################################################################
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


##### Preprocessing #################################################################################################################
def split_tweets(df):
    # splits documents into positive sentiment/negative sentiment tweets
    return df[df['sentiment'] == 0], df[df['sentiment'] == 1]

def load(name):
    data = pd.read_csv(name)
    data = data.sort_values(by=['sentiment'])
    return data

def preprocess(df):
    column, out = 'text', 'clean_text'

    # Remove empty rows
    df = df[pd.notnull(df[column])]

    # Lower case
    textc = df[column].str.lower()

    # Tokenize
    tokens = textc.str.split() 

    # Remove stopwords
# worked better without
#     stop_words = set(stopwords.words("english"))
#     new_words = ["the", "of", "and", "in", "a", "to", "for", "is", "on", "Re", "Subject", ">>", "*", "r"]
#     stop_words = stop_words.union(new_words)

#     tokens = tokens.apply(lambda words: [word for word in words if word not in stop_words])

    # Remove punctuation
    tokens = tokens.apply(lambda words: map(partial(re.sub, "&lt;/?.*?&gt;", " &lt;&gt; "), words))
    tokens = tokens.apply(lambda words: map(partial(re.sub, "<.*?>", ""), words))
    tokens = tokens.apply(lambda words: map(partial(re.sub, "(\\d|\\W)+", " "), words))
    #remove all the words that are less than 2
    tokens = tokens.apply(lambda words: map(partial(re.sub, r'\b\w{1,1}\b', " "), words))
    tokens = tokens.apply(lambda words: [word.strip() for word in words if word.strip()])

    # Lemmatize
# worked better without
#     lem = WordNetLemmatizer()
#     tokens = tokens.apply(lambda words: list(map(lem.lemmatize, words)))

#     # Stem
#     porter_stemmer = PorterStemmer()
#     tokens = tokens.apply(lambda words: list(map(porter_stemmer.stem, words)))

    # Join
    df[out] = tokens.apply(" ".join)
    return df


# load training data
train_data = load("IA3-train.csv")
train_data = preprocess(train_data)
train_data.head(5)

# load validation data
val_data = load("IA3-dev.csv")
val_data = preprocess(val_data)
val_data.head(5)

# spit the training tweets pos/neg
pos, neg = split_tweets(train_data)

y_train = train_data['sentiment']
y_test = val_data['sentiment']

print("\n Testing if common airlines are in the GloVe vocab:")
ge = GloVe_Embedder("GloVe_Embedder_data.txt")
airlines = ['usairways', 'americanair', 'united', 'jetblue', 'continentalair']
for a in airlines:
    v = ge.embed_str(a)
    if np.count_nonzero(v) == 0:
        print('Empty for', a)

print("\n Let's add unknown words from the training set to GloVe")
non_glove_words = set()
glove_words = set()
vocabulary = {}
vocab_index = 0
for a in list(train_data['clean_text']) + list(val_data['clean_text']):
    for w in str(a).split(' '):
        if ge.token_in_vocab(w):
            if w not in glove_words:
                glove_words.add(w)
                vocabulary[w] = vocab_index
                vocab_index += 1
        else:
            non_glove_words.add(w)
print(f'Found {len(non_glove_words)} words that were not in GloVe vocab')

# split multiple joined words. Pulled from https://stackoverflow.com/questions/195010/how-can-i-split-multiple-joined-words
def viterbi_segment(text):
    probs, lasts = [1.0], [0]
    for i in range(1, len(text) + 1):
        prob_k, k = max((probs[j] * word_prob(text[j:i]), j)
                        for j in range(max(0, i - max_word_length), i))
        probs.append(prob_k)
        lasts.append(k)
    words = []
    i = len(text)
    while 0 < i:
        words.append(text[lasts[i]:i])
        i = lasts[i]
    words.reverse()
    return words

def word_prob(word):
    return dictionary[word] / total

dictionary = Counter([w for w in glove_words if len(w) > 1])  # use the GloVe dictionary with non-short segments
max_word_length = max(map(len, dictionary))
total = float(sum(dictionary.values()))

failed_words = 0
special_words = ['airport', 'air', 'jet', 'blue', 'concord', 'business', 'sorry', 'help', 'bad', 'poor', 'busy', 'good', 'life', 'festivity', 'error', 'travel', 'advisor', 'people', 'unprofessional', 'professional']
for non_word in non_glove_words:
    subwords = []
    for special in special_words:
        if special in non_word:
            subwords.append(special)
            non_word.replace(special, '')
        
    # split rest of words
    if len(non_word) > 0:
        unclean_subwords = viterbi_segment(non_word)
    else:
        unclean_subwords = []
    # if subwords are nonsense let's remove some of them
    for sub in unclean_subwords:
        if len(sub) <= 1:  # stuff like a, b, c, _
            continue
        elif len(sub) == 2 and sub[0] == sub[1]:  # aa, bb, cc, etc are not useful weights for embeddings
            continue
        subwords.append(sub.strip(' '))  # "cleaned"
    
    if len(subwords) > 1:  # we found new subwords
        embeds = ge.embed_list(subwords)
        
        # construct a new embedding that's just the weighted average of the words (based on length of word)
        # total_weight = 0.0
        # avg_embed = np.zeros(embeds.shape[1], np.float32)
        # for ind, sub in enumerate(subwords):
        #     weight = float(len(sub))
        #     avg_embed += weight * embeds[ind, :]
        #     total_weight += weight
        # avg_embed /= total_weight
        avg_embed = np.mean(embeds, axis=0)
        
        # add it to our dictionary
        ge.embedding_dict[non_word] = avg_embed
        
        # add to vocabulary for tokenizer
        vocabulary[non_word] = vocab_index
        vocab_index += 1
    else:
        ge.embedding_dict[non_word] = np.zeros((200,), np.float32)
        vocabulary[non_word] = vocab_index
        vocab_index += 1
        failed_words += 1
print(f'Failed to sub-divide {failed_words} words')

# save the new GloVe embeddings
ge.save_to_file('Custom_Glove_Embedding.txt')

# reload the embeddings
ge = GloVe_Embedder('Custom_Glove_Embedding.txt')
#### End of "preprocessing" for part 2 ###############################################################################################




######################################################################################################################################
# try running PCA on all positive/negative tweets words
tweet_words = set()
positive_words = set()
negative_words = set()
for tweet, sent in zip(train_data['clean_text'], train_data['sentiment']):
    for w in tweet.split(' '):
        tweet_words.add(w)
        if sent == 1:
            positive_words.add(w)
        else:
            negative_words.add(w)

embeds = ge.embed_list(list(tweet_words))
pca = PCA(n_components=3)
pca.fit(embeds)
pca_positive = pca.transform(ge.embed_list(list(positive_words)))
pca_negative = pca.transform(ge.embed_list(list(negative_words)))
fig = plt.figure(figsize=(7,7))
ax = plt.axes(projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)
plt.scatter(pca_positive[:, 0], pca_positive[:, 1], pca_positive[:, 2], label='Positive', alpha=0.5)
plt.scatter(pca_negative[:, 0], pca_negative[:, 1], pca_negative[:, 2], label='Negative', alpha=0.5)
ax.legend()
ax.grid()
plt.savefig("pca_experimental.png")



def tweet_embedding_matrix(BAG, tweets, vocab, embed_dim=200):
    """ Converts a Count/TFID vectorized version of a tweet into a weighted average embedding matrix that represents the tweet """
    n, m = BAG.shape
    tweet_mat = np.zeros((n, embed_dim), np.float32)
    
    # construct weighted average for tweets
    missed_words = set()
    for t in range(n):  # iter through each tweet
        words = tweets[t].split(' ')
        embeds = ge.embed_list(words)  # embed words
        
        total_weight = 0.0
        for w_ind, word in enumerate(words):  # iter through each word
            if word in vocab:  # only if vectorizer weight exists
                weight = BAG[t, vocab[word]]
                tweet_mat[t, :] += weight * embeds[w_ind, :]
                total_weight += weight
            else:
                missed_words.add(word)
        
        # normalize
        if np.abs(total_weight) >= 1e-8:
            tweet_mat[t, :] /= total_weight
    
    print('Missed words', len(missed_words))
    return tweet_mat
    
tfIdfVectorizer = TfidfVectorizer(use_idf=True, lowercase=False, vocabulary=vocabulary) 

# fit transform returns bag of words
tfIdf_X = tfIdfVectorizer.fit_transform(train_data['clean_text'])
tfIdf_X_val = tfIdfVectorizer.transform(val_data['clean_text'])

# tweet GloveEmbedding
tfTweets = tweet_embedding_matrix(tfIdf_X, list(train_data['clean_text']), vocabulary)
tfTweets_val = tweet_embedding_matrix(tfIdf_X_val, list(val_data['clean_text']), vocabulary)

# Let's show the dimension reduced embeddings
pca = PCA(n_components=2)
      
# fit on all tweets
pca.fit(tfTweets)

# now do the positive from tfTweets
tfTweets_pos = tweet_embedding_matrix(tfIdfVectorizer.transform(pos['clean_text']), list(pos['clean_text']), vocabulary)
tfTweets_neg = tweet_embedding_matrix(tfIdfVectorizer.transform(neg['clean_text']), list(neg['clean_text']), vocabulary)

# now transform for positive/negative separately
pca_positive = pca.transform(tfTweets_pos)
pca_negative = pca.transform(tfTweets_neg)
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

# plot dimension reduced versions of positive and negative tweets
plt.scatter(pca_positive[:, 0], pca_positive[:, 1], label='Positive', alpha=0.5)
plt.scatter(pca_negative[:, 0], pca_negative[:, 1], label='Negative', alpha=0.5)
ax.legend()
ax.grid()
plt.savefig("pca_tweet_avg_embed_experimental.png")

#Trying decision tree with ENTROPY
print("\nDecision Tree with Entropy measure")
DTREE = DecisionTreeClassifier(criterion='entropy').fit(tfTweets, y_train)
y_pred_DTREE = DTREE.predict(tfTweets_val)
report = classification_report(y_test, y_pred_DTREE, output_dict=True)
print("Decision Tree with entropy report")
print(pd.DataFrame(report).transpose())

#Trying decision tree with GINI index
print("\nDecision Tree with Gini Index measure")
DTREE = DecisionTreeClassifier(criterion='gini').fit(tfTweets, y_train)
y_pred_DTREE = DTREE.predict(tfTweets_val)
report = classification_report(y_test, y_pred_DTREE, output_dict=True)
print("Decision Tree with gini report")
print(pd.DataFrame(report).transpose())

print("\nRandom Forest with 100 estimators")
forest = RandomForestClassifier(n_estimators=100, random_state=100)
forest.fit(tfTweets,y_train)
predictions = forest.predict(tfTweets_val)
report = classification_report(y_test, predictions, output_dict=True)
print("Ensemble report")
print(pd.DataFrame(report).transpose())

print("\nLet's run linear SVM now")
def test_hyperparameters(parameters, log=True):
    if log:
        print("\nTesting hyperparameters:", parameters)
    
    # train SVM using chosen parameters
    svc = svm.SVC(**parameters)
    svc.fit(tfTweets, train_data['sentiment'])
    n_vec = np.sum(svc.n_support_)
    
    # find validation performance (accuracy)
    acc_dev = svc.score(tfTweets_val, val_data['sentiment'])
    acc_train = svc.score(tfTweets, train_data['sentiment'])
    if log:
        print("Validation accuracy:", acc_dev, "Train accuracy:", acc_train)
    return acc_dev, acc_train, svc, n_vec

all_c = []
all_train_acc = []
all_val_acc = []
all_n_vec = []

def scan_c_param(c_list, start_params={'kernel': 'linear'}):
    global all_train_acc, all_val_acc, all_c, all_n_vec
    parameters = deepcopy(start_params)
    best_acc = 0.0
    best_train_acc = 0.0
    best_params = None
    best_svc = None
    for c in c_list:
        parameters['C'] = c
        acc_dev, acc_train, svc, n_vec = test_hyperparameters(parameters)
        all_c.append(c)
        all_train_acc.append(acc_train)
        all_val_acc.append(acc_dev)
        all_n_vec.append(n_vec)
        if acc_dev > best_acc:
            best_acc = acc_dev
            best_train_acc = acc_train
            best_params = deepcopy(parameters)
            best_svc = svc
    print('\nBest params', best_params, 'with validation accuracy', best_acc, 'train acc', best_train_acc)
    
cs = [.0001, .001, .01, 0.1, 1, 10]
scan_c_param(cs)


print("\nLet's just run on the average/non-tfid of the embeddings")
tfTweets = np.zeros((len(train_data['clean_text']), 200))
tfTweets_val = np.zeros((len(val_data['clean_text']), 200))

for i, w in enumerate(train_data['clean_text']):
    tfTweets[i, :] = np.mean(ge.embed_list(w.split(' ')), axis=0)

for i, w in enumerate(val_data['clean_text']):
    tfTweets_val[i, :] = np.mean(ge.embed_list(w.split(' ')), axis=0)
    
cs = [.0001, .001, .01, 0.1, 1, 10]
scan_c_param(cs)

# this resulted in good performance!

# run kernel SVM on tfTweets
#based on the previous assigment the best C and gamma for rbf were 10 and 0.1
print("\nSVM with RBF kernel")
SVM = svm.SVC(kernel='rbf', C=1, gamma=0.01).fit(tfTweets, y_train)
y_pred_svm = SVM.predict(tfTweets_val) # predict y hat
report = classification_report(y_test, y_pred_svm, output_dict=True)
print("SVM report")
print(pd.DataFrame(report).transpose())

#Trying decision tree with ENTROPY
print("\nDecision Tree with Entropy measure")
DTREE = DecisionTreeClassifier(criterion='entropy').fit(tfTweets, y_train)
y_pred_DTREE = DTREE.predict(tfTweets_val)
report = classification_report(y_test, y_pred_DTREE, output_dict=True)
print("Decision Tree with entropy report")
print(pd.DataFrame(report).transpose())

#Trying decision tree with GINI index
print("\nDecision Tree with Gini Index measure")
DTREE = DecisionTreeClassifier(criterion='gini').fit(tfTweets, y_train)
y_pred_DTREE = DTREE.predict(tfTweets_val)
report = classification_report(y_test, y_pred_DTREE, output_dict=True)
print("Decision Tree with gini report")
print(pd.DataFrame(report).transpose())

print("\nRandom Forest with 100 estimators")
forest = RandomForestClassifier(n_estimators=100, random_state=100)
forest.fit(tfTweets,y_train)
predictions = forest.predict(tfTweets_val)
report = classification_report(y_test, predictions, output_dict=True)
print("Ensemble report")
print(pd.DataFrame(report).transpose())
####### End of BAG of embeddings #######################################################################################################
