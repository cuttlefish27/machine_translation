import pandas as pd

def normalize_text(text):
    normal = text.strip().lower()
    return normal

def split_sentence(text):
    word_list = text.split()
    return word_list

def toCharacters(word):
    return list(word) + ['</w>']

def normalize(corpus):
    processed = []

    for sentence in corpus:
        sentence = normalize_text(sentence)
        words = split_sentence(sentence)
        sentence_list = []
        for word in words:
            sentence_list.append(toCharacters(word))

        processed.append(sentence_list)

    return processed



def generate_initial_vocab(corpus):
    vocab = []
    for sentence in corpus:
        for word in sentence:
            for character in word:
                if character not in vocab:
                    vocab.append(character)
    return vocab


def count_pairs(corpus):
    pairs = dict()
    word_freq = dict()
    for sentence in corpus:
        for word in sentence:
            word = tuple(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    
    for word in word_freq:
        for i in range (len(word) -1):
            pair = (word[i], word[i+1])

            
            if pair in pairs:
                pairs[pair] += word_freq[word]
            else:
                pairs[pair] = word_freq[word]
    return pairs


def get_frequent_pair(pairs):
    max_pair = None

    for key, value in pairs.items():
        if max_pair is None or value > pairs[max_pair]:
            max_pair = key

    return max_pair


def merge_tokens(sentence, max_pair):
    new_sentence = []
    for word in sentence:
        new_word = []
        i = 0
        while i < (len(word)):
            if i < len(word)-1 and (word[i],word[i+1]) == max_pair:
                new_word.append(word[i] + word[i+1])
                i+=1
            else:
                new_word.append(word[i])
            i += 1
        new_sentence.append(new_word)
    return new_sentence


def merge_corpus(corpus, max_pair):
    new_corpus = []
    for sentence in corpus:
        new_sentence = merge_tokens(sentence, max_pair)
        new_corpus.append(new_sentence)

    return new_corpus


def bytePairEncoding(corpus, MAX_VOCAB):
    merge_rules = []

    corpus = normalize(corpus)
    vocab = generate_initial_vocab(corpus)
    while len(merge_rules) < MAX_VOCAB:
        pairs = count_pairs(corpus)
        if len(pairs) == 0: 
            break
        max_pair = get_frequent_pair(pairs)
        merge_rules.append(max_pair)
        if (max_pair[0] + max_pair[1]) not in vocab:
            vocab.append(max_pair[0] + max_pair[1])    
        corpus = merge_corpus(corpus, max_pair)
    return corpus, merge_rules, vocab


df = pd.read_csv("spa-eng/spa.txt", sep='\t')
df = df.drop(df.columns[2], axis = 1)
corpus = df.values.ravel().tolist()
corpus, merge_rules, vocab = bytePairEncoding(corpus, 10)


print(merge_rules)