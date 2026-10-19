#This file contains all the functions required to train a tokenizer and then encode/decode strings to and from tokens

def normalize_text(text):
    normal = text.strip().lower()
    return normal

def split_sentence(text):
    word_list = text.split()
    return word_list

def to_characters(word):
    return list(word) + ['</w>']

def normalize(corpus):
    processed = []

    for sentence in corpus:
        sentence = normalize_text(sentence)
        words = split_sentence(sentence)
        sentence_list = []
        for word in words:
            sentence_list.append(to_characters(word))

        processed.append(sentence_list)

    return processed

def corpus_to_word_frequency(corpus):

    word_frequency = dict()

    for sentence in corpus:
        for word in sentence:
            word = tuple(word)
            if word in word_frequency:
                word_frequency[word] += 1
            else:
                word_frequency[word] = 1
    return word_frequency

def generate_initial_vocab(word_freq):
    vocab = []
    seen = set()
    for word in word_freq:
        for character in word:
            if character not in seen:
                vocab.append(character)
                seen.add(character)
    return vocab

def count_pairs(word_freq):
    pairs = dict()
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

def merge_word (word, merge_pair):
    new_word = []
    i = 0
    while i < (len(word)):
        if i < len(word)-1 and (word[i],word[i+1]) == merge_pair:
            new_word.append(word[i] + word[i+1])
            i+=1
        else:
            new_word.append(word[i])
        i += 1
    return new_word


def apply_merge(word_freq, max_pair):
    new_freq = {}

    for word, freq in word_freq.items():
        new_word = tuple(merge_word(word, max_pair))
        new_freq[new_word] = new_freq.get(new_word, 0) + freq

    return new_freq

def byte_pair_encoding(corpus, MAX_MERGES):
    merge_rules = []

    corpus = normalize(corpus)
    word_freq = corpus_to_word_frequency(corpus)

    vocab = generate_initial_vocab(word_freq)

    while len(merge_rules) < MAX_MERGES:
        pairs = count_pairs(word_freq)
        if len(pairs) == 0: 
            break
        max_pair = get_frequent_pair(pairs)
        merge_rules.append(max_pair)
        if (max_pair[0] + max_pair[1]) not in vocab:
            vocab.append(max_pair[0] + max_pair[1])    
        word_freq = apply_merge(word_freq, max_pair)

    return word_freq, merge_rules, vocab


def generate_vocab_dictionary(vocab):
    vocab_dict = {
        "<PAD>": 0,
        "<BOS>": 1,
        "<EOS>": 2,
        "<UNK>": 3
    }

    for token in vocab:
        if token not in vocab_dict:
            vocab_dict[token] = len(vocab_dict)

    return vocab_dict

def tokenize_word(word, merge_rules):
    word = to_characters(word)
    for merge in merge_rules:
        word = merge_word(word, merge)
    return word


def tokenize_corpus(corpus, merge_rules):
    cache = {}
    tokenized_corpus =[]

    for sentence in corpus:
        sentence = normalize_text(sentence)
        words = split_sentence(sentence)
        tokens = []
        for word in words:
            if word not in cache:
                cache[word] = tokenize_word(word, merge_rules)

            tokens.extend(cache[word])

        tokenized_corpus.append(tokens)
    return tokenized_corpus

def encode(tokens, vocab):
    encoded = [vocab["<BOS>"]]
    for token in tokens:
        if token in vocab:
            encoded.append(vocab[token])
        else:
            encoded.append(vocab["<UNK>"])
    encoded.append(vocab["<EOS>"])
    return encoded


def decode(encoded, vocab):
    id_to_vocab = {id: token for token, id in vocab.items()}
    tokens = []
    decoded = ""
    for num in encoded:
        token = id_to_vocab.get(num, "<UNK>")
        
        if token in ("<BOS>","<EOS>","<PAD>"):
            continue
        elif token.endswith("</w>"):
            decoded += (token[:-len("</w>")] + " ")
        elif token == "<UNK>":
            decoded += " <unk> "
        else:
            decoded += token

    return "".join(decoded).strip()
