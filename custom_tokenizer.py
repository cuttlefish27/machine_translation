#This file contains all the functions required to train a tokenizer and then encode/decode strings to and from tokens

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
    for word in word_freq:
        for character in word:
            if character not in vocab:
                vocab.append(character)
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
    new_freq = dict()
    for word in word_freq:
        new_word = merge_word(word, max_pair)
        new_freq[tuple(new_word)] = word_freq[word]
    return new_freq

def bytePairEncoding(corpus, MAX_MERGES):
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

def generateVocabDictionary(vocab):
    vocab_dict = dict()
    vocab_dict["<PAD>"] = 0
    vocab_dict["<BOS>"] = 1
    vocab_dict["<EOS>"] = 2
    vocab_dict["<UNK>"] = 3

    vocab_dict.update({word: idx +4  for idx, word in enumerate(vocab)})

    return vocab_dict

def tokenizeSentence(sentence: str, merge_rules):
    sentence = normalize_text(sentence)
    sentence = split_sentence(sentence)

    sentence_list = []
    for word in sentence:
        word = toCharacters(word)
        sentence_list.append(word)
    
    for merge in merge_rules:
        for i in range(len(sentence_list)):
            sentence_list[i] = merge_word(sentence_list[i], merge)
    
    tokens = [token for word in sentence_list for token in word]
    return tokens

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
    for num in encoded:
        if num in vocab.values():
            tokens.append(id_to_vocab[num])
        else:
            tokens.append("<UNK>")
    
    decoded = ""
    for token in tokens:
        if token in ("<BOS>","<EOS>","<PAD>"):
            continue
        elif token == "</w>":
            decoded += " "
        elif token == "<UNK>":
            decoded += " <unk> "
        else:
            decoded += token

    decoded.strip()
    return decoded
