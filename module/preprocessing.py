import re
import tensorflow as tf

def pre_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
    sentence = sentence.strip()
    sentence = '<start> ' + sentence + ' <end>'
    return sentence

def tokenize(corpus):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=12000, 
        filters=' ',
        oov_token="<unk>"
    )
    tokenizer.fit_on_texts(corpus)
    tensor = tokenizer.texts_to_sequences(corpus)
    temp_tensor = []
    for i in tensor:
        if(len(i))<=15:
            temp_tensor.append(i)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(temp_tensor, padding='post')
    return tensor, tokenizer

def pre_sentences(raw_corpus):
    corpus = []
    for sentence in raw_corpus:
        if((len(sentence)==0) or (sentence[-1] == ':')):
            continue
        pred_sentence = pre_sentence(sentence)
        corpus.append(pred_sentence)

    return tokenize(corpus)