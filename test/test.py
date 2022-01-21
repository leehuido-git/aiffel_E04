import glob
import os
import sys
from tkinter.tix import Tree
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

img_save = True
model_save = True
embedding_size = 2
hidden_size = 8
epoch = 4


def main():
    txt_file_path = os.path.join(local_path, os.pardir, 'data', 'lyrics', '*')
    ckpt_path = os.path.join(local_path, os.pardir, 'train', 'model', 'model.cpkt')
    txt_list = glob.glob(txt_file_path)
    raw_corpus = []
    for txt_file in txt_list:
        with open(txt_file, 'r', encoding='UTF-8') as f:
            raw = f.read().splitlines()
            raw_corpus.extend(raw)
    print("데이터 크기:", len(raw_corpus))

    tensor, tokenizer = pre_sentences(raw_corpus)
    src_input = tensor[:, :-1]
    tgt_input = tensor[:, 1:]
    enc_train, enc_val, dec_train, dec_val = train_test_split(src_input, tgt_input, test_size=0.2, random_state=42)
    
    model = TextGenerator(tokenizer.num_words + 1, embedding_size, hidden_size)
    model.load_weights(ckpt_path)

    while True:
        sent = input()
        generate_text(model, tokenizer, init_sentence=("<start>"+sent), max_len=20)




if __name__ == '__main__':
    local_path = os.getcwd()
    sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    from module.preprocessing import pre_sentences
    from module.model import TextGenerator
    from module.gen_sentence import generate_text
    main()