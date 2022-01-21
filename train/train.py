import glob
import os
import sys
from tkinter.tix import Tree
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

img_save = True
model_save = True
embedding_size = 512
hidden_size = 2048
epoch = 10


def main():
    txt_file_path = os.path.join(local_path, os.pardir, 'data', 'lyrics', '*')
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
    print("train 크기: {}".format(len(enc_train)))
    print("val   크기: {}".format(len(enc_val)))

    model = TextGenerator(tokenizer.num_words + 1, embedding_size, hidden_size)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')\
        , optimizer=tf.keras.optimizers.Adam())
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(local_path, 'model', 'model.cpkt'),
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
    hist = model.fit(enc_train, dec_train, epochs=epoch, validation_data=(enc_val, dec_val), callbacks=[model_checkpoint_callback])
    if img_save:
        fig, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()

        loss_ax.plot(hist.history['loss'], 'y', label='train loss')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        loss_ax.legend(loc='upper left')

        acc_ax.plot(hist.history['val_loss'], 'b', label='validation acc')
        acc_ax.set_ylabel('val_loss')
        acc_ax.legend(loc='upper right')
        plt.savefig(os.path.join(local_path,"result", "loss.png"))


if __name__ == '__main__':
    local_path = os.getcwd()
    sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    from module.preprocessing import pre_sentences
    from module.model import TextGenerator
    main()