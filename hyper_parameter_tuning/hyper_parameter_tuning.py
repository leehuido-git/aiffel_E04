import glob
import os
import sys
from tkinter.tix import Tree
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

img_save = True
model_save = False
#embedding_size = [2, 4, 8, 16, 32, 64, 128, 256]
#hidden_size = [8, 16, 32, 64, 128, 256, 512, 1024]
embedding_size = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
hidden_size = [2048]
hist = []
epoch = 5


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

    for i in range(len(embedding_size)):
        for j in range(len(hidden_size)):
            model = TextGenerator(tokenizer.num_words + 1, embedding_size[i], hidden_size[j])
            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')\
                , optimizer=tf.keras.optimizers.Adam())
            hist.append(model.fit(enc_train, dec_train, epochs=epoch, validation_data=(enc_val, dec_val)))
            if model_save:
                model.save_weights(os.path.join(local_path, 'model', '{}__{}_lyricist'.format(i, j)))
            if img_save:
                fig, loss_ax = plt.subplots()
                loss_ax.set_title("embedding= {}, hidden= {}".format(embedding_size[i], hidden_size[j]))
                acc_ax = loss_ax.twinx()

                loss_ax.plot(hist[-1].history['loss'], 'y', label='train loss')
                loss_ax.set_xlabel('epoch')
                loss_ax.set_ylabel('loss')
                loss_ax.legend(loc='upper left')

                acc_ax.plot(hist[-1].history['val_loss'], 'b', label='validation acc')
                acc_ax.set_ylabel('val_loss')
                acc_ax.legend(loc='upper right')
                acc_ax.text(0, 0, "embedding= {}, hidden= {}".format(embedding_size[i], hidden_size[j]))
                plt.savefig(os.path.join(local_path,"result", "{}_{}_loss.png".format(i, j)))

    if img_save:
        plt.clf()
        plt.title("validation")
        plt.figure(figsize=(16, 16))
        plt.xlabel('epoch')
        plt.ylabel('val_loss')
        cmap = plt.get_cmap('jet_r')
        result = ("line num, embedding size, hidden size\n")
        cnt = 0
        for i in range(len(embedding_size)):
            for j in range(len(hidden_size)):            
                color = cmap(float(cnt)/len(hist))
#                plt.plot(hist[cnt].history['val_loss'], c=color, label='{}_val: em={}, hi={}'.format(cnt, embedding_size[i], hidden_size[j]))
                plt.plot(hist[cnt].history['val_loss'], c=color)
                plt.text(epoch-1, hist[cnt].history['val_loss'][-1], "{}".format(cnt))                
                result += '{}, {}, {}\n'.format(cnt, embedding_size[i], hidden_size[j])
                cnt += 1
#        plt.legend(loc='upper right')
        plt.savefig(os.path.join(local_path,"result", "loss.png"))
        with open(os.path.join(local_path,'result', 'result.csv'), 'w') as f:
            f.write(result)
            f.close()



if __name__ == '__main__':
    local_path = os.getcwd()
    sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    from module.preprocessing import pre_sentences
    from module.model import TextGenerator
    main()
