import pickle
import argparse
import logging

from sklearn.model_selection import train_test_split
from collections import Counter
from data_utils import parse_one_billion_word


def get_args():
    parser = argparse.ArgumentParser(
        description='Data Preprocessing for Named Entity Recognition')
    parser.add_argument('--train', type=str, default=None,
                        help='train data path')
    parser.add_argument('--dev', type=str, default=None,
                        help='dev data path')
    parser.add_argument('--test', type=str, default=None,
                        help='test data path')
    parser.add_argument('--unlabeled', type=str, default=None,
                        help='unlabeled data path')
    args = parser.parse_args()
    return args


def process(word):
    word = "".join(c if not c.isdigit() else '0' for c in word)
    return word


def process_file(data_file):
    logging.info("loading data from " + data_file + " ...")
    sents = []
    tags = []
    sent = []
    tag = []
    doc_count = 0
    with open(data_file, 'r', encoding='utf-8') as df:
        for line in df.readlines():
            if line[0:10] == '-DOCSTART-':
                doc_count += 1
                continue
            if line.strip():
                word = line.strip().split(" ")[0]
                t = line.strip().split(" ")[-1]
                sent.append(process(word))
                tag.append(t)
            else:
                if sent and tag:
                    sents.append(sent)
                    tags.append(tag)
                    if len(sents) % 500 == 0:
                        logging.info('%d docs are parsed, %d sentences are parsed' % (doc_count, len(sents)))
                sent = []
                tag = []

    return sents, tags


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M')
    args = get_args()
    train = process_file(args.train)
    dev = process_file(args.dev)
    test = process_file(args.test)
    logging.info('processing %s' % args.unlabeled)
    unlabeled = parse_one_billion_word(args.unlabeled)

    tag_counter = Counter(sum(train[1], []) +
                          sum(dev[1], []) + sum(test[1], []))

    with open("ner_tagfile", "w+", encoding='utf-8') as fp:
        fp.write('\n'.join(sorted(tag_counter.keys())))

    logging.info("#unlabeled data: {}".format(len(unlabeled)))

    with open("ner_unlabel.data", "w+", encoding='utf-8') as fp:
        fp.write("\n".join([" ".join(sent) for sent in unlabeled]))
    logging.info("unlabeled data saved to {}".format("ner_unlabel.data"))

    logging.info("#train data: {}".format(len(train[0])))
    logging.info("#dev data: {}".format(len(dev[0])))
    logging.info("#test data: {}".format(len(test[0])))

    pickle.dump(
        [train, dev, test], open("ner.data", "wb+"),
        protocol=-1)
