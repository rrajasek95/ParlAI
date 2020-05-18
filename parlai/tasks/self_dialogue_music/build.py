import hashlib
import os
from random import Random
import pickle as pkl

def generate_file_splits(data_path, random_state=1337):
    conversation_list = [file for file in os.listdir(data_path)
                         if '.txt' in file]  # We extract all text files
    Random(random_state).shuffle(conversation_list)

    # Do a 80:10:10 split
    TRAIN_SIZE = 0.8
    VALID_SIZE = 0.1
    num_convs = len(conversation_list)

    num_train = int(num_convs * TRAIN_SIZE)
    num_valid = int(num_convs * VALID_SIZE)

    train_files = conversation_list[:num_train]
    valid_files = conversation_list[num_train:num_train + num_valid]
    test_files = conversation_list[num_train + num_valid:]

    return train_files, valid_files, test_files


def preprocess_conversations(data_path):
    split_files = list(generate_file_splits(data_path))
    splits = ['train', 'valid', 'test']

    split_string = ""

    for (split, files) in zip(splits, split_files):
        conversations = []

        for file in files:
            with open(os.path.join(data_path, file), 'r') as data_file:
                turns = [turn.strip() for turn in data_file.readlines()]

            conversation = []
            for i in range(1, len(turns), 2):
                conversation.append([
                    turns[i - 1],
                    turns[i]
                ])

            conversations.append(conversation)

        with open(os.path.join(data_path, f'{split}.pkl'), 'wb') as split_file:
            pkl.dump(conversations, split_file)

        split_string += "".join(files)

    m = hashlib.sha256()
    m.update(bytes(split_string, encoding='utf8'))
    digest = m.hexdigest()

    with open(os.path.join(data_path, 'SHA256SUM'), 'w') as digest_file:
        digest_file.write(digest)

def build(opt):
    data_path = os.path.join(opt['datapath'],  'self-dialogue-music')
    if not os.path.exists(os.path.join(data_path, 'SHA256SUM')):
        preprocess_conversations(data_path)