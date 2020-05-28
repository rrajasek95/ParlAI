import hashlib
import json
import os
import random
import pickle as pkl
import re

from parlai.core.teachers import FixedDialogTeacher

random.seed(1337)

def strip_ssml(text):
    return re.sub("<.*?>", "", text)


def preprocess_conversations(data_path):
    conversations_file_path = os.path.join(data_path, "athena-conversations.json")

    with open(conversations_file_path, 'r') as conversations_file:
        conversation_data = json.load(conversations_file)

    random.shuffle(conversation_data)

    TRAIN_SIZE = 0.8
    VALID_SIZE = 0.1

    num_convs = len(conversation_data)
    num_train = int(num_convs * TRAIN_SIZE)
    num_valid = int(num_convs * VALID_SIZE)

    train_convs = conversation_data[:num_train]
    valid_convs = conversation_data[num_train: num_train + num_valid]
    test_convs = conversation_data[num_train + num_valid:]

    split_names = ['train', 'valid', 'test']
    split_convs = [train_convs, valid_convs, test_convs]

    checksum_entities = []  # A sequence of strings to be used for computing a checksum
    for (split, convs) in zip(split_names, split_convs):
        processed_conversations = []


        for conv in convs:
            previous_turns = []

            skip_conversation = False

            for turn in conv["history_turns"]:
                if not isinstance(turn["user_text"], str) or not isinstance(turn["athena_resp"], str):
                    skip_conversation = True
                    break

                previous_turns.append(turn["user_text"])
                previous_turns.append(strip_ssml(turn["athena_resp"]))

            if skip_conversation:
                continue  # Invalid data, skip the conversation

            previous_turns.append(conv["this_turn_text"])

            labels = []
            candidates = []

            for candidate in conv["response_candidates"]:
                if candidate["label"] == "should_say":
                    labels.append(strip_ssml(candidate["candidate_text"]))

                candidates.append(strip_ssml(candidate["candidate_text"]))
            text = "\n".join(previous_turns)

            conv_data = [text, labels, candidates]
            processed_conversations.append([conv_data])  # Each entry is a single episode
            checksum_entities.append(conv["last_turn_topic"] if conv["last_turn_topic"] else "")

        with open(os.path.join(data_path, f'{split}.pkl'), 'wb') as split_file:
            pkl.dump(processed_conversations, split_file)

    m = hashlib.sha256()
    m.update(bytes("".join(checksum_entities), encoding="utf8"))
    digest = m.hexdigest()

    with open(os.path.join(data_path, 'SHA256SUM'), 'w') as digest_file:
        digest_file.write(digest)


def build(opt):
    data_path = os.path.join(opt['datapath'], 'athena')

    if not os.path.exists(os.path.join(data_path, 'SHA256SUM')):
        preprocess_conversations(data_path)


class AthenaTopicalConversationsTeacher(FixedDialogTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        self.opt = opt
        self.datatype = opt.get('datatype', 'train').split(':')[0]

        self.datapath = os.path.join(
            self.opt['datapath'],
            'athena'
        )

        build(opt)

        if shared:
            self.data = shared['data']
        else:
            self._setup_data()

        self.num_exs = len(self.data)
        self.num_eps = len(self.data)

        self.reset()

    @classmethod
    def add_cmdline_args(cls, argparser):
        pass

    def num_episodes(self) -> int:
        return self.num_eps

    def num_examples(self) -> int:
        return self.num_exs

    def _setup_data(self):
        with open(os.path.join(self.datapath,
                               f"{self.datatype}.pkl"), 'rb') as conversations_file:
            self.data = pkl.load(conversations_file)

    def get(self, episode_idx, entry_idx=0):
        ep = self.data[episode_idx]
        ep_i = ep[entry_idx]
        episode_done = entry_idx >= (len(ep) - 1)

        action = {
            'text': ep_i[0],
            'labels': ep_i[1],
            'label_candidates': ep_i[2],
            'episode_done': episode_done
        }

        return action

    def share(self):
        shared = super().share()
        shared['data'] = self.data

        return shared

class DefaultTeacher(AthenaTopicalConversationsTeacher):
    pass
