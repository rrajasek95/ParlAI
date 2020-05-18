import json
import os
import random

from parlai.core.teachers import FixedDialogTeacher

random.seed(1337)

class TopicalChatsTeacher(FixedDialogTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt

        self.datatype = opt.get('datatype', 'train').split(':')[0]

        if self.datatype in ['valid', 'test']:
            self.datatype = f'{self.datatype}_freq'  # TODO: implement flags to perform validation over rare sets

        self.datapath = os.path.join(
            self.opt['datapath'],
            'alexa-prize-topical-chat-dataset',
            'conversations',
            self.datatype + '.json'
        )
        if shared:
            self.data = shared['data']
        else:
            self._setup_data()

        self.num_exs = sum([len(d) for d in self.data])  # Num examples
        self.num_eps = len(self.data)  # Num of conversations
        self.reset()

    @classmethod
    def add_cmdline_args(cls, argparser):
        pass

    def num_episodes(self) -> int:
        return self.num_eps

    def num_examples(self) -> int:
        return self.num_exs

    def _setup_data(self):
        with open(self.datapath, 'r') as data_file:
            data = json.load(data_file)

        self.data = []

        conversation_ids = list(data.keys())

        for conversation_id in conversation_ids:
            conv_data = []
            reverse_conversation = []

            turns = data[conversation_id]["content"]

            for i in range(1, len(turns)):
                distractor_conv_id = random.choice(conversation_ids)
                if distractor_conv_id == conversation_id:
                    distractor_conv_id = random.choice(conversation_ids)
                distractor_turn = random.choice(data[distractor_conv_id]["content"])

                turn_data = [
                    turns[i - 1]["message"],
                    turns[i]["message"],
                    [turns[i]["message"], distractor_turn["message"]]
                ]

                if i % 2 == 1:
                    conv_data.append(turn_data)
                else:
                    reverse_conversation.append(turn_data)

            self.data.append(conv_data) # Load conversation-wise
            self.data.append(reverse_conversation)

    def get(self, episode_idx, entry_idx=0):
        ep = self.data[episode_idx]
        ep_i = ep[entry_idx]
        episode_done = entry_idx >= (len(ep) - 1)

        action = {
            'text': ep_i[0],
            'labels': [ep_i[1]],
            'label_candidates': ep_i[2],
            'episode_done': episode_done
        }

        return action

    def share(self):
        shared = super().share()
        shared['data'] = self.data

        return shared


class DefaultTeacher(TopicalChatsTeacher):
    pass