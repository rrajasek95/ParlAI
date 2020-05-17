import json
import os

from parlai.core.teachers import FixedDialogTeacher

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
            prev_turn_data = None
            for (turn, turn_data) in enumerate(data[conversation_id]["content"]):
                if turn % 2 == 1:
                    conv_data.append([
                        prev_turn_data["message"],  # text (agent 1)
                        turn_data["message"],  # label (agent 2)
                    ])
                prev_turn_data = turn_data


            self.data.append(conv_data)  # Load conversation-wise

    def get(self, episode_idx, entry_idx=0):
        ep = self.data[episode_idx]
        ep_i = ep[entry_idx]
        episode_done = entry_idx >= (len(ep) - 1)

        action = {
            'text': ep_i[0],
            'labels': [ep_i[1]],
            'episode_done': episode_done
        }

        return action

    def share(self):
        shared = super().share()
        shared['data'] = self.data

        return shared


class DefaultTeacher(TopicalChatsTeacher):
    pass