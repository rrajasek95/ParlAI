import os
import pickle as pkl

from parlai.core.teachers import FixedDialogTeacher


class SelfDialogueMusicTeacher(FixedDialogTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt

        self.datatype = opt.get('datatype', 'train').split(':')[0]

        self.datapath = os.path.join(
            self.opt['datapath'],
            'self-dialogue-music',
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
        with open(os.path.join(self.datapath,
                               f"{self.datatype}.pkl"), 'rb') as conversations_file:
            self.data = pkl.load(conversations_file)

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


class DefaultTeacher(SelfDialogueMusicTeacher):
    pass
