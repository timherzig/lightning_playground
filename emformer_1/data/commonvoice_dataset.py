import os
import warnings
import torchaudio

from torch import Tensor, transpose
from typing import Tuple

import pandas as pd

class CommonVoiceDataset():

    def __init__(self, split, root_dir, n_rows=None) -> None:
        
        self.audio_dir = os.path.join(root_dir, 'clips')

        self.dataset = pd.read_table(os.path.join(root_dir, split + '.tsv'), nrows=n_rows)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index) -> Tuple[Tensor, int, str]:
        audio_path = os.path.join(self.audio_dir, self.dataset.iloc[index, 1]) # 1 is the path
        transcription = self.dataset.iloc[index, 2]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            speech_array, sampling_rate = torchaudio.load(audio_path, format='mp3')

        audio_len = speech_array.size()[1]

        speech_array = transpose(speech_array, 0, 1)

        return speech_array, Tensor(audio_len), transcription