from torch import squeeze

from torch.nn.utils.rnn import pad_sequence

def pad_batch(batch):
    speech_array, audio_len, transcription = zip(*batch)

    speech_array = pad_sequence(speech_array, batch_first=True)

    return speech_array, audio_len, transcription