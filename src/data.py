"""
Logic:
1. AudioDataLoader generate a minibatch from AudioDataset, the size of this
   minibatch is AudioDataLoader's batchsize. For now, we always set
   AudioDataLoader's batchsize as 1. The real minibatch size we care about is
   set in AudioDataset's __init__(...). So actually, we generate the
   information of one minibatch in AudioDataset.
2. After AudioDataLoader getting one minibatch from AudioDataset,
   AudioDataLoader calls its collate_fn(batch) to process this minibatch.

"""

import json
import math
import os
import sys

import numpy as np
import torch
import torch.utils.data as data

import copy
import librosa
import soundfile as sf

class AudioDataset(data.Dataset):

    def __init__(self, json_dir, batch_size, sample_rate=16000, segment=4.0, cv_maxlen=8.0):
        """
        Args:
            json_dir: directory including mix.json, s1.json and s2.json
            segment: duration of audio segment, when set to -1, use full audio

        xxx_infos is a list and each item is a tuple (wav_file, #samples)
        """
        super(AudioDataset, self).__init__()
        mix_json = os.path.join(json_dir, 'mix.json')
        s1_json = os.path.join(json_dir, 's1.json')
        s2_json = os.path.join(json_dir, 's2.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        with open(s1_json, 'r') as f:
            s1_infos = json.load(f)
        with open(s2_json, 'r') as f:
            s2_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[1]), reverse=True)
        sorted_mix_infos = sort(mix_infos)
        sorted_s1_infos = sort(s1_infos)
        sorted_s2_infos = sort(s2_infos)
        if segment >= 0.0:
            # segment length and count dropped utts
            segment_len = int(segment * sample_rate)  # 4s * 16000/s = 64000 samples
            drop_utt, drop_len = 0, 0
            for _, sample in sorted_mix_infos:
                if sample < segment_len:
                    drop_utt += 1
                    drop_len += sample
            print("Drop {} utts({:.2f} h) which is short than {} samples".format(
                drop_utt, drop_len/sample_rate/36000, segment_len))
            # generate minibach infomations
            minibatch = []
            start = 0
            while True:
                num_segments = 0
                end = start
                part_mix, part_s1, part_s2 = [], [], []
                while num_segments < batch_size and end < len(sorted_mix_infos):
                    utt_len = int(sorted_mix_infos[end][1])
                    if utt_len >= segment_len:  # skip too short utt
                        num_segments += math.ceil(utt_len / segment_len)
                        # Ensure num_segments is less than batch_size
                        if num_segments > batch_size:
                            # if num_segments of 1st audio > batch_size, skip it
                            if start == end: end += 1
                            break
                        part_mix.append(sorted_mix_infos[end])
                        part_s1.append(sorted_s1_infos[end])
                        part_s2.append(sorted_s2_infos[end])
                    end += 1
                if len(part_mix) > 0:
                    minibatch.append([part_mix, part_s1, part_s2,
                                      sample_rate, segment_len])
                if end == len(sorted_mix_infos):
                    break
                start = end
            self.minibatch = minibatch
        else:  # Load full utterance but not segment
            # generate minibach infomations
            minibatch = []
            start = 0
            while True:
                end = min(len(sorted_mix_infos), start + batch_size)
                # Skip long audio to avoid out-of-memory issue
                if int(sorted_mix_infos[start][1]) > cv_maxlen * sample_rate:
                    start = end
                    continue
                minibatch.append([sorted_mix_infos[start:end],
                                  sorted_s1_infos[start:end],
                                  sorted_s2_infos[start:end],
                                  sample_rate, segment])
                if end == len(sorted_mix_infos):
                    break
                start = end
            self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class AudioDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


def _collate_fn(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        sources_pad: B x C x T, torch.Tensor
    """
    # batch should be located in list
    assert len(batch) == 1
    mixtures, sources = load_mixtures_and_sources(batch[0])

    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures])

    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    sources_pad = pad_list([torch.from_numpy(s).float()
                            for s in sources], pad_value)
    # N x T x C -> N x C x T
    sources_pad = sources_pad.permute((0, 2, 1)).contiguous()
    return mixtures_pad, ilens, sources_pad


# Eval data part

class EvalDataset(data.Dataset):

    def __init__(self, mix_dir, mix_json, batch_size, sample_rate=16000):
        """
        Args:
            mix_dir: directory including mixture wav files
            mix_json: json file including mixture wav files
        """
        super(EvalDataset, self).__init__()
        assert mix_dir != None or mix_json != None
        
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[1]), reverse=True)
        sorted_mix_infos = sort(mix_infos)
        # generate minibach infomations
        minibatch = []
        start = 0
        while True:
            end = min(len(sorted_mix_infos), start + batch_size)
            minibatch.append([sorted_mix_infos[start:end],
                              sample_rate])
            if end == len(sorted_mix_infos):
                break
            start = end
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class EvalDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(EvalDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_eval


def _collate_fn_eval(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        filenames: a list contain B strings
    """
    # batch should be located in list
    assert len(batch) == 1
    mixtures, filenames = load_mixtures(batch[0])

    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in mixtures])

    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in mixtures], pad_value)
    ilens = torch.from_numpy(ilens)
    return mixtures_pad, ilens, filenames


# ASR data part
class AudioASRDataset(data.Dataset):
    def __init__(self, json_dir, batch_size, sample_rate=16000):
        """
        Args:
            json_dir: directory including mix.json, s1.json and s2.json
            segment: duration of audio segment, when set to -1, use full audio

        xxx_infos is a list and each item is a tuple (wav_file, #samples)
        """
        super(AudioASRDataset, self).__init__()
        s1_json = os.path.join(json_dir, 's1.json')
        with open(s1_json, 'r') as f:
            s1_infos = json.load(f)

        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(infos, key=lambda info: int(info[1]), reverse=False)
        sorted_s1_infos = sort(s1_infos)

        # generate minibach infomations
        minibatch = []
        start = 0
        while True:
            end = min(len(sorted_s1_infos), start + batch_size)
            minibatch.append([sorted_s1_infos[start:end], sample_rate])
            if end == len(sorted_s1_infos):
                break
            start = end
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)

class AudioASRDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(AudioASRDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_ASR

def _collate_fn_ASR(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioASRDataset.__getitem__()
    Returns:
        sources: a list containing B items, item is T np.ndarray
        targets: a list containing B items, item is L np.ndarray
        T varies from item to item.
        L varies from item to item. 
    """
    
    # batch should be located in list
    assert len(batch) == 1

    sources, targets = [], []
    s1_infos, sample_rate = batch[0]
    # for each utterance
    for s1_info in s1_infos:
        s1_path = s1_info[0]
        laborg = s1_info[2]
        cpulab = np.array(laborg, dtype=np.int32)
        # read wav file
        s1, _ = librosa.load(s1_path, sr=sample_rate)

        sources.append(s1)
        targets.append(cpulab)

    return sources, targets

class EvalAudioASRDataset(data.Dataset):
    def __init__(self, json_dir, s1_json, batch_size=1, sample_rate=16000):
        super(EvalAudioASRDataset, self).__init__()
        if json_dir is not None:
            s1_json = os.path.join(json_dir, 'eval.json')
        with open(s1_json, 'r') as f:
            s1_infos = json.load(f)
        minibatch = []
        start = 0
        while True:
            end = min(len(s1_infos), start + batch_size)
            minibatch.append([s1_infos[start:end], sample_rate])
            if end == len(s1_infos):
                break
            start = end
        self.minibatch = minibatch
        
    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)

class EvalAudioASRDataLoader(data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(EvalAudioASRDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_eval_ASR

def _collate_fn_eval_ASR(batch):
    assert len(batch) == 1
    sources, names = [], []
    s1_infos, sample_rate = batch[0]
    for s1_info in s1_infos:
        s1_path = s1_info[0]
        s1, _ = librosa.load(s1_path, sr=sample_rate)
        sources.append(s1)
        names.append(s1_path)

    return sources, names


# all data

class AudioAllDataset(data.Dataset):
    def __init__(self, json_dir, batch_size, sample_rate=16000):
        """
        Args:
            json_dir: directory including mix.json, s1.json and s2.json
            segment: duration of audio segment, when set to -1, use full audio

        xxx_infos is a list and each item is a tuple (wav_file, #samples)
        """
        super(AudioAllDataset, self).__init__()
        mix_json = os.path.join(json_dir, 'mix.json')
        s1_json = os.path.join(json_dir, 's1.json')
        s2_json = os.path.join(json_dir, 's2.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        with open(s1_json, 'r') as f:
            s1_infos = json.load(f)
        with open(s2_json, 'r') as f:
            s2_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(infos, key=lambda info: int(info[1]), reverse=False)
        sorted_mix_infos = sort(mix_infos)
        sorted_s1_infos = sort(s1_infos)
        sorted_s2_infos = sort(s2_infos)
        
        # Load full utterance but not segment
        # generate minibach infomations
        minibatch = []
        start = 0
        while True:
            end = min(len(sorted_mix_infos), start + batch_size)
            minibatch.append([sorted_mix_infos[start:end], sorted_s1_infos[start:end], sorted_s2_infos[start:end], sample_rate])

            if end == len(sorted_mix_infos):
                break
            start = end
            if sorted_mix_infos[start][1] > 320000:
                break
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)

class AudioAllDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(AudioAllDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_All

def _collate_fn_All(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioAllDataset.__getitem__()
    Returns:
        sources: a list containing B items, item is T np.ndarray
        targets: a list containing B items, item is L np.ndarray
        T varies from item to item.
        L varies from item to item. 
    """
    # batch should be located in list
    assert len(batch) == 1
    mixtures, sources, targets = [], [], []
    mix_infos, s1_infos, s2_infos, sample_rate = batch[0]
    # for each utterance
    
    for mix_info, s1_info, s2_info in zip(mix_infos, s1_infos, s2_infos):
        mix_path = mix_info[0]
        s1_path = s1_info[0]
        s2_path = s2_info[0]
        laborg = mix_info[2]
        cpulab = np.array(laborg, dtype=np.int32)
        # read wav file
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        s1, _ = librosa.load(s1_path, sr=sample_rate)
        s2, _ = librosa.load(s2_path, sr=sample_rate)
        s = np.dstack((s1, s2))[0]  # T x C, C = 2
        mixtures.append(mix)
        sources.append(s)
        targets.append(cpulab)


    ilens = np.array([mix.shape[0] for mix in sources])
    pad_value = 0
    padded_mixtures = pad_list([torch.from_numpy(mix).float() for mix in mixtures], pad_value)
    padded_sources = pad_list([torch.from_numpy(s).float() for s in sources], pad_value)
    padded_sources = padded_sources.transpose(1, 2)
    ilens = torch.from_numpy(ilens)
    return padded_mixtures, padded_sources, ilens, targets, mix_infos


class EvalAllDataset(data.Dataset):

    def __init__(self, mix_dir, mix_json, batch_size, sample_rate=16000):
        """
        Args:
            mix_dir: directory including mixture wav files
            mix_json: json file including mixture wav files
        """
        super(EvalAllDataset, self).__init__()

        assert mix_dir != None or mix_json != None

        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        minibatch = []
        start = 0
        while True:
            end = min(len(mix_infos), start + batch_size)
            minibatch.append([mix_infos[start:end], sample_rate])
            if end == len(mix_infos):
                break
            start = end
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class EvalAllDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(EvalAllDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_eval_all


def _collate_fn_eval_all(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        filenames: a list contain B strings
    """
    # batch should be located in list
    assert len(batch) == 1
    sources, names = [], []
    mix_infos, sample_rate = batch[0]
    for mix_info in mix_infos:
        mix_path = mix_info[0]
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        sources.append(mix)
        names.append(mix_path)

    # get batch of lengths of input sequences
    ilens = np.array([mix.shape[0] for mix in sources])

    # perform padding and convert to tensor
    pad_value = 0
    mixtures_pad = pad_list([torch.from_numpy(mix).float()
                             for mix in sources], pad_value)
    ilens = torch.from_numpy(ilens)
    return mixtures_pad, ilens, names








# ------------------------------ utils ------------------------------------
def load_mixtures_and_sources(batch):
    """
    Each info include wav path and wav duration.
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        sources: a list containing B items, each item is T x C np.ndarray
        T varies from item to item.
    """
    mixtures, sources = [], []
    mix_infos, s1_infos, s2_infos, sample_rate, segment_len = batch
    # for each utterance
    for mix_info, s1_info, s2_info in zip(mix_infos, s1_infos, s2_infos):
        mix_path = mix_info[0]
        s1_path = s1_info[0]
        s2_path = s2_info[0]
        assert mix_info[1] == s1_info[1] and s1_info[1] == s2_info[1]
        # read wav file
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        s1, _ = librosa.load(s1_path, sr=sample_rate)
        s2, _ = librosa.load(s2_path, sr=sample_rate)
        # merge s1 and s2
        s = np.dstack((s1, s2))[0]  # T x C, C = 2
        utt_len = mix.shape[-1]
        if segment_len >= 0:
            # segment
            for i in range(0, utt_len - segment_len + 1, segment_len):
                mixtures.append(mix[i:i+segment_len])
                sources.append(s[i:i+segment_len])
            if utt_len % segment_len != 0:
                mixtures.append(mix[-segment_len:])
                sources.append(s[-segment_len:])
        else:  # full utterance
            mixtures.append(mix)
            sources.append(s)
    return mixtures, sources



def load_mixtures(batch):
    """
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        filenames: a list containing B strings
        T varies from item to item.
    """
    mixtures, filenames = [], []
    mix_infos, sample_rate = batch
    # for each utterance
    for mix_info in mix_infos:
        mix_path = mix_info[0]
        # read wav file
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        mixtures.append(mix)
        filenames.append(mix_path)
    return mixtures, filenames


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


if __name__ == "__main__":
    import sys
    json_dir, batch_size = sys.argv[1:3]
    dataset = AudioDataset(json_dir, int(batch_size))
    data_loader = AudioDataLoader(dataset, batch_size=1,
                                  num_workers=4)
    for i, batch in enumerate(data_loader):
        mixtures, lens, sources = batch
        print(i)
        print(mixtures.size())
        print(sources.size())
        print(lens)
        if i < 10:
            print(mixtures)
            print(sources)
