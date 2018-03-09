# coding: utf-8
"""
Extraction of the train and test datasets 
for Voice Conversion over VCTK-Corpus.

Using the 28 speakers version.

Usage:
	extraction.py [options] <in_dir> <out_dir>

options:
	--num_workers=<n>		Num workers.
	--hparams=<params>	Hyper parameters [default: ].
	-h, --help					Show help message.
"""
from docopt import docopt
import os
from os.path import join, exists
from multiprocessing import cpu_count
from tqdm import tqdm
from hparams import hparams
from shutil import copy

target_speakers = ['p287', 'p282', 'p278', 'p277']	# 2M, 2F

def _rm_hidden(files):
    return [file for file in files if not file.startswith('.')]


def _maybe_all_data(in_dir):
    """return tuplas of subdirs with data."""
    if os.path.isdir(in_dir):
        dirs = _rm_hidden(os.listdir(in_dir))
        for subdir in dirs:
            if not subdir.startswith('.') and os.path.isdir(join(in_dir, subdir)):
                if 'train' in subdir and 'wav' in subdir:
                    wav_train = join(in_dir, subdir)
                elif 'train' in subdir and 'txt' in subdir:
                    txt_train = join(in_dir, subdir)
                elif 'test' in subdir and 'wav' in subdir:
                    wav_test = join(in_dir, subdir)
                elif 'test' in subdir and 'txt' in subdir:
                    txt_test = join(in_dir, subdir)

    return (wav_train, txt_train), (wav_test, txt_test)


def _get_speakers(in_dir):
    all_files = _rm_hidden(os.listdir(in_dir))
    speakers = list()
    for file in all_files:
        spk = file.split('_')[0]
        if spk not in speakers:
            speakers.append(spk)
    return speakers
    

def _is_equal_sentence(ref, cand):
    return ref == cand


def _get_common_sentences(in_dir, speakers):
    spk_idx = 0
    ref_sentences = [file for file in os.listdir(in_dir) if speakers[spk_idx] in file]
    speakers = [speakers[idx] for idx in range(len(speakers)) if idx != spk_idx]
    
    is_common = 0
    sentence_times = 0

    for step, file in enumerate(tqdm(ref_sentences)):
        sentence_times = 0
        ref_path = join(in_dir, file)
        with open(ref_path, 'r', encoding='utf-8') as f:
            ref = f.read()
            print(ref)
            for spk in speakers:
                spk_sentences = [join(in_dir, file) for file in _rm_hidden(os.listdir(in_dir))
                    if spk in file]
                for cand_path in spk_sentences:
                    with open(cand_path, 'r', encoding='utf-8') as c:
                        cand = c.read()
                        if _is_equal_sentence(ref, cand):
                            sentence_times += 1
                            print(cand)
        if sentence_times == len(speakers):
            is_common += 1
    print(is_common)

        
    

if __name__ == "__main__":
    args = docopt(__doc__)
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]

    # Override hyper parameters
    assert hparams.name == "wavenet_vocoder"

    # Presets
    if hparams.preset is not None and hparams.preset != "":
        preset = hparams.presets[hparams.preset]
        import json
        hparams.parse_json(json.dumps(preset))
        print("Override hyper parameters with preset \"{}\": {}".format(
            hparams.preset, json.dumps(preset, indent=4)))

    traindir, testdir = _maybe_all_data(in_dir)
    
    _get_common_sentences(traindir[1], _get_speakers(traindir[1]))
    # sort_data(testdir, join(out_dir, 'test'))


