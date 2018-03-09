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
from multiprocessing import cpu_count
from tqdm import tqdm
from hparams import hparams

train_speakers = ['p226', 'p227', 'p228', 'p230', 'p231',
	'p233', 'p236', 'p239', 'p243', 'p244', 'p250', 'p254', 'p256',
	'p258', 'p259', 'p267', 'p268', 'p269', 'p270', 'p273', 'p274',
	'p276', 'p277', 'p278', 'p279', 'p282', 'p286', 'p287']
test_speakers = ['p232', 'p257'] # 1F, 1M
target_speakers = ['p287', 'p282', 'p278', 'p277']	# 2M, 2F

def _check_common_sentences(in_dir):
	pass


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

