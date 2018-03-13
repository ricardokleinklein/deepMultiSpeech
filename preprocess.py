# coding: utf-8
"""
Preprocess dataset. The dataset is structured as:
- root
    - train
        - noisy
            - speakers
                - txt
                - wav
        - clean
            - speakers
                - txt
                - wav
    - test
        - noisy
            - speakers
                -txt
                -wav
        -clean
            -speakers
                - txt
                - wav

The speakers for test and for training are different.
The metadata generated will be different as well, depending
on the task to perform: Metadata.csv stores the paths to the
source and target audio files:
path_to_source | path_to_target | text | speaker

The resulting file contains the following information:
target audio | input melSpec | timesteps | text | speaker 

usage: preprocess.py [options] <in_dir> <out_dir>

options:
    --num_workers=<n>        Num workers.
    --hparams=<params>       Hyper parameters [default: ].
    -h, --help               Show help message.
"""
from docopt import docopt
import os
from multiprocessing import cpu_count
from tqdm import tqdm
import importlib
from hparams import hparams


def preprocess(mod, in_dir, out_root, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    metadata = mod.build_from_path(in_dir, out_dir, num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[2] for m in metadata])
    sr = hparams.sample_rate
    hours = frames / sr / 3600
    print('Wrote %d utterances, %d time steps (%.2f hours)' % (len(metadata), frames, hours))
    print('Max input length:  %d' % max(len(m[3]) for m in metadata))
    print('Max target length (timesteps): %d' % max(m[2] for m in metadata))
    
if __name__ == "__main__":
    args = docopt(__doc__)
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]
    num_workers = args["--num_workers"]
    num_workers = cpu_count() if num_workers is None else int(num_workers)

    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "wavenet_vocoder"

    # Presets
    if hparams.preset is not None and hparams.preset != "":
        preset = hparams.presets[hparams.preset]
        import json
        hparams.parse_json(json.dumps(preset))
        print("Override hyper parameters with preset \"{}\": {}".format(
            hparams.preset, json.dumps(preset, indent=4)))

    print("Sampling frequency: {}".format(hparams.sample_rate))

    assert hparams.modal in ["se", "vc", "tts"]
    mod = importlib.import_module("features")
    preprocess(mod, in_dir, out_dir, num_workers)
