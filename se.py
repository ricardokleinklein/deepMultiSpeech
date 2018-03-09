from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import audio

from nnmnkwii.io import hts
from nnmnkwii import preprocessing as P
from hparams import hparams
from os.path import exists
import librosa

from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw

from hparams import hparams

def _maybe_metadata(in_dir):
	"""Create metadata for SE files if missed."""
	# Check if metadata with data paths.
	# Tree of the data root dir:
	# - root
	#		- train (clean | noisy)
	#		- test (clean | noisy)
	if os.path.isfile(os.path.join(in_dir, "metadata.csv")):
		return

	# metadata.csv collects files' paths.
	subdirs = ("clean", "noisy")
	files_target = os.listdir(os.path.join(
		in_dir, subdirs[0]))
	files_source = os.listdir(os.path.join(
		in_dir, subdirs[1]))
	assert len(files_source) == len(files_target)

	with open(os.path.join(in_dir, "metadata.csv"), 'w', encoding='utf-8') as f:
		for src, target in zip(files_source, files_target):
			src_path = os.path.join(in_dir, subdirs[1], src)
			target_path = os.path.join(in_dir, subdirs[0], target)
			f.write(src_path + "|" + target_path + "\n")

def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
	executor = ProcessPoolExecutor(max_workers=num_workers)
	futures = []
	index = 1

	_maybe_metadata(in_dir)

	with open(os.path.join(in_dir, "metadata.csv"), 'r', encoding='utf-8') as f:
		for line in f:
			parts = line.strip().split('|')
			path_source = parts[0]
			path_target = parts[1]
			futures.append(executor.submit(
				partial(_process_utterance, out_dir, index, path_source, path_target)))
			index += 1
	return [future.result() for future in tqdm(futures)]

def _extract_mel_spectrogram(wav_path):
	# Load the audio to a numpy array. Resampled if needed.
	wav = audio.load_wav(wav_path)

	if hparams.rescaling:
		wav = wav / np.abs(wav).max() * hparams.rescaling_max

	# Mu-law quantize
	if is_mulaw_quantize(hparams.input_type):
		# [0, quantize_channels)
		out = P.mulaw_quantize(wav, hparams.quantize_channels)

		# Trim silences
		start, end = audio.start_and_end_indices(out, hparams.silence_threshold)
		wav = wav[start:end]
		out = out[start:end]
		constant_values = P.mulaw_quantize(0, hparams.quantize_channels)
		out_dtype = np.int16
	elif is_mulaw(hparams.input_type):
		# [-1, 1]
		out = P.mulaw(wav, hparams.quantize_channels)
		constant_values = P.mulaw(0.0, hparams.quantize_channels)
		out_dtype = np.float32
	else:
		# [-1, 1]
		out = wav
		constant_values = 0.0
		out_dtype = np.float32

	# Compute a mel-scale spectrogram from the trimmed wav:
	# (N, D)
	mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T
	# lws pads zeros internally before performing stft
	# this is needed to adjast time resolution between audio and mel-spectrogram
	l, r = audio.lws_pad_lr(wav, hparams.fft_size, audio.get_hop_size())

	# zero pad for quantized signal
	out = np.pad(out, (l, r), mode="constant", constant_values=constant_values)
	N = mel_spectrogram.shape[0]
	assert len(out) >= N * audio.get_hop_size()

	# time resolution adjastment
	# ensure length of raw audio is multiple of hop_size so that we can use
	# transposed convolution to upsample
	out = out[:N * audio.get_hop_size()]
	assert len(out) % audio.get_hop_size() == 0
	assert len(out) // N == audio.get_hop_size()

	timesteps = len(out)

	return out, mel_spectrogram, timesteps, out_dtype

def _process_utterance(out_dir, index, path_source, path_target):
	sr = hparams.sample_rate

	audio_input, _, timesteps_input, dtype_input = _extract_mel_spectrogram(
		path_source)
	audio_target, _, timesteps_target, dtype_target = _extract_mel_spectrogram(
		path_target)

	# Write files to disk.
	input_filename = "source-audio-%05d.npy" % index
	target_filename = "target-audio-%05d.npy" % index
	melspec_filename = "melspec-%05d.npy" % index

	np.save(os.path.join(out_dir, input_filename),
		audio_input.astype(dtype_input), allow_pickle=False)
	np.save(os.path.join(out_dir, target_filename),
		audio_target.astype(dtype_target), allow_pickle=False)
	# np.save(os.path.join(out_dir, melspec_filename),
		# mel_target.astype(np.float32), allow_pickle=False)

	return (input_filename, target_filename, 
		timesteps_input, timesteps_target)