from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import audio

from nnmnkwii.io import hts
from nnmnkwii import preprocessing as P
from hparams import hparams
from os.path import exists, join
import librosa
from string import punctuation
from tqdm import tqdm

from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw

from hparams import hparams

src_spks = ['p226', 'p227']
target_speakers = ['p287', 'p282', 'p278', 'p277']
test_spks = ['p232', 'p257']
SPKS = target_speakers + test_spks
test_id = [SPKS.index(spk) for spk in test_spks]

def _dtw(melSpec_src, melSpec_target):
	src = np.swapaxes(melSpec_src, 0, 1)
	dst = np.swapaxes(melSpec_target, 0, 1)

	_, wp = librosa.core.dtw(src, dst)

	indices = list()
	tgt_wp = list()

	for al in wp:
		if al[1] not in tgt_wp:
			tgt_wp.append(al[1])
			indices.append((al[0]))

	new_src = np.take(src, indices[::-1], axis=1)
	
	if new_src.shape[1] < dst.shape[1]:
		new_src.append(new_src[-1])

	return new_src

def _se_metadata(in_dir, name):
	"""Create metada.csv file for speech enhancement task."""
	subdir = ('noisy', 'clean')

	def _rm_hidden(files):
		return [file for file in files if not file.startswith(".")]

	def _point_txt(wav_path):
		return wav_path.replace('wav', 'txt')

	with open(join(in_dir, name), 'w', encoding='utf-8') as f:
		for spk in SPKS:
			stage = 'train' if spk in target_speakers else 'test'
			spk_src_path = join(
				in_dir, stage, subdir[0], spk, 'wav')
			spk_src_files = _rm_hidden(os.listdir(spk_src_path))
			spk_id = str(SPKS.index(spk))
			for src in spk_src_files:
				src_path = join(spk_src_path, src)
				target_path = src_path.replace(subdir[0], subdir[1])
				txt_path = _point_txt(src_path)
				txt = open(txt_path, 'r', encoding='utf-8')
				text = txt.read()[:-1]
				f.write(src_path + '|' + target_path + '|' + 
					text + '|' + spk_id + '\n')


def _vc_metadata(in_dir, name):
	"""Create metadata.csv file for Voice Conversion task."""
	phase = ('train', 'test')

	def _rm_hidden(files):
		return [file for file in files if not file.startswith(".")]

	def _point_wav(txt_path):
		return txt_path.replace('txt', 'wav')

	def _rm_spaces(string):
		return string.replace(' ','').replace('\n','')

	def _rm_punctuation(string):
		string = _rm_spaces(string)
		string = [s for s in string if s not in punctuation]
		return ''.join(string).lower()


	def collect_sentences(spk_path):
		refs = list()
		for file in _rm_hidden(os.listdir(spk_path)):
			with open(join(spk_path, file), 'r', encoding='utf-8') as f:
				txt = _rm_punctuation(f.read())
				refs.append(txt)
		return refs


	def make_sentence_dict(speakers):
		sentences = dict()
		for spk in speakers:
			phase = 'train' if spk not in test_spks else 'test'
			spk_path = join(in_dir, phase, 'clean', spk, 'txt')
			files = _rm_hidden(os.listdir(spk_path))
			for file in files:
				file_path = join(spk_path, file)
				with open(file_path, 'r', encoding='utf-8') as f:
					txt = _rm_punctuation(f.read())
					if txt not in sentences:
						sentences[txt] = 1
					else:
						sentences[txt] += 1
		return sentences

	src_spks = _rm_hidden(os.listdir(join(in_dir, 'train', 'clean')))
	src_spks = [spk for spk in src_spks if spk not in target_speakers]
	speakers = src_spks + SPKS
	sentences = make_sentence_dict(speakers)
	num_speakers = len(speakers)
	print([key for key in sentences.keys() if sentences[key] == num_speakers])
	exit()
	

def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
	executor = ProcessPoolExecutor(max_workers=num_workers)
	futures = []
	index = 1

	# TODO: metadata.csv creation
	metafile = 'metadata_' + hparams.modal + '.csv'
	if hparams.modal == "se":
		_se_metadata(in_dir, metafile)
	elif hparams.modal == "vc":
		_vc_metadata(in_dir, metafile)

	with open(join(in_dir, metafile), 'r', encoding='utf-8') as f:
		for line in f:
			parts = line.strip().split('|')
			path_src = parts[0]
			path_target = parts[1]
			text = parts[2]
			spk = parts[-1]
			futures.append(executor.submit(
				partial(_process_utterance, out_dir,
					index, path_src, path_target, text, spk)))
			index += 1
	return [future.result() for future in tqdm(futures)]


def _extract_melSpec(wav_path):
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


def _process_utterance(out_dir, index, path_src, 
	path_target, text, speaker):
	sr = hparams.sample_rate

	_, mel_src, timesteps_src, dtype_src = _extract_melSpec(
		path_src)
	audio_target, mel_target, timesteps_target, dtype_target = _extract_melSpec(
		path_target)

	if hparams.modal == "vc":
		mel_src = _dtw(mel_src, mel_target)

	# Write files to disk
	if int(speaker) in test_id:
		melSpec_filename = "source-melSpec-test-%05d.npy" % index
		audio_filename = "target-audio-test-%05d.npy" % index
	else:
		melSpec_filename = "source-melSpec-%05d.npy" % index
		audio_filename = "target-audio-%05d.npy" % index

	np.save(join(out_dir, melSpec_filename),
		mel_src.astype(np.float32), allow_pickle=False)
	np.save(join(out_dir, audio_filename),
		audio_target.astype(dtype_target), allow_pickle=False)

	return (audio_filename, melSpec_filename, 
		timesteps_target, text, speaker)