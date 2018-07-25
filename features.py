from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import audio

from nnmnkwii.io import hts
from nnmnkwii import preprocessing as P
from os.path import exists, join
import librosa
from string import punctuation
from tqdm import tqdm

from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw

from hparams import hparams

# These lists define the set of speakers to use for experiments.
# User is free to change SRC_SPKS and/or TARGET_SPKS. 
# Both of them are taken from the train set, whereas the TEST_SPKS
# are taken separately according to the database.
# TODO: change pipeline to include all speakers automatically

SRC_SPKS = ['p226', 'p227']
TARGET_SPKS = ['p228', 'p230', 'p231', 'p233', 'p287', 'p282', 'p278', 'p277']
TEST_SPKS = ['p232', 'p257']

SPKS = SRC_SPKS + TARGET_SPKS + TEST_SPKS
TARGET_ID = [SPKS.index(spk) for spk in TARGET_SPKS]
TEST_ID = [SPKS.index(spk) for spk in TEST_SPKS]


def _rm_hidden(files):
		return [file for file in files if not file.startswith(".")]


def _train_first(dirs):
		# Just for convenience, but absolutely unnecessary.
		return (dirs[1], dirs[0]) if 'train' in dirs[1] else dirs


def _find_interest_dirs(path, task):
	interest_dirs = list()
	try:
		for d in _rm_hidden(os.listdir(path)):
			if task == "se" and 'txt' not in d and 'noisy' in d:
				# Keep noisy only
				interest_dirs.append(d)
			elif task == "vc" and "noisy" not in d and 'txt' in d:
				# Keep txt only
				interest_dirs.append(d)
	except:
		raise OSError('Incomplete dataset')
	return interest_dirs


def _dtw(mel_src, mel_target):
	mel_src = np.swapaxes(mel_src, 0, 1)
	mel_target = np.swapaxes(mel_target, 0, 1)
	_, wp = librosa.core.dtw(mel_src, mel_target)
	new_src = np.zeros(mel_target.shape)
	n_frames = mel_target.shape[1]
	last_frame = 0
	for i in range(n_frames):
		if i in wp[:,1]:
			idx_wp, = np.where(wp[:,1] == i)
			avg_frame = np.mean(mel_src[:,wp[idx_wp,0]], axis=1)
			new_src[:,i] = avg_frame
			last_frame = wp[max(idx_wp), 0]
		else:
			if i == 0:
				new_src[:,0] = mel_src[:,0]
			else:
				new_src[:,i] = mel_src[:,last_frame]
	new_src = np.swapaxes(new_src, 0, 1)
	return new_src


def _se_metadata(in_dir, name):
	name = join(in_dir, name)
	dirs = _train_first(_find_interest_dirs(in_dir, "se"))
	info = list()

	def _get_txt_path(wav_file, is_train=True):
		if is_train:
			return join(in_dir, 'trainset_28spk_txt', file.replace('wav', 'txt'))
		else:
			return join(in_dir, 'testset_txt', file.replace('wav', 'txt'))

	def _get_clean(noisy_path):
		return noisy_path.replace('noisy', 'clean')

	for d in dirs:
		dir_path = join(in_dir, d)
		files = _rm_hidden(os.listdir(dir_path))
		for file in files:
			speaker = file.split("_")[0]
			if speaker not in SRC_SPKS and speaker in SPKS:
				speaker_id = str(SPKS.index(speaker))
				is_train = speaker in TARGET_SPKS
				txt_path = _get_txt_path(file, is_train)

				with open(txt_path, 'r', encoding='utf-8') as f:
					text = f.read()[:-1]
				src_path = join(in_dir, d, file)
				target_path = _get_clean(src_path)

				info.append((src_path, target_path, text, speaker_id))

	with open(name, 'w', encoding='utf-8') as f:
		for l in info:
			f.write(l[0] + '|' + l[1] + '|' + l[2] + '|' + l[3] + '\n' )


def _vc_metadata(in_dir, name):
	name = join(in_dir, name)
	dirs = _train_first(_find_interest_dirs(in_dir, "vc"))
	info = list()

	def _collect_target(path):
		all_files = _rm_hidden(os.listdir(path))
		target_utts = dict()
		for file in all_files:
			speaker = file.split("_")[0]
			if speaker in TARGET_SPKS:
				if speaker not in target_utts:
					target_utts[speaker] = list()
					target_utts[speaker].append(file)
				else:
					target_utts[speaker].append(file)
		return target_utts

	def _rm_spaces(string):
		return string.replace(' ','').replace('\n','')

	def _rm_punctuation(string):
		string = _rm_spaces(string)
		string = [s for s in string if s not in punctuation]
		return ''.join(string).lower()

	def _read_file(path):
		with open(path, 'r', encoding='utf-8') as f:
			text = f.read()[:-1]
			txt = _rm_punctuation(text)
		return text, txt

	def _get_audio(text_path, stage):
		s = 'clean_' + stage
		path = text_path.replace(stage, s)
		path = path.replace('txt', 'wav')
		return path

	target_utts = _collect_target(join(in_dir, dirs[0]))

	for d in dirs:
		dir_path = join(in_dir, d)
		files = _rm_hidden(os.listdir(dir_path))
		for file in files:
			speaker = file.split("_")[0]
			if speaker in SRC_SPKS or speaker in TEST_SPKS:
				text, ref = _read_file(join(dir_path, file))
				for spk in target_utts:
					speaker_id = str(SPKS.index(spk))
					for utt in target_utts[spk]:
						_, cand = _read_file(join(in_dir, dirs[0], utt))
						if ref == cand:
							src_path = _get_audio(join(in_dir, d, file), d)
							target_path = _get_audio(join(in_dir, dirs[0], utt), dirs[0])
							info.append((src_path, target_path, text, speaker_id))

	with open(name, 'w', encoding='utf-8') as f:
		for l in info:
			f.write(l[0] + '|' + l[1] + '|' + l[2] + '|' + l[3] + '\n' )


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
	executor = ProcessPoolExecutor(max_workers=num_workers)
	futures = []
	index = 1

	metafile = 'metadata_' + hparams.modality + '.csv'
	print('Preparing metadata file %s' % metafile)
	if hparams.modality == "se":
		_se_metadata(in_dir, metafile)
	elif hparams.modality == "vc":
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
	

def _extract_mel(wav_path):
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
	audio_src, mel_src, timesteps_src, dtype_src = _extract_mel(path_src)
	_, mel_target, timesteps_target, dtype_target = _extract_mel(
		path_target)

	if hparams.modality == "vc":
		mel_src = _dtw(mel_src, mel_target)

	# Write files to disk
	if hparams.modality == "se":
		if int(speaker) in TEST_ID:
			audio_filename = "source-audio-test-%05d.npy" % index
			melSpec_filename = "target-mel-test-%05d.npy" % index
		else:
			audio_filename = "source-audio-%05d.npy" % index
			melSpec_filename = "target-mel-%05d.npy" % index

	if hparams.modality == "vc":
		if TEST_SPKS[0] in path_src or TEST_SPKS[1] in path_src:
			audio_filename = "source-audio-test-%05d.npy" % index
			melSpec_filename = "target-mel-test-%05d.npy" % index
		else:
			audio_filename = "source-audio-%05d.npy" % index
			melSpec_filename = "target-mel-%05d.npy" % index

	np.save(join(out_dir, audio_filename),
		audio_src.astype(dtype_src), allow_pickle=False)
	np.save(join(out_dir, melSpec_filename),
		mel_target.astype(np.float32), allow_pickle=False)
	

	return (audio_filename, melSpec_filename, 
		timesteps_target, text, speaker)

