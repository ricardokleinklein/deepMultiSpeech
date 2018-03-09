import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

sample = Variable(torch.rand(1,1,28000))


class prepare(nn.Module):

	def __init__(self, up_rate, cin_channels):
		super(prepare, self).__init__()

		layers = int(cin_channels/up_rate)
		self.first = nn.Conv1d(1, up_rate, 3, padding=1)
		self.upsampling = nn.ModuleList(
			[nn.Conv1d(10*i, 10*(i+1), 3, padding=1) for i in range(1, layers)])

	def forward(self, inputs):
		h = self.first(inputs)
		for layer in self.upsampling:
			h = layer(h)
			h = F.relu(h)
		return h

model = prepare(10, 80)

out = model(sample)

print(type(out))