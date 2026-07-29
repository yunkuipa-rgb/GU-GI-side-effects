import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):

	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size, vote_size):

		super(LSTMClassifier, self).__init__()

		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.num_layers = 3
		self.D = 1
		self.b = self.D * self.num_layers

		self.embedding = nn.Linear(vocab_size, embedding_dim)
		self.lstm = [nn.LSTM(embedding_dim, hidden_dim, num_layers=self.num_layers, batch_first=True) for i in range(vote_size)]

		self.hidden2out = nn.Linear(hidden_dim, output_size)
		self.softmax = nn.Sigmoid()

		self.dropout_layer = nn.Dropout(p=0.2)
		self.vote1 = nn.Linear(vote_size, vote_size)
		self.act = nn.ReLU()
		self.vote2 = nn.Linear(vote_size, output_size)
		self.vote_size = vote_size


	def forward(self, x):
		batch = x.shape[0]
		res = []
		for i in range(self.vote_size):
			embeds = self.embedding(x[:, i, ...])
			h0, c0 = torch.randn(self.b, batch, self.hidden_dim), torch.randn(self.b, batch, self.hidden_dim)
			hidden = (h0.to(x.device), c0.to(x.device))
			m = self.lstm[i].to(x.device)
			_, (ht, ct) = m(embeds, hidden)
			output = self.dropout_layer(ht[-1])
			output = self.hidden2out(output)
			output = self.softmax(output)
			res.append(output)
		res = torch.cat(res, dim=-1)
		out = self.vote1(res)
		out = self.act(out)
		out = self.vote2(res)
		out = self.softmax(out)

		return out, res
	

class LSTMClassifier2(nn.Module):

	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size):

		super(LSTMClassifier2, self).__init__()

		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.num_layers = 4 # original 4
		self.D = 1
		self.b = self.D * self.num_layers

		self.embedding = nn.Linear(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=self.num_layers, batch_first=True)

		self.dropout_layer = nn.Dropout(p=0.1)
		self.hidden2out = nn.Linear(hidden_dim, hidden_dim // 2)
		self.act = nn.ReLU()
		self.hidden2out2 = nn.Linear(hidden_dim // 2, output_size)
		self.softmax = nn.Sigmoid()


	def forward(self, x):
		batch = x.shape[0]
		embeds = self.embedding(x)
		h0, c0 = torch.randn(self.b, batch, self.hidden_dim), torch.randn(self.b, batch, self.hidden_dim)
		hidden = (h0.to(x.device), c0.to(x.device))
		m = self.lstm.to(x.device)
		_, (ht, ct) = m(embeds, hidden)
		output = self.dropout_layer(ht[-1])
		output = self.hidden2out(output)
		output = self.act(output)
		output = self.hidden2out2(output)
		output = self.softmax(output)

		return output