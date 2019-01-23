import random
import numpy as np
import os

import torch
from torch.autograd import Variable

class DataUtil(object):

  def __init__(self, hparams, decode=True):
    self.hparams = hparams
    self.src_i2w_list = []
    self.src_w2i_list = []
    
    i2w, w2i = self._build_vocab_list(self.hparams.src_vocab_list, max_vocab_size=self.hparams.src_vocab_size)
    for i in range(len(self.hparams.src_vocab_list)):
      self.src_i2w_list.append(i2w)
      self.src_w2i_list.append(w2i)
    self.hparams.src_vocab_size = len(i2w)
    #for i, v_file in enumerate(hparams.src_vocab_list):
    #  #v_file = os.path.join(self.hparams.data_path, v_file)
    #  i2w, w2i = self._build_vocab(v_file, max_vocab_size=self.hparams.src_vocab_size)   
    #  self.src_i2w_list.append(i2w)
    #  self.src_w2i_list.append(w2i)
    #  if i == 0:
    #    self.hparams.src_vocab_size = len(i2w)
    #    print("setting src_vocab_size to {}...".format(self.hparams.src_vocab_size))
    if self.hparams.lan_code_rl:
      num_extra_lan = len(self.hparams.train_src_file_list) - 1
      self.lan_code_list = [self.hparams.src_vocab_size+i for i in range(num_extra_lan)]
      self.hparams.src_vocab_size += num_extra_lan
    self.trg_i2w_list = []
    self.trg_w2i_list = []
    for i, v_file in enumerate(hparams.trg_vocab_list):
      #v_file = os.path.join(self.hparams.data_path, v_file)
      i2w, w2i = self._build_vocab(v_file, max_vocab_size=self.hparams.trg_vocab_size)   
      self.trg_i2w_list.append(i2w)
      self.trg_w2i_list.append(w2i)
      if i == 0:
        self.hparams.trg_vocab_size = len(i2w)
        print("setting trg_vocab_size to {}...".format(self.hparams.trg_vocab_size))
    while len(self.trg_i2w_list) < len(self.src_i2w_list):
      self.trg_i2w_list.append(self.trg_i2w_list[-1])
      self.trg_w2i_list.append(self.trg_w2i_list[-1])

    if hasattr(self.hparams, 'uni') and self.hparams.uni:
      self.pretrained_src_emb_list = []
      self.pretrained_trg_emb = None
      self.src_w2i_list = []
      self.src_i2w_list = []
      for i, emb_file in enumerate(self.hparams.pretrained_src_emb_list):
        emb, i2w, w2i = self.load_pretrained(emb_file)
        self.pretrained_src_emb_list.append(emb)
        self.src_i2w_list.append(i2w)
        self.src_w2i_list.append(w2i)
      self.pretrained_trg_emb, _, _ = self.load_pretrained(self.hparams.pretrained_trg_emb)
      
    if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram or self.hparams.char_input is not None:
      self.src_char_i2w, self.src_char_w2i = self._build_char_vocab_from(self.hparams.src_char_vocab_from, self.hparams.src_char_vocab_size, n=self.hparams.n, single_n=self.hparams.single_n)
      self.trg_char_i2w, self.trg_char_w2i = self._build_char_vocab_from(self.hparams.trg_char_vocab_from, self.hparams.trg_char_vocab_size, n=self.hparams.n, single_n=self.hparams.single_n)
      self.src_char_vsize, self.trg_char_vsize = len(self.src_char_i2w), len(self.trg_char_i2w)

      #print(self.src_char_i2w)
      #print(self.src_char_w2i)
      #print(self.trg_char_i2w)
      #print(self.trg_char_w2i)

      setattr(self.hparams, 'src_char_vsize', self.src_char_vsize)
      setattr(self.hparams, 'trg_char_vsize', self.trg_char_vsize)
      print("src_char_vsize={} trg_char_vsize={}".format(self.src_char_vsize, self.trg_char_vsize))
    else:
      self.src_char_vsize, self.trg_char_vsize = None, None
      setattr(self.hparams, 'src_char_vsize', None)
      setattr(self.hparams, 'trg_char_vsize', None)
    if not self.hparams.decode:
      assert len(self.src_i2w_list) >= len(self.hparams.train_src_file_list)
      assert len(self.trg_i2w_list) >= len(self.hparams.train_trg_file_list)
      self.train_x = []
      self.train_y = []

      self.train_x_char_kv = []
      self.train_y_char_kv = []
      i, self.train_size = 0, []
      self.n_train_batches = []

      train_x_lens = []
      self.file_idx = []
      for s_file,t_file in zip(self.hparams.train_src_file_list, self.hparams.train_trg_file_list):
        if s_file and t_file:
          train_x, train_y, x_char_kv, y_char_kv, src_len = self._build_parallel(s_file, t_file, i)
        else:
          train_x, train_y, x_char_kv, y_char_kv, src_len = [], [], [], [], []
        if self.hparams.shuffle_train:
          self.train_x.extend(train_x)
          self.train_y.extend(train_y)
        else:
          self.train_x.append(train_x)
          self.train_y.append(train_y)
        if not x_char_kv is None:
          if self.hparams.shuffle_train:
            self.train_x_char_kv.extend(x_char_kv)
            self.train_y_char_kv.extend(y_char_kv)
          else:
            self.train_x_char_kv.append(x_char_kv)
            self.train_y_char_kv.append(y_char_kv)
        if self.hparams.shuffle_train:
          self.file_idx.extend([i for _ in range(len(train_x))])
        else:
          self.file_idx.append([i for _ in range(len(train_x))])

        i += 1
        self.train_size.append(len(train_x))
        train_x_lens.extend(src_len)
      dev_src_file = self.hparams.dev_src_file
      dev_trg_file = self.hparams.dev_trg_file
      self.dev_x, self.dev_y, self.dev_x_char_kv, self.dev_y_char_kv, src_len = self._build_parallel(dev_src_file, dev_trg_file, 0, is_train=False)
      self.dev_size = len(self.dev_x)
      self.dev_index = 0
      #self.dev_x_char, self.dev_y_char = self.get_trans_char(self.dev_x_char_kv), self.get_trans_char(self.dev_y_char_kv)
      if self.hparams.shuffle_train:
        print("Heuristic sort based on source lengths")
        indices = np.argsort(train_x_lens)
        self.train_x = [[self.train_x[idx] for idx in indices]]
        self.train_y = [[self.train_y[idx] for idx in indices]]
        self.train_x_char_kv = [[self.train_x_char_kv[idx] for idx in indices]]
        self.train_y_char_kv = [[self.train_y_char_kv[idx] for idx in indices]]
        self.file_idx = [[self.file_idx[idx] for idx in indices]]
        self.train_size = [sum(self.train_size)]
      self.reset_train()
    else:
      #test_src_file = os.path.join(self.hparams.data_path, self.hparams.test_src_file)
      #test_trg_file = os.path.join(self.hparams.data_path, self.hparams.test_trg_file)
      test_src_file = self.hparams.test_src_file
      test_trg_file = self.hparams.test_trg_file
      self.test_x, self.test_y, self.test_x_char_kv, self.test_y_char_kv, src_len = self._build_parallel(test_src_file, test_trg_file, 0, is_train=False)
      self.test_size = len(self.test_x)
      self.test_index = 0
      if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram or self.hparams.char_input is not None:
        self.test_x_char = self.get_trans_char(self.test_x_char_kv, self.src_char_vsize)
        self.test_y_char = self.get_trans_char(self.test_y_char_kv, self.trg_char_vsize)
      else:
        self.test_x_char, self.test_y_char = None, None
  
  def load_pretrained(self, pretrained_emb_file):
    f = open(pretrained_emb_file, 'r', encoding='utf-8')
    header = f.readline().split(' ')
    count = int(header[0])
    dim = int(header[1])
    #matrix = np.zeros((len(w2i), dim), dtype=np.float32)
    matrix = np.zeros((count, dim), dtype=np.float32)
    #i2w = ['<pad>', '<unk>', '<s>', '<\s>']
    i2w = []
    #w2i = {'<pad>': 0, '<unk>':1, '<s>':2, '<\s>':3}
    w2i = {}

    for i in range(count):
      word, vec = f.readline().split(' ', 1)
      w2i[word] = len(w2i)
      i2w.append(word)
      matrix[i] = np.fromstring(vec, sep=' ', dtype=np.float32)
      #if not word in w2i:
      #  print("{} no in vocab".format(word))
      #  continue
      #matrix[w2i[word]] = np.fromstring(vec, sep=' ', dtype=np.float32) 
    return torch.FloatTensor(matrix), i2w, w2i

  def get_trans_char(self, char_raw, char_vsize):
    ret_char = []
    if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram:
      for kvs in char_raw:
        key, val = [], []
        sent_sparse = []
        for i, kv in enumerate(kvs):
          key.append(torch.LongTensor([[i for _ in range(len(kv.keys()))], list(kv.keys())]))
          val.extend(list(kv.values()))
        key = torch.cat(key, dim=1)
        val = torch.FloatTensor(val)
        sent_sparse = torch.sparse.FloatTensor(key, val, torch.Size([len(kvs), char_vsize]))
        # (batch_size, max_len, char_dim)
        ret_char.append([sent_sparse])
    elif self.hparams.char_input is not None:
      #max_char_len = max([len(w) for sent in char_raw for w in sent])
      for char_sent in char_raw:
        max_char_len = max([len(w) for w in char_sent])
        padded_char_sent = [s + ([self.hparams.pad_id]*(max_char_len-len(s))) for s in char_sent]
        #padded_char_sent += [[pad_id]*max_char_len] * (max_len - len(padded_char_sent))
        #padded_char_sents.append(padded_char_sent)
        # (batch_size, max_len, max_char_len, char_dim)
        #print(char_sent)
        char_sent = Variable(torch.LongTensor(padded_char_sent).unsqueeze(0))
        if self.hparams.cuda: char_sent = char_sent.cuda()
        ret_char.append(char_sent)
     
    return ret_char

  def get_char_emb(self, word_idx, is_trg=True):
    if is_trg:
      w2i, i2w, vsize = self.trg_char_w2i, self.trg_char_i2w, self.hparams.trg_char_vsize
      word = self.trg_i2w_list[0][word_idx]
    else:
      w2i, i2w, vsize = self.src_char_w2i, self.src_char_i2w, self.hparams.src_char_vsize
      word = self.src_i2w_list[0][word_idx]
    if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram:
      if word_idx == self.hparams.bos_id or word_idx == self.hparams.eos_id:
        kv = {0:0}
      elif self.hparams.char_ngram_n:
        kv = self._get_ngram_counts(word, i2w, w2i, self.hparams.char_ngram_n)
      elif self.hparams.bpe_ngram:
        kv = self._get_bpe_ngram_counts(word, i2w, w2i)
      key = torch.LongTensor([[0 for _ in range(len(kv.keys()))], list(kv.keys())])
      val = torch.FloatTensor(list(kv.values()))
      ret = [torch.sparse.FloatTensor(key, val, torch.Size([1, vsize]))]
    elif self.hparams.char_input is not None:
      ret = self._get_char(word, i2w, w2i, n=self.hparams.n)
      ret = Variable(torch.LongTensor(ret).unsqueeze(0).unsqueeze(0))
      if self.hparams.cuda: ret = ret.cuda()
    return ret

  def reset_train(self):
    if self.hparams.batcher == "word":
      if not self.n_train_batches:
        self.start_indices, self.end_indices = [], []
        for i, train_size in enumerate(self.train_size):
          start_indices, end_indices = [], []
          start_index = 0
          while start_index < train_size:
            end_index = start_index
            word_count = 0
            while (end_index + 1 < train_size and word_count + len(self.train_x[i][end_index]) + len(self.train_y[i][end_index]) <= self.hparams.batch_size):
              end_index += 1
              word_count += (len(self.train_x[i][end_index]) + len(self.train_y[i][end_index]))
            start_indices.append(start_index)
            end_indices.append(end_index+1)
            start_index = end_index + 1
            #print(start_indices[-1], end_indices[-1])
          assert len(start_indices) == len(end_indices)
          self.n_train_batches.append(len(start_indices))
          self.start_indices.append(start_indices)
          self.end_indices.append(end_indices)
    elif self.hparams.batcher == "sent":
      if not self.n_train_batches:
        for train_size in self.train_size:
          self.n_train_batches.append((train_size + self.hparams.batch_size - 1) // self.hparams.batch_size)
    else:
      print("unknown batcher")
      exit(1)
    self.train_queue = []
    for n_train_batches in self.n_train_batches:
      self.train_queue.append(np.random.permutation(n_train_batches))
    self.train_index = 0
    self.train_data_index = 0

  def next_train(self):
    data_idx = self.train_data_index
    if self.hparams.batcher == "word":
      start_index = self.start_indices[data_idx][self.train_queue[data_idx][self.train_index]]
      end_index = self.end_indices[data_idx][self.train_queue[data_idx][self.train_index]]
    elif self.hparams.batcher == "sent":
      start_index = (self.train_queue[data_idx][self.train_index] * self.hparams.batch_size)
      end_index = min(start_index + self.hparams.batch_size, self.train_size[data_idx])
    else:
      print("unknown batcher")
      exit(1)

    x_train = self.train_x[data_idx][start_index:end_index]
    y_train = self.train_y[data_idx][start_index:end_index]
    train_file_index = self.file_idx[data_idx][start_index:end_index]
    if self.hparams.sample_rl and data_idx != 0:
      x_train_sample = []
      for x_t in x_train:
        x_train_sample.append([x_t[0]] + [random.randint(3, self.hparams.src_vocab_size-1) for i in range(len(x_t)-2)] + [x_t[-1]])
      x_train = x_train_sample
    if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram or self.hparams.char_input is not None:
      x_train_char_kv = self.train_x_char_kv[data_idx][start_index:end_index]
      y_train_char_kv = self.train_y_char_kv[data_idx][start_index:end_index]
      x_train, y_train, x_train_char_kv, y_train_char_kv, train_file_index = self.sort_by_xlen(x_train, y_train, x_train_char_kv, y_train_char_kv, train_file_index)
    else:
      x_train, y_train, train_file_index = self.sort_by_xlen(x_train, y_train,file_index=train_file_index)

    self.train_index += 1
    batch_size = len(x_train)
    y_count = sum([len(y) for y in y_train])
    while self.train_index >= self.n_train_batches[self.train_data_index]:
      self.train_data_index += 1
      self.train_index = 0
      if self.train_data_index >= len(self.train_x): break
    # pad 
    if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram:
      x_train, x_mask, x_count, x_len, x_pos_emb_idxs, x_train_char = self._pad(x_train, self.hparams.pad_id, x_train_char_kv, self.hparams.src_char_vsize)
      y_train, y_mask, y_count, y_len, y_pos_emb_idxs, y_train_char = self._pad(y_train, self.hparams.pad_id, y_train_char_kv, self.hparams.trg_char_vsize)
    elif self.hparams.char_input is not None:
      x_train, x_mask, x_count, x_len, x_pos_emb_idxs, x_train_char = self._pad(x_train, self.hparams.pad_id, char_sents=x_train_char_kv)
      y_train, y_mask, y_count, y_len, y_pos_emb_idxs, y_train_char = self._pad(y_train, self.hparams.pad_id, char_sents=y_train_char_kv)
    else:
      x_train_char, y_train_char = None, None
      x_train, x_mask, x_count, x_len, x_pos_emb_idxs = self._pad(x_train, self.hparams.pad_id)
      y_train, y_mask, y_count, y_len, y_pos_emb_idxs = self._pad(y_train, self.hparams.pad_id)

    if self.train_data_index >= len(self.train_x):
      self.reset_train()
      eop = True
    else:
      eop = False
    #print(x_train)
    #print(x_mask)
    #print(y_train)
    #print(y_mask)
    #exit(0)
    return x_train, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_train_char, y_train_char, eop, train_file_index 

  def next_dev(self, dev_batch_size=1):
    start_index = self.dev_index
    end_index = min(start_index + dev_batch_size, self.dev_size)
    batch_size = end_index - start_index

    x_dev = self.dev_x[start_index:end_index]
    y_dev = self.dev_y[start_index:end_index]
    if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram or self.hparams.char_input is not None:
      x_dev_char_kv = self.dev_x_char_kv[start_index:end_index]
      y_dev_char_kv = self.dev_y_char_kv[start_index:end_index]
      x_dev, y_dev, x_dev_char_kv, y_dev_char_kv = self.sort_by_xlen(x_dev, y_dev, x_dev_char_kv, y_dev_char_kv)
    else:
      #pass
      x_dev, y_dev = self.sort_by_xlen(x_dev, y_dev)

    if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram:
      x_dev, x_mask, x_count, x_len, x_pos_emb_idxs, x_dev_char_sparse = self._pad(x_dev, self.hparams.pad_id, x_dev_char_kv, self.hparams.src_char_vsize)
      y_dev, y_mask, y_count, y_len, y_pos_emb_idxs, y_dev_char_sparse = self._pad(y_dev, self.hparams.pad_id, y_dev_char_kv, self.hparams.trg_char_vsize)
    elif self.hparams.char_input is not None:
      x_dev, x_mask, x_count, x_len, x_pos_emb_idxs, x_dev_char_sparse = self._pad(x_dev, self.hparams.pad_id, char_sents=x_dev_char_kv)
      y_dev, y_mask, y_count, y_len, y_pos_emb_idxs, y_dev_char_sparse = self._pad(y_dev, self.hparams.pad_id, char_sents=y_dev_char_kv)
    else:
      x_dev_char_sparse, y_dev_char_sparse = None, None
      x_dev, x_mask, x_count, x_len, x_pos_emb_idxs = self._pad(x_dev, self.hparams.pad_id)
      y_dev, y_mask, y_count, y_len, y_pos_emb_idxs = self._pad(y_dev, self.hparams.pad_id)

    if end_index >= self.dev_size:
      eop = True
      self.dev_index = 0
    else:
      eop = False
      self.dev_index += batch_size

    return x_dev, x_mask, x_count, x_len, x_pos_emb_idxs, y_dev, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, eop, x_dev_char_sparse, y_dev_char_sparse

  def next_test(self, test_batch_size=10):
    start_index = self.test_index
    end_index = min(start_index + test_batch_size, self.test_size)
    batch_size = end_index - start_index

    x_test = self.test_x[start_index:end_index]
    y_test = self.test_y[start_index:end_index]
    if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram or self.hparams.char_input is not None:
      x_test_char_kv = self.test_x_char_kv[start_index:end_index]
      y_test_char_kv = self.test_y_char_kv[start_index:end_index]
      x_test, y_test, x_test_char_kv, y_test_char_kv = self.sort_by_xlen(x_test, y_test, x_test_char_kv, y_test_char_kv)
    else:
      #pass
      x_test, y_test = self.sort_by_xlen(x_test, y_test)

    if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram:
      x_test, x_mask, x_count, x_len, x_pos_emb_idxs, x_test_char_sparse = self._pad(x_test, self.hparams.pad_id, x_test_char_kv, self.hparams.src_char_vsize)
      y_test, y_mask, y_count, y_len, y_pos_emb_idxs, y_test_char_sparse = self._pad(y_test, self.hparams.pad_id, y_test_char_kv, self.hparams.trg_char_vsize)
    elif self.hparams.char_input is not None:
      x_test, x_mask, x_count, x_len, x_pos_emb_idxs, x_test_char_sparse = self._pad(x_test, self.hparams.pad_id, char_sents=x_test_char_kv)
      y_test, y_mask, y_count, y_len, y_pos_emb_idxs, y_test_char_sparse = self._pad(y_test, self.hparams.pad_id, char_sents=y_test_char_kv)
    else:
      x_test_char_sparse, y_test_char_sparse = None, None
      x_test, x_mask, x_count, x_len, x_pos_emb_idxs = self._pad(x_test, self.hparams.pad_id)
      y_test, y_mask, y_count, y_len, y_pos_emb_idxs = self._pad(y_test, self.hparams.pad_id)

    if end_index >= self.test_size:
      eop = True
      self.test_index = 0
    else:
      eop = False
      self.test_index += batch_size

    return x_test, x_mask, x_count, x_len, x_pos_emb_idxs, y_test, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, eop, x_test_char_sparse, y_test_char_sparse


    #if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram or self.hparams.char_input is not None:
    #  x_test_char_kv = self.test_x_char_kv[start_index:end_index]
    #  y_test_char_kv = self.test_y_char_kv[start_index:end_index]
    #if self.hparams.char_ngram_n > 0 or self.hparams.ngram_n or self.hparams.char_input is not None:
    #  x_test, x_mask, x_count, x_len, x_pos_emb_idxs, x_test_char = self._pad(x_test, self.pad_id, x_test_char_kv, self.hparams.src_char_vsize)
    #  y_test, y_mask, y_count, y_len, y_pos_emb_idxs, y_test_char = self._pad(y_test, self.pad_id, y_test_char_kv, self.hparams.trg_char_vsize)
    #else:
    #  x_test_char, y_test_char = None, None
    #  x_test, x_mask, x_count, x_len, x_pos_emb_idxs = self._pad(x_test, self.pad_id)
    #  y_test, y_mask, y_count, y_len, y_pos_emb_idxs = self._pad(y_test, self.pad_id)

    #if end_index >= self.test_size:
    #  eop = True
    #  self.test_index = 0
    #else:
    #  eop = False
    #  self.test_index += batch_size

    #return x_test, x_mask, x_count, x_len, x_pos_emb_idxs, y_test, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, eop, x_test_char, y_test_char

  def sort_by_xlen(self, x, y, x_char_kv=None, y_char_kv=None, file_index=None, descend=True):
    x = np.array(x)
    y = np.array(y)
    x_len = [len(i) for i in x]
    index = np.argsort(x_len)
    if descend:
      index = index[::-1]
    x, y = x[index].tolist(), y[index].tolist()
    if file_index:
      file_index = np.array(file_index)
      file_index = file_index[index].tolist()
    if x_char_kv:
      x_char_kv, y_char_kv = np.array(x_char_kv), np.array(y_char_kv)
      x_char_kv, y_char_kv = x_char_kv[index].tolist(), y_char_kv[index].tolist()
      if file_index:
        return x, y, x_char_kv, y_char_kv, file_index
      else:
        return x, y, x_char_kv, y_char_kv
    if file_index:
      return x, y, file_index
    else:
      return x, y

  def _pad(self, sentences, pad_id, char_kv=None, char_dim=None, char_sents=None):
    batch_size = len(sentences)
    lengths = [len(s) for s in sentences]
    count = sum(lengths)
    max_len = max(lengths)
    padded_sentences = [s + ([pad_id]*(max_len - len(s))) for s in sentences]
    if char_kv:
      for s, char in zip(sentences, char_kv): assert len(s) == len(char)
      char_sparse = []
      for kvs in char_kv:
        sent_sparse = []
        key, val = [], []
        for i, kv in enumerate(kvs):
          key.append(torch.LongTensor([[i for _ in range(len(kv.keys()))], list(kv.keys())]))
          val.extend(list(kv.values()))
        key = torch.cat(key, dim=1)
        val = torch.FloatTensor(val)
        sent_sparse = torch.sparse.FloatTensor(key, val, torch.Size([max_len, char_dim]))
        # (batch_size, max_len, char_dim)
        char_sparse.append(sent_sparse)
    elif char_sents:
      padded_char_sents = []
      max_char_len = max([len(w) for sent in char_sents for w in sent])
      for char_sent in char_sents:
        padded_char_sent = [s + ([pad_id]*(max_char_len-len(s))) for s in char_sent]
        padded_char_sent += [[pad_id]*max_char_len] * (max_len - len(padded_char_sent))
        padded_char_sents.append(padded_char_sent)
      # (batch_size, max_len, max_char_len, char_dim)
      padded_char_sents = Variable(torch.LongTensor(padded_char_sents))
      if self.hparams.cuda: padded_char_sents = padded_char_sents.cuda()
    mask = [[0]*len(s) + [1]*(max_len - len(s)) for s in sentences]
    padded_sentences = Variable(torch.LongTensor(padded_sentences))
    mask = torch.ByteTensor(mask)
    pos_emb_indices = [[i+1 for i in range(len(s))] + ([0]*(max_len - len(s))) for s in sentences]
    pos_emb_indices = Variable(torch.FloatTensor(pos_emb_indices))
    if self.hparams.cuda:
      padded_sentences = padded_sentences.cuda()
      pos_emb_indices = pos_emb_indices.cuda()
      mask = mask.cuda()
    if char_kv:
      return padded_sentences, mask, count, lengths, pos_emb_indices, char_sparse
    elif char_sents:
      return padded_sentences, mask, count, lengths, pos_emb_indices, padded_char_sents
    else:
      return padded_sentences, mask, count, lengths, pos_emb_indices

  def _get_char(self, word, i2w, w2i, n=1):
    chars = []
    for i in range(0, max(1, len(word)-n+1)):
      j = min(len(word), i+n)
      c = word[i:j]
      if c in w2i:
        chars.append(w2i[c])
      else:
        chars.append(self.hparams.unk_id)
    return chars

  def _get_ngram_counts(self, word, i2w, w2i, n):
    count = {}
    for i in range(len(word)):
      for j in range(i+1, min(len(word), i+n)+1):
        ngram = word[i:j]
        if ngram in w2i:
          ngram = w2i[ngram]
        else:
          ngram = 0
        if ngram not in count: count[ngram] = 0
        count[ngram] += 1
    return count

  def _get_bpe_ngram_counts(self, word, i2w, w2i):
    count = {}
    word = "▁" + word
    n = len(word)
    for i in range(len(word)):
      for j in range(i+1, min(len(word), i+n)+1):
        ngram = word[i:j]
        if ngram in w2i:
          ngram = w2i[ngram]
        else:
          ngram = 0
        if ngram not in count: count[ngram] = 0
        count[ngram] += 1
    return count


  def _build_parallel(self, src_file_name, trg_file_name, i, is_train=True):
    print("loading parallel sentences from {} {} with vocab {}".format(src_file_name, trg_file_name, i))
    with open(src_file_name, 'r', encoding='utf-8') as f:
      src_lines = f.read().split('\n')
    with open(trg_file_name, 'r', encoding='utf-8') as f:
      trg_lines = f.read().split('\n')
    if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram:
      src_char_kv_data = []
      trg_char_kv_data = []
    elif self.hparams.char_input is not None:
      src_char_data = []
      trg_char_data = []
    src_data = []
    trg_data = []
    line_count = 0
    skip_line_count = 0
    src_unk_count = 0
    trg_unk_count = 0

    src_lens = []
    if self.hparams.lan_code_rl and i > 0:
      src_unk_id = self.lan_code_list[i-1]
    else:
      src_unk_id = self.hparams.unk_id
    for src_line, trg_line in zip(src_lines, trg_lines):
      src_tokens = src_line.split()
      trg_tokens = trg_line.split()
      if is_train and not src_tokens or not trg_tokens: 
        skip_line_count += 1
        continue
      if is_train and not self.hparams.decode and self.hparams.max_len and len(src_tokens) > self.hparams.max_len and len(trg_tokens) > self.hparams.max_len:
        skip_line_count += 1
        continue
      
      src_lens.append(len(src_tokens))
      src_indices, trg_indices = [self.hparams.bos_id], [self.hparams.bos_id] 
      if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram:
          src_char_kv, trg_char_kv = [{0:0}], [{0:0}]
      elif self.hparams.char_input is not None:
          src_char, trg_char = [[self.hparams.pad_id]], [[self.hparams.pad_id]]
      src_w2i = self.src_w2i_list[i]
      for src_tok in src_tokens:
        #print(src_tok)
        if src_tok not in src_w2i:
          src_indices.append(src_unk_id)
          src_unk_count += 1
          #print("unk {}".format(src_unk_count))
        else:
          src_indices.append(src_w2i[src_tok])
          #print("src id {}".format(src_w2i[src_tok]))
        # calculate char ngram emb for src_tok
        if self.hparams.char_ngram_n > 0:
          ngram_counts = self._get_ngram_counts(src_tok, self.src_char_i2w, self.src_char_w2i, self.hparams.char_ngram_n)
          src_char_kv.append(ngram_counts)
        elif self.hparams.bpe_ngram:
          ngram_counts = self._get_bpe_ngram_counts(src_tok, self.src_char_i2w, self.src_char_w2i)
          src_char_kv.append(ngram_counts)
        elif not self.hparams.char_input is None:
          src_char.append(self._get_char(src_tok, self.src_char_i2w,
            self.src_char_w2i, n=self.hparams.n))

      trg_w2i = self.trg_w2i_list[i]
      for trg_tok in trg_tokens:
        if trg_tok not in trg_w2i:
          trg_indices.append(self.hparams.unk_id)
          trg_unk_count += 1
        else:
          trg_indices.append(trg_w2i[trg_tok])
        # calculate char ngram emb for trg_tok
        if self.hparams.char_ngram_n > 0:
          ngram_counts = self._get_ngram_counts(trg_tok, self.trg_char_i2w, self.trg_char_w2i, self.hparams.char_ngram_n)
          trg_char_kv.append(ngram_counts)
        elif self.hparams.bpe_ngram:
          ngram_counts = self._get_bpe_ngram_counts(trg_tok, self.trg_char_i2w, self.trg_char_w2i)
          trg_char_kv.append(ngram_counts)
        elif self.hparams.char_input is not None:
          trg_char.append(self._get_char(trg_tok, self.trg_char_i2w,
            self.trg_char_w2i, n=self.hparams.n))

      src_indices.append(self.hparams.eos_id)
      trg_indices.append(self.hparams.eos_id)
      src_data.append(src_indices)
      trg_data.append(trg_indices)
      if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram:
        src_char_kv.append({0:0})
        trg_char_kv.append({0:0})
        src_char_kv_data.append(src_char_kv)
        trg_char_kv_data.append(trg_char_kv)
      elif self.hparams.char_input is not None:
        src_char.append([self.hparams.pad_id])
        trg_char.append([self.hparams.pad_id])
        src_char_data.append(src_char)
        trg_char_data.append(trg_char)
      line_count += 1
      if line_count % 10000 == 0:
        print("processed {} lines".format(line_count))
    if is_train:
      if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram:
        src_data, trg_data, src_char_kv_data, trg_char_kv_data = self.sort_by_xlen(src_data, trg_data, src_char_kv_data, trg_char_kv_data, descend=False)
      elif self.hparams.char_input is not None:
        src_data, trg_data, src_char_data, trg_char_data = self.sort_by_xlen(src_data, trg_data, src_char_kv_data, trg_char_kv_data, descend=False)
      else:
        src_data, trg_data = self.sort_by_xlen(src_data, trg_data, descend=False)
    print("src_unk={}, trg_unk={}".format(src_unk_count, trg_unk_count))
    assert len(src_data) == len(trg_data)
    print("lines={}, skipped_lines={}".format(len(src_data), skip_line_count))
    if self.hparams.char_ngram_n or self.hparams.bpe_ngram:
      return src_data, trg_data, src_char_kv_data, trg_char_kv_data, src_lens
    elif self.hparams.char_input is not None:
      return src_data, trg_data, src_char_data, trg_char_data, src_lens
    return src_data, trg_data, None, None, src_lens

  def _build_char_vocab(self, lines, n=1):
    i2w = ['<pad>', '<unk>', '<s>', '<\s>']
    w2i = {'<pad>': 0, '<unk>':1, '<s>':2, '<\s>':3}
    assert i2w[self.hparams.pad_id] == '<pad>'
    assert i2w[self.hparams.unk_id] == '<unk>'
    assert i2w[self.hparams.bos_id] == '<s>'
    assert i2w[self.hparams.eos_id] == '<\s>'
    assert w2i['<pad>'] == self.hparams.pad_id
    assert w2i['<unk>'] == self.hparams.unk_id
    assert w2i['<s>'] == self.hparams.bos_id
    assert w2i['<\s>'] == self.hparams.eos_id
    for line in lines:
      words = line.split()
      for w in words:
        for i in range(0, max(1, len(w)-n+1)):
        #for c in w:
          j = min(len(w), i+n)
          c = w[i:j]
          if c not in w2i:
            w2i[c] = len(w2i)
            i2w.append(c)
    return i2w, w2i

  def _build_char_ngram_vocab(self, lines, n, max_char_vocab_size=None):
    i2w = ['<unk>']
    w2i = {}
    w2i['<unk>'] = 0

    for line in lines:
      words = line.split()
      for w in words:
        for i in range(len(w)):
          for j in range(i+1, min(i+n, len(w))+1):
            char = w[i:j]
            if char not in w2i:
              w2i[char] = len(w2i)
              i2w.append(char)
              if max_char_vocab_size and len(i2w) >= max_char_vocab_size: 
                return i2w, w2i
    return i2w, w2i

  def _build_vocab(self, vocab_file, max_vocab_size=None):
    i2w = []
    w2i = {}
    i = 0
    with open(vocab_file, 'r', encoding='utf-8') as f:
      for line in f:
        w = line.strip()
        if i == 0 and w != "<pad>":
          i2w = ['<pad>', '<unk>', '<s>', '<\s>']
          w2i = {'<pad>': 0, '<unk>':1, '<s>':2, '<\s>':3}
          i = 4
        w2i[w] = i
        i2w.append(w)
        i += 1
        if max_vocab_size and i >= max_vocab_size:
          break
    assert i2w[self.hparams.pad_id] == '<pad>'
    assert i2w[self.hparams.unk_id] == '<unk>'
    assert i2w[self.hparams.bos_id] == '<s>'
    assert i2w[self.hparams.eos_id] == '<\s>'
    assert w2i['<pad>'] == self.hparams.pad_id
    assert w2i['<unk>'] == self.hparams.unk_id
    assert w2i['<s>'] == self.hparams.bos_id
    assert w2i['<\s>'] == self.hparams.eos_id
    return i2w, w2i

  def _build_vocab_list(self, vocab_file_list, max_vocab_size=None):
    i2w = ['<pad>', '<unk>', '<s>', '<\s>']
    w2i = {'<pad>': 0, '<unk>':1, '<s>':2, '<\s>':3}
    i = 4
    for vocab_file in vocab_file_list:
      with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
          w = line.strip()
          if w == '<unk>' or w == '<pad>' or w == '<s>' or w == '<\s>': continue
          w2i[w] = i
          i2w.append(w)
          i += 1
          if max_vocab_size and i >= max_vocab_size:
            break
    assert i2w[self.hparams.pad_id] == '<pad>'
    assert i2w[self.hparams.unk_id] == '<unk>'
    assert i2w[self.hparams.bos_id] == '<s>'
    assert i2w[self.hparams.eos_id] == '<\s>'
    assert w2i['<pad>'] == self.hparams.pad_id
    assert w2i['<unk>'] == self.hparams.unk_id
    assert w2i['<s>'] == self.hparams.bos_id
    assert w2i['<\s>'] == self.hparams.eos_id
    return i2w, w2i


  def _build_char_vocab_from(self, vocab_file_list, vocab_size_list, n=None,
      single_n=False):
    vfile_list = vocab_file_list.split(",")
    vsize_list = [int(s) for s in vocab_size_list.split(",")]
    if self.hparams.ordered_char_dict:
      i2w = [ '<unk>']
      i2w_set = set(i2w) 
      for vfile, size in zip(vfile_list, vsize_list):
        cur_vsize = 0
        with open(vfile, 'r', encoding='utf-8') as f:
          for line in f:
            w = line.strip()
            if single_n and n and len(w) != n: continue
            if not single_n and n and len(w) > n: continue 
            if w == '<unk>' or w == '<pad>' or w == '<s>' or w == '<\s>': continue
            if w not in i2w_set:
              cur_vsize += 1
              i2w.append(w)
              i2w_set.add(w)
              if size >= 0 and cur_vsize > size: break
    else:
      i2w_sets = []
      for vfile, size in zip(vfile_list, vsize_list):
        i2w = []
        with open(vfile, 'r', encoding='utf-8') as f:
          for line in f:
            w = line.strip()
            if single_n and n and len(w) != n: continue
            if not single_n and n and len(w) > n: continue 
            if w == '<unk>' or w == '<pad>' or w == '<s>' or w == '<\s>': continue
            i2w.append(w)
            if size > 0 and len(i2w) > size: break 
        i2w_sets.append(set(i2w))
      i2w_set = set([])
      for s in i2w_sets:
        i2w_set = i2w_set | s
      i2w = ['<unk>'] + list(i2w_set)

    w2i = {}
    for i, w in enumerate(i2w):
      w2i[w] = i
    return i2w, w2i

