import random
import numpy as np
import os
import functools

import torch
from torch.autograd import Variable
# multilingual data utils


class MultDataUtil(object):
  def __init__(self, hparams, shuffle=True):
    self.hparams = hparams
    self.src_i2w_list = []
    self.src_w2i_list = []
    
    self.shuffle = shuffle

    if self.hparams.src_vocab:
      self.src_i2w, self.src_w2i = self._build_vocab(self.hparams.src_vocab, max_vocab_size=self.hparams.src_vocab_size)
      self.hparams.src_vocab_size = len(self.src_i2w)
    else:
      print("not using single src word vocab..")

    if self.hparams.trg_vocab:
      self.trg_i2w, self.trg_w2i = self._build_vocab(self.hparams.trg_vocab, max_vocab_size=self.hparams.trg_vocab_size)
      self.hparams.trg_vocab_size = len(self.trg_i2w)
    else:
      print("not using single trg word vocab..")

    if self.hparams.lang_file:
      self.train_src_file_list = []
      self.train_trg_file_list = []
      if self.hparams.src_char_vocab_from:
        self.src_char_vocab_from = []
      if self.hparams.src_vocab_list:
        self.src_vocab_list = []
      if self.hparams.sample_load:
        self.sample_prob_list = []
      self.lans = []
      with open(self.hparams.lang_file, "r") as myfile:
        for line in myfile:
          lan = line.strip()
          self.lans.append(lan)
          if self.hparams.src_char_vocab_from:
            self.src_char_vocab_from.append(self.hparams.src_char_vocab_from.replace("LAN", lan))
          self.train_src_file_list.append(self.hparams.train_src_file_list[0].replace("LAN", lan))
          self.train_trg_file_list.append(self.hparams.train_trg_file_list[0].replace("LAN", lan))
          if self.hparams.src_vocab_list:
            self.src_vocab_list.append(self.hparams.src_vocab_list[0].replace("LAN", lan))
          if self.hparams.sample_load:
            self.sample_prob_list.append(self.hparams.sample_prob_list.replace("LAN", lan))

        if self.hparams.select_data:
          for i in range(1,len(self.train_src_file_list)):
            self.train_src_file_list[i] = self.train_src_file_list[i] + "." + self.hparams.sel
            self.train_trg_file_list[i] = self.train_trg_file_list[i] + "." + self.hparams.sel
            print(self.train_src_file_list)
            print(self.train_trg_file_list)
      self.hparams.lan_size = len(self.train_src_file_list)
    if self.hparams.semb_num > 1:
      self.src_i2w_list, self.src_w2i_list = [], []
      for i, src_vocab in enumerate(self.src_vocab_list):
        src_i2w, src_w2i = self._build_vocab(src_vocab, max_vocab_size=self.hparams.src_vocab_size)
        print("data {} src vocab size is {}".format(i, len(src_i2w)))
        self.src_i2w_list.append(src_i2w)
        self.src_w2i_list.append(src_w2i)

    if self.hparams.src_vocab_list:
      self.src_i2w, self.src_w2i = self._build_char_vocab_from(self.src_vocab_list, self.hparams.src_vocab_size)
      self.hparams.src_vocab_size = len(self.src_i2w)
      print("use combined src vocab at size {}".format(self.hparams.src_vocab_size))

    if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram:
      self.src_char_i2w, self.src_char_w2i = self._build_char_vocab_from(self.src_char_vocab_from, self.hparams.src_char_vocab_size, n=self.hparams.char_ngram_n, single_n=self.hparams.single_n)
      self.src_char_vsize = len(self.src_char_i2w)
      setattr(self.hparams, 'src_char_vsize', self.src_char_vsize)
      print("src_char_vsize={}".format(self.src_char_vsize))
      #if self.hparams.compute_ngram:
      #  self.
    else:
      self.src_char_vsize = None
      setattr(self.hparams, 'src_char_vsize', None)

    if (not self.hparams.decode):
      self.start_indices = [[] for i in range(len(self.train_src_file_list))]
      self.end_indices = [[] for i in range(len(self.train_src_file_list))]
    if self.hparams.sample_select:
      self.start_indices_shared = []
      self.end_indices_shared = []     
      # {trg: [(src1, src_char1, src_len1, sim1), (src2, src_char1, src_len1, sim2), ..., srcn]}
      self.trg2srcs, self.shared_trgs = self.get_trg2srcs()

    if hasattr(self.hparams, 'uni') and self.hparams.uni:
      self.pretrained_src_emb_list = []
      self.pretrained_trg_emb = None
      #src_w2i = {}
      #src_i2w = []
      for i, emb_file in enumerate(self.hparams.pretrained_src_emb_list):
        print("load emb from {}".format(emb_file))
        emb, i2w, w2i = self.load_pretrained(emb_file)
        self.pretrained_src_emb_list.append(emb)
        #for w in i2w:
        #  if w not in src_w2i:
        #    src_w2i[w] = len(src_w2i)
        #    src_i2w.append(w)
        self.src_w2i_list.append(w2i)
        self.src_i2w_list.append(i2w)
      #self.src_i2w = src_i2w
      #self.src_w2i = src_w2i
      
      self.pretrained_trg_emb, _, _ = self.load_pretrained(self.hparams.pretrained_trg_emb)
 
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


  def get_trg2srcs(self):
    trg2srcs = {}
    x_train, y_train, x_char_kv, x_len, x_rank = self._build_parallel(self.train_src_file_list[0], self.train_trg_file_list[0], 0, outprint=True, not_sample=True)
    for i, y in enumerate(y_train):
      y = tuple(y)
      if not y in trg2srcs:
        trg2srcs[y] = [[[] for _ in range(self.hparams.lan_size)], [[] for _ in range(self.hparams.lan_size)], [0 for _ in range(self.hparams.lan_size)]]
      if x_train:
        trg2srcs[y][0][0] = x_train[i]
      else:
        trg2srcs[y][1][0] = x_char_kv[i]
      trg2srcs[y][2][0] = x_len[i]

    for data_idx in range(1, self.hparams.lan_size):
      x_train, y_train, x_char_kv, x_len, x_rank = self._build_parallel(self.train_src_file_list[data_idx], self.train_trg_file_list[data_idx], data_idx, outprint=True, not_sample=True)
      for i, y in enumerate(y_train):
        y = tuple(y)
        if y in trg2srcs:
          if x_train:
            trg2srcs[y][0][data_idx] = x_train[i]
          else:
            trg2srcs[y][1][data_idx] = x_char_kv[i]
          trg2srcs[y][2][data_idx] = x_len[i]
    shared_trgs = []
    for k, v in trg2srcs.items():
      skip = False
      for i in range(self.hparams.lan_size):
        if v[2][i] == 0:
          skip = True
          break
      if not skip: shared_trgs.append(k)
    print("total number of shared engs: {}".format(len(shared_trgs)))
    return trg2srcs, shared_trgs

  def next_train_select(self):
    step = 0
    while True:
      # set batcher indices once
      if not self.start_indices_shared:
        print("batching data...")
        start_indices, end_indices = [], []
        trgs = np.array(list(self.trg2srcs.keys()))
        lens = [len(t) for t in self.shared_trgs]
        trg_idx = np.argsort(lens)
        self.shared_trgs = np.array(self.shared_trgs)[trg_idx].tolist()
        if self.hparams.batcher == "word":
          start_index, end_index, count = 0, 0, 0
          for trg in self.shared_trgs:
            src_item = self.trg2srcs[trg]
            count += (max(src_item[2]) + len(trg))
            end_index += 1
            if count > self.hparams.batch_size: 
              start_indices.append(start_index)
              end_indices.append(end_index)
              count = 0
              start_index = end_index
          if start_index < end_index:
            start_indices.append(start_index)
            end_indices.append(end_index)
        elif self.hparams.batcher == "sent":
          start_index, end_index, count = 0, 0, 0
          while end_index < len(self.trg2srcs):
            end_index = min(start_index + self.hparams.batch_size, len(self.trg2srcs))
            start_indices.append(start_index)
            end_indices.append(end_index)
            start_index = end_index
        else:
          print("unknown batcher")
          exit(1)
        self.start_indices_shared = start_indices
        self.end_indices_shared = end_indices
        print("finished batching data...")
      for step_b, batch_idx in enumerate(np.random.permutation(len(self.start_indices_shared))):
      #for step_b, batch_idx in enumerate([i for i in range(len(self.start_indices_shared))]):
        step += 1
        start_index, end_index = self.start_indices_shared[batch_idx], self.end_indices_shared[batch_idx]
        for src_idx in range(self.hparams.lan_size):
          x, y, x_char, train_file_index = [], [], [], []
          for i in range(start_index, end_index):
            trg = self.shared_trgs[i]
            src_item = self.trg2srcs[trg]
            if len(src_item[1][src_idx]) > 0 or len(src_item[0][src_idx]) > 0:
              y.append(list(trg))
              if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram:
                x_char.append(src_item[1][src_idx])
              else:
                x.append(src_item[0][src_idx])
              train_file_index.append(src_idx)
          if self.shuffle:
            x, y, x_char, train_file_index = self.sort_by_xlen([x, y, x_char, train_file_index])
          # pad
          x, x_mask, x_count, x_len, x_pos_emb_idxs, x_char, x_rank = self._pad(x, self.hparams.pad_id, x_char, self.hparams.src_char_vsize)
          y, y_mask, y_count, y_len, y_pos_emb_idxs, y_char, y_rank = self._pad(y, self.hparams.pad_id)
          batch_size = end_index - start_index
          if step_b == len(self.start_indices_shared)-1:
            eop = True
          else:
            eop = False
          yield x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, None, eop, eop, train_file_index, []

  def next_train_select_all(self):
    step = 0
    while True:
      # set batcher indices once
      if not self.start_indices_shared:
        print("batching data...")
        start_indices, end_indices = [], []
        trgs = np.array(list(self.trg2srcs.keys()))
        lens = [len(t) for t in self.shared_trgs]
        trg_idx = np.argsort(lens)
        self.shared_trgs = np.array(trgs)[trg_idx].tolist()
        if self.hparams.batcher == "word":
          start_index, end_index, count = 0, 0, 0
          for trg in self.shared_trgs:
            src_item = self.trg2srcs[trg]
            count += (max(src_item[2]) + len(trg))
            end_index += 1
            if count > self.hparams.batch_size: 
              start_indices.append(start_index)
              end_indices.append(end_index)
              count = 0
              start_index = end_index
          if start_index < end_index:
            start_indices.append(start_index)
            end_indices.append(end_index)
        elif self.hparams.batcher == "sent":
          start_index, end_index, count = 0, 0, 0
          while end_index < len(self.trg2srcs):
            end_index = min(start_index + self.hparams.batch_size, len(self.trg2srcs))
            start_indices.append(start_index)
            end_indices.append(end_index)
            start_index = end_index
        else:
          print("unknown batcher")
          exit(1)
        self.start_indices_shared = start_indices
        self.end_indices_shared = end_indices
        print("finished batching data...")
      self.ave_grad = 20
      for src_idx in range(1, self.hparams.lan_size):
        for step_b, batch_idx in enumerate(np.random.permutation(len(self.start_indices_shared))):
        #for step_b, batch_idx in enumerate([i for i in range(len(self.start_indices_shared))]):
          step += 1
          if step_b == self.ave_grad: break
          start_index, end_index = self.start_indices_shared[batch_idx], self.end_indices_shared[batch_idx]
          x1, y, x1_char, train_file_index_1, train_file_index_0, x0, x0_char = [], [], [], [], [], [], []
          for i in range(start_index, end_index):
            trg = self.shared_trgs[i]
            src_item = self.trg2srcs[trg]
            if len(src_item[1][src_idx]) > 0 or len(src_item[0][src_idx]) > 0:
              y.append(list(trg))
              if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram:
                x1_char.append(src_item[1][src_idx])
                x0_char.append(src_item[1][0])
              else:
                x1.append(src_item[0][src_idx])
                x0.append(src_item[0][0])
              train_file_index_1.append(src_idx)
              train_file_index_0.append(0)
          if len(train_file_index_0) == 0: continue
          for x, x_char, train_file_index in zip([x0, x1], [x0_char, x1_char], [train_file_index_0, train_file_index_1]):
            if self.shuffle:
              x, y, x_char, train_file_index = self.sort_by_xlen([x, y, x_char, train_file_index])
            # pad
            x, x_mask, x_count, x_len, x_pos_emb_idxs, x_char, x_rank = self._pad(x, self.hparams.pad_id, x_char, self.hparams.src_char_vsize)
            y, y_mask, y_count, y_len, y_pos_emb_idxs, y_char, y_rank = self._pad(y, self.hparams.pad_id)
            batch_size = end_index - start_index
            if step_b == len(self.start_indices_shared)-1:
              eop = True
            else:
              eop = False
            yield x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, None, eop, eop, train_file_index, []

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

  def next_train(self):
      #if self.hparams.sample_select:
      #  return self.next_train_select()
      #else:
      #  return self.next_train_normal()
      return self.next_train_normal()

  def next_train_normal(self):
    while True:
      step = 0
      if self.hparams.lang_shuffle:
        self.train_data_queue = np.random.permutation(len(self.train_src_file_list))
      else:
        self.train_data_queue = [i for i in range(len(self.train_src_file_list))]
      if hasattr(self, "topk_train_queue"):
        self.train_data_queue = self.topk_train_queue
      for data_idx in self.train_data_queue:
        x_train, y_train, x_char_kv, x_len, x_rank = self._build_parallel(self.train_src_file_list[data_idx], self.train_trg_file_list[data_idx], data_idx, outprint=(self.hparams.sample_load or len(self.start_indices[data_idx]) == 0))
        #x_train, y_train, x_char_kv, x_len, x_rank = self._build_parallel(self.train_src_file_list[data_idx], self.train_trg_file_list[data_idx], outprint=True)
        # set batcher indices once
        if len(x_len) == 0:
          print("skipping 0 size file {}".format(self.train_src_file_list[data_idx]))
          continue
        if not self.start_indices[data_idx] or self.hparams.sample_load:
          start_indices, end_indices = [], []
          if self.hparams.batcher == "word":
            start_index, end_index, count = 0, 0, 0
            #lines, max_src_count, max_trg_count = 0, 0, 0
            while True:
              count += (x_len[end_index] + len(y_train[end_index]))
              #max_src_count = max(max_src_count, x_len[end_index])
              #max_trg_count = max(max_trg_count, len(y_train[end_index]))
              end_index += 1
              #lines += 1
              #count = lines * (max_src_count + max_trg_count)
              if end_index >= len(x_len):
                if self.hparams.update_batch > 1:
                  interv = (end_index - start_index) // self.hparams.update_batch
                  start_indices_list = [start_index + i * interv for i in range(self.hparams.update_batch)]
                  for i in range(self.hparams.update_batch-1):
                    if start_indices_list[i] < start_indices_list[i+1]-1:
                      start_indices.append(start_indices_list[i]) 
                      end_indices.append(start_indices_list[i+1]-1) 
                  if start_indices_list[-1] < end_index:
                    start_indices.append(start_indices_list[-1]) 
                    end_indices.append(end_index) 
                else: 
                  start_indices.append(start_index)
                  end_indices.append(end_index)
                break
              if count > self.hparams.batch_size:
                if self.hparams.update_batch > 1:
                  interv = (end_index - start_index) // self.hparams.update_batch
                  start_indices_list = [start_index + i * interv for i in range(self.hparams.update_batch)]
                  for i in range(self.hparams.update_batch-1):
                    if start_indices_list[i] < start_indices_list[i+1]-1:
                      start_indices.append(start_indices_list[i]) 
                      end_indices.append(start_indices_list[i+1]-1) 
                  if start_indices_list[-1] < end_index:
                    start_indices.append(start_indices_list[-1]) 
                    end_indices.append(end_index)
                else: 
                  start_indices.append(start_index)
                  end_indices.append(end_index)
                count = 0
                start_index = end_index
                #lines, max_src_count, max_trg_count = 0, 0, 0
          elif self.hparams.batcher == "sent":
            start_index, end_index, count = 0, 0, 0
            while end_index < len(x_len):
              end_index = min(start_index + self.hparams.batch_size, len(x_len))
              start_indices.append(start_index)
              end_indices.append(end_index)
              start_index = end_index
          else:
            print("unknown batcher")
            exit(1)
          self.start_indices[data_idx] = start_indices
          self.end_indices[data_idx] = end_indices
        cached = []
        for step_b, batch_idx in enumerate(np.random.permutation(len(self.start_indices[data_idx]))):
          step += 1

          yield self.yield_data(data_idx, batch_idx, x_train, x_char_kv, y_train, step_b, x_rank)
 
  def yield_data(self, data_idx, batch_idx, x_train, x_char_kv, y_train, step, x_train_rank):
    start_index, end_index = self.start_indices[data_idx][batch_idx], self.end_indices[data_idx][batch_idx]
    x, y, x_char, x_rank = [], [], [], []
    if x_train:
      x = x_train[start_index:end_index]
    if x_char_kv:
      x_char = x_char_kv[start_index:end_index]
    if x_train_rank:
      x_rank = x_train_rank[start_index:end_index]

    y = y_train[start_index:end_index]
    train_file_index = [data_idx for i in range(end_index - start_index)] 
    if self.shuffle:
      x, y, x_char, train_file_index, x_rank = self.sort_by_xlen([x, y, x_char, train_file_index, x_rank])
    # pad
    x, x_mask, x_count, x_len, x_pos_emb_idxs, x_char, x_rank = self._pad(x, self.hparams.pad_id, x_char, self.hparams.src_char_vsize, x_rank)
    y, y_mask, y_count, y_len, y_pos_emb_idxs, y_char, y_rank = self._pad(y, self.hparams.pad_id)
    batch_size = end_index - start_index
    if step == len(self.start_indices[data_idx])-1:
      eof = True
    else:
      eof = False
    if data_idx != 0 and data_idx == self.train_data_queue[-1] and step == len(self.start_indices[data_idx])-1:
      eop = True
    else:
      eop = False
    return x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, None, eop, eof, train_file_index, x_rank

  def next_dev(self, dev_batch_size=1, data_idx=0):
    first_dev = True
    idxes = [0]
    if data_idx != 0:
      idxes.append(data_idx)
    else:
      idxes.append(1)
    if len(self.hparams.dev_src_file_list) == 1:
      idxes = [0]
    while True:
      #for data_idx in range(len(self.hparams.dev_src_file_list)):
      for data_idx in idxes:
        x_dev, y_dev, x_char_kv, x_dev_len, x_dev_rank = self._build_parallel(self.hparams.dev_src_file_list[data_idx], self.hparams.dev_trg_file_list[data_idx], data_idx, is_train=False, outprint=True)
        first_dev = False
        start_index, end_index = 0, 0
        while end_index < len(x_dev_len):
          end_index = min(start_index + dev_batch_size, len(x_dev_len))
          x, y, x_char = [], [], [] 
          if x_dev:
            x = x_dev[start_index:end_index]
          if x_char_kv:
            x_char = x_char_kv[start_index:end_index]
          y = y_dev[start_index:end_index]
          dev_file_index = [self.hparams.dev_file_idx_list[data_idx] for i in range(end_index - start_index)]
          if self.hparams.semb_num > 1:
            x_rank = x_dev_rank[start_index:end_index]
          else:
            x_rank = []
          if self.shuffle:
            x, y, x_char, dev_file_index, x_rank = self.sort_by_xlen([x, y, x_char, dev_file_index, x_rank])
          # pad
          x, x_mask, x_count, x_len, x_pos_emb_idxs, x_char, x_rank = self._pad(x, self.hparams.pad_id, x_char, self.hparams.src_char_vsize, x_rank)
          y, y_mask, y_count, y_len, y_pos_emb_idxs, y_char, y_rank = self._pad(y, self.hparams.pad_id)
          batch_size = end_index - start_index
          if end_index == len(x_dev_len):
            eof = True
          else:
            eof = False
          if data_idx == idxes[-1] and eof:
            eop = True
          else:
            eop = False
          start_index = end_index
          yield x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, None, eop, eof, dev_file_index, x_rank


  def next_test(self, test_batch_size=1):
    while True:
      for data_idx in range(len(self.hparams.test_src_file_list)):
        x_test, y_test, x_char_kv, x_test_len, x_rank = self._build_parallel(self.hparams.test_src_file_list[data_idx], self.hparams.test_trg_file_list[data_idx], data_idx, is_train=False, outprint=True)
        start_index, end_index = 0, 0
        while end_index < len(x_test_len):
          end_index = min(start_index + test_batch_size, len(x_test_len))
          x, y, x_char = [], [], [] 
          if x_test:
            x = x_test[start_index:end_index]
          if x_char_kv:
            x_char = x_char_kv[start_index:end_index]
          y = y_test[start_index:end_index]
          test_file_index = [self.hparams.test_file_idx_list[data_idx] for i in range(end_index - start_index)] 
          if self.shuffle:
            x, y, x_char, test_file_index = self.sort_by_xlen([x, y, x_char, test_file_index])

          # pad
          x, x_mask, x_count, x_len, x_pos_emb_idxs, x_char, x_rank = self._pad(x, self.hparams.pad_id, x_char, self.hparams.src_char_vsize, x_rank)
          y, y_mask, y_count, y_len, y_pos_emb_idxs, y_char, y_rank = self._pad(y, self.hparams.pad_id)
          batch_size = end_index - start_index
          if end_index == len(x_test_len):
            eof = True
          else:
            eof = False
          if data_idx == len(self.hparams.test_src_file_list)-1 and eof:
            eop = True
          else:
            eop = False
          start_index = end_index
          yield x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, None, eop, eof, test_file_index, x_rank


  def sort_by_xlen(self, data_list, descend=True):
    array_list = [np.array(x) for x in data_list]
    if data_list[0]:
      x_len = [len(i) for i in data_list[0]]
    else:
      x_len = [len(i) for i in data_list[2]]
    index = np.argsort(x_len)
    if descend:
      index = index[::-1]
    for i, x in enumerate(array_list):
      if x is not None and len(x) > 0:
        data_list[i] = x[index].tolist()
    return data_list 

  def _pad(self, sentences, pad_id, char_kv=None, char_dim=None, x_rank=None):
    if sentences:
      batch_size = len(sentences)
      lengths = [len(s) for s in sentences]
      count = sum(lengths)
      max_len = max(lengths)
      padded_sentences = [s + ([pad_id]*(max_len - len(s))) for s in sentences]
      padded_sentences = Variable(torch.LongTensor(padded_sentences))
      char_sparse = None
      padded_x_rank = None
    else:
      batch_size = len(char_kv)
      lengths = [len(s) for s in char_kv]
      padded_sentences = None
      count = sum(lengths)
      max_len = max(lengths)
      char_sparse = []
      if x_rank:
        padded_x_rank = [s + ([0]*(max_len - len(s))) for s in x_rank]
      else:
        padded_x_rank = None

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
    mask = [[0]*l + [1]*(max_len - l) for l in lengths]
    mask = torch.ByteTensor(mask)
    pos_emb_indices = [[i+1 for i in range(l)] + ([0]*(max_len - l)) for l in lengths]
    pos_emb_indices = Variable(torch.FloatTensor(pos_emb_indices))
    if self.hparams.cuda:
      if sentences:
        padded_sentences = padded_sentences.cuda()
      pos_emb_indices = pos_emb_indices.cuda()
      mask = mask.cuda()
    return padded_sentences, mask, count, lengths, pos_emb_indices, char_sparse, padded_x_rank

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

  @functools.lru_cache(maxsize=8000, typed=False)
  def _get_ngram_counts(self, word):
    count = {}
    for i in range(len(word)):
      for j in range(i+1, min(len(word), i+self.hparams.char_ngram_n)+1):
        ngram = word[i:j]
        if ngram in self.src_char_w2i:
          ngram = self.src_char_w2i[ngram]
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

  def update_prob_list(self, data_weights):
    trg2srcs = {}
    t = 1
    lan_lists = self.lans[1:] 
    sim_rank = data_weights[1:]
    if len(self.lans[1:]) > self.hparams.topk:
      sorted_idx = np.argsort(sim_rank)[::-1]
      lan_lists = np.array(lan_lists)[sorted_idx].tolist()[:self.hparams.topk]
      sim_rank = np.array(sim_rank)[sorted_idx].tolist()[:self.hparams.topk]
      
      self.topk_train_queue = ((sorted_idx + 1).tolist())[:self.hparams.topk]
      print("training using the top k language {}".format(self.topk_train_queue))
    print(lan_lists)
    print(sim_rank)
    
    assert len(lan_lists) == len(sim_rank)
    #sim_rank = [63.31, 42.56, 40.76, 39.41, 36.73]

    sim_rank = [i/t for i in sim_rank]
    out_probs = []
    for i, lan in enumerate(lan_lists):
      trg_file = "data/{}_eng/ted-train.mtok.spm8000.eng".format(lan)
      trg_sents = open(trg_file, 'r').readlines()
      out_probs.append([0 for _ in range(len(trg_sents))])
      line = 0
      for trg in trg_sents:
        if trg not in trg2srcs: trg2srcs[trg] = []
        trg2srcs[trg].append([i, line, sim_rank[i]])
        line += 1
    print("eng size: {}".format(len(trg2srcs)))
    for trg, src_list in trg2srcs.items():
      sum_score = 0
      for s in src_list:
        s[2] = np.exp(s[2])
        sum_score += s[2]
      for s in src_list:
        s[2] = s[2] / sum_score
        out_probs[s[0]][s[1]] = s[2]
    base_lines = len(open( "data/{}_eng/ted-train.mtok.spm8000.eng".format(self.lans[0])).readlines())
    if len(self.lans[1:]) > self.hparams.topk:
      self.sample_probs = out_probs
      out_probs_all = [[] for _ in range(len(self.lans))]
      for i, idx in enumerate(sorted_idx[:self.hparams.topk]):
        out_probs_all[idx+1] = out_probs[i]
      self.sample_probs = out_probs_all
    else:
      self.sample_probs = out_probs
      self.topk_train_queue = lan_lists
    print("using topK train queue {}...".format(str(self.topk_train_queue)))

  def update_base_prob_list(self):
    base_lines = len(open( "data/{}_eng/ted-train.mtok.spm8000.eng".format(self.lans[0])).readlines())
    self.sample_probs = [[1 for _ in range(base_lines)]] 
    self.topk_train_queue = [0]
    

  def _build_parallel(self, src_file_name, trg_file_name, data_idx, is_train=True, shuffle=True, outprint=False, not_sample=False):
    if outprint:
      print("loading parallel sentences from {} {}".format(src_file_name, trg_file_name))
    with open(src_file_name, 'r', encoding='utf-8') as f:
      src_lines = f.read().split('\n')
    with open(trg_file_name, 'r', encoding='utf-8') as f:
      trg_lines = f.read().split('\n')
    if is_train and not self.hparams.decode and self.hparams.sample_load and not not_sample:
      if hasattr(self, "sample_probs"):
        probs = self.sample_probs[data_idx]
      else:
        f = self.sample_prob_list[data_idx]
        probs = [float(i) for i in open(f, 'r').readlines()]

    src_char_kv_data = []
    src_data = []
    trg_data = []
    line_count = 0
    skip_line_count = 0
    src_unk_count = 0
    trg_unk_count = 0
    src_word_rank = [] 
    if self.hparams.semb_num > 1:
      cur_src_w2i = self.src_w2i_list[data_idx]

    src_lens = []
    line_n = -1
    for src_line, trg_line in zip(src_lines, trg_lines):
      src_tokens = src_line.split()
      trg_tokens = trg_line.split()
      line_n += 1
      if is_train and not src_tokens or not trg_tokens: 
        skip_line_count += 1
        continue
      if is_train and not self.hparams.decode and self.hparams.max_len and len(src_tokens) > self.hparams.max_len and len(trg_tokens) > self.hparams.max_len:
        skip_line_count += 1
        continue
      if is_train and not self.hparams.decode and self.hparams.sample_load and not not_sample:
        if not np.random.binomial(1, probs[line_n]):
          skip_line_count += 1
          continue
      src_lens.append(len(src_tokens))
      trg_indices = [self.hparams.bos_id]
      add_indice = False 
      if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram:
        src_char_kv = [{0:0}]
      else:
        add_indice = True
      if ( not add_indice and self.hparams.uni) or add_indice:
        src_indices = [self.hparams.bos_id] 
      if self.hparams.semb_num > 1:
        src_ranks = [0]
      if self.hparams.uni:
        src_w2i = self.src_w2i_list[data_idx]
      elif self.hparams.char_ngram_n <= 0 and not self.hparams.bpe_ngram:
        src_w2i = self.src_w2i
      for src_tok in src_tokens:
        # calculate char ngram emb for src_tok
        add_indice = False 
        if self.hparams.char_ngram_n > 0:
          ngram_counts = self._get_ngram_counts(src_tok)
          src_char_kv.append(ngram_counts)
        elif self.hparams.bpe_ngram:
          ngram_counts = self._get_bpe_ngram_counts(src_tok, self.src_char_i2w, self.src_char_w2i)
          src_char_kv.append(ngram_counts)
        else:
          add_indice = True 
        if ( not add_indice and self.hparams.uni) or add_indice:
          if src_tok not in src_w2i:
            src_indices.append(self.hparams.unk_id)
            src_unk_count += 1
          else:
            src_indices.append(src_w2i[src_tok])
        if self.hparams.semb_num > 1:
          if src_tok in cur_src_w2i:
            cur_idx = cur_src_w2i[src_tok]
            #rank = cur_idx // (len(cur_src_w2i) // self.hparams.semb_num + 1)
            if data_idx < 0:
              rank = cur_idx // 2500
            else:
              rank = cur_idx // 5000
            rank = min(rank, self.hparams.semb_num-1)
            rank = rank % self.hparams.semb_num
          else:
            rank = self.hparams.semb_num - 1
          src_ranks.append(rank)

      for trg_tok in trg_tokens:
        if trg_tok not in self.trg_w2i:
          trg_indices.append(self.hparams.unk_id)
          trg_unk_count += 1
        else:
          trg_indices.append(self.trg_w2i[trg_tok])

      trg_indices.append(self.hparams.eos_id)
      trg_data.append(trg_indices)
      add_indice = False
      if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram:
        src_char_kv.append({0:0})
        src_char_kv_data.append(src_char_kv)
      else:
        add_indice = True
      if ( not add_indice and self.hparams.uni) or add_indice:
        src_indices.append(self.hparams.eos_id)
        src_data.append(src_indices)
      if self.hparams.semb_num > 1:
        src_ranks.append(0)
        src_word_rank.append(src_ranks)
      line_count += 1
      #if line_count == 20: break
      if outprint:
        if line_count % 10000 == 0:
          print("processed {} lines".format(line_count))

    if is_train and shuffle:
      src_data, trg_data, src_char_kv_data, src_word_rank = self.sort_by_xlen([src_data, trg_data, src_char_kv_data, src_word_rank], descend=False)
    if outprint:
      print("src_unk={}, trg_unk={}".format(src_unk_count, trg_unk_count))
      print("lines={}, skipped_lines={}".format(len(trg_data), skip_line_count))
    return src_data, trg_data, src_char_kv_data, src_lens, src_word_rank

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
    vfile_list = vocab_file_list
    if type(vocab_file_list) != list:
      vsize_list = [int(s) for s in vocab_size_list.split(",")]
    elif not vocab_size_list:
      vsize_list = [0 for i in range(len(vocab_file_list))]
    else:
      vsize_list = [int(vocab_size_list) for i in range(len(vocab_file_list))]
    while len(vsize_list) < len(vfile_list):
      vsize_list.append(vsize_list[-1])
    #if self.hparams.ordered_char_dict:
    i2w = [ '<unk>']
    i2w_set = set(i2w)
    if self.hparams.compute_ngram:
      i2w_base = ['<unk>']
      i2w_four = []
    for vfile, size in zip(vfile_list, vsize_list):
      cur_vsize = 0
      with open(vfile, 'r', encoding='utf-8') as f:
        for line in f:
          w = line.strip()
          if single_n and n and len(w) != n: continue
          if not single_n and n and len(w) > n: continue 
          if w == '<unk>' or w == '<pad>' or w == '<s>' or w == '<\s>': continue
          cur_vsize += 1
          if w not in i2w_set:
            if self.hparams.compute_ngram:
              if len(w) == 4:
                i2w_four.append(w)
              else:
                i2w_base.append(w)
            else:
              i2w.append(w)
              i2w_set.add(w)
            if size > 0 and cur_vsize > size: break
    if self.hparams.compute_ngram:
      i2w = i2w_base + i2w_four
      self.i2w_base = i2w_base
      self.i2w_four = i2w_four
      self.w2i_base = {}
      for i, w in enumerate(i2w_base):
        self.w2i_base[w] = i
      self.w2i_four = {}
      for i, w in enumerate(i2w_four):
        self.w2i_four[w] = i
    w2i = {}
    for i, w in enumerate(i2w):
      w2i[w] = i
    return i2w, w2i

