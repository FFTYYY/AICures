import torch as tc
import fitlog
from tqdm import tqdm
import pdb
from entry import E
import random
import torch.nn as nn

class Sampler:
	def __init__(self , set):
		self.set = set
		self.pos_set = [x for x in set if x[1] == 1]
		self.neg_set = [x for x in set if x[1] == 0]

	def normal_sample(self , num):
		return random.sample(self.set , num)

	def sample(self , num):
		num_pos = num // 2
		poss = random.sample(self.pos_set , num_pos)
		negs = random.sample(self.neg_set , num - num_pos)
		return poss + negs

def train(C , model , dataset , loss_func , optimer , epoch_id , run_id , device , finger_dict = None):
	model = model.train()

	batch_num = (len(dataset) // C.bs) + int(len(dataset) % C.bs != 0)
	sampler = Sampler(dataset)

	ac_losses = 0
	pbar = tqdm(range(batch_num) , ncols = 130 , desc = "[{0}]Training. Epoch {1}".format(run_id , epoch_id))
	for step , batch_id in enumerate(pbar):

		if C.uniform_sample:
			bdata  = dataset[batch_id * C.bs : (batch_id+1) * C.bs]
		else:
			bdata = sampler.sample(C.bs) 
		gs 	   = [d[0] for d in bdata]
		smiles = [d[2] for d in bdata]
		labels = [d[1] for d in bdata]
		if finger_dict:
			fingers = [finger_dict.get(s , [0] * C.finger_size) for s in smiles ]
			fingers = tc.LongTensor(fingers).cuda(device) #(bs , 1024)
		else:
			fingers = None

		pred  = model(gs , smiles = smiles , fingers = fingers)
		labels = tc.LongTensor(labels).cuda(device)

		loss = loss_func(pred , labels)
		optimer.zero_grad()
		loss.backward()
		if C.grad_clip > 0:
			nn.utils.clip_grad_value_(model.parameters(), C.grad_clip)
		optimer.step()

		ac_losses += float(loss)

		pbar.set_postfix_str("recent loss = %.4f" % (ac_losses / (step + 1)))

	loss = ac_losses / (step + 1)
	E["Train Loss"][str(run_id)].update(loss , epoch_id)

	return model , loss
