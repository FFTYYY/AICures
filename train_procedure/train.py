import torch as tc
import fitlog
from tqdm import tqdm
import pdb
from entry import E
import random

class Sampler:
	def __init__(self , set):
		self.set = set
		self.pos_set = [x for x in set if x[1] == 1]
		self.neg_set = [x for x in set if x[1] == 0]
	def sample(self , num):
		num_pos = num // 2
		poss = random.sample(self.pos_set , num_pos)
		negs = random.sample(self.neg_set , num - num_pos)
		return poss + negs

def train(C , model , dataset , loss_func , optimer , epoch_id , run_id , device):
	model = model.train()

	batch_num = (len(dataset) // C.bs) + int(len(dataset) % C.bs != 0)
	sampler = Sampler(dataset)

	ac_losses = 0
	pbar = tqdm(range(batch_num) , ncols = 130 , desc = "[{0}]Training. Epoch {1}".format(run_id , epoch_id))
	for step , batch_id in enumerate(pbar):

		bdata = sampler.sample(C.bs) 
		#bdata  = dataset[batch_id * C.bs : (batch_id+1) * C.bs]
		gs 	   = [d[0] for d in bdata]
		labels = [d[1] for d in bdata]

		pred  = model(gs)
		labels = tc.LongTensor(labels).cuda(device)

		loss = loss_func(pred , labels)
		optimer.zero_grad()
		loss.backward()
		optimer.step()

		ac_losses += float(loss)

		pbar.set_postfix_str("recent loss = %.4f" % (ac_losses / (step + 1)))

	loss = ac_losses / (step + 1)
	E["Train Loss"][str(run_id)].update(loss , epoch_id)

	return model , loss
