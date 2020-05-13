import torch as tc
import fitlog
from tqdm import tqdm
import pdb
from config import E

def train(C , model , dataset , loss_func , optimer , epoch_id , run_id , device):
	model = model.train()

	batch_num = (len(dataset) // C.bs) + int(len(dataset) % C.bs != 0)

	ac_losses = 0
	pbar = tqdm(range(batch_num) , ncols = 130 , desc = "[%d]Traning. Epoch %d" % (run_id , epoch_id))
	for step , batch_id in enumerate(pbar):

		bdata  = dataset[batch_id * C.bs : (batch_id+1) * C.bs]
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

	E["Train Loss"][str(run_id)].update((ac_losses / (step + 1)) , epoch_id)

	return model
