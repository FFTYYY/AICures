import torch as tc
from tqdm import tqdm
import pdb
from config import E

def evaluate(C , model , dataset , loss_func , epoch_id , run_id , device , eval_name):
	model = model.eval()
	batch_num = (len(dataset) // C.bs) + int(len(dataset) % C.bs != 0)

	ac_loss = 0
	ac_ghit = 0
	pbar = tqdm(range(batch_num) , ncols = 130 , desc = "[%d]%sing. Epoch %d" % (
		run_id , eval_name , epoch_id
	))
	for step , batch_id in enumerate(pbar):

		bdata  = dataset[batch_id * C.bs : (batch_id+1) * C.bs]
		gs 	   = [d[0] for d in bdata]
		labels = [d[1] for d in bdata]

		with tc.no_grad():
			pred  = model(gs)
		labels = tc.LongTensor(labels).cuda(device)

		ac_loss += float(loss_func(pred , labels))
		ac_ghit += int((pred.max(-1)[1] == labels).long().sum())

		pbar.set_postfix_str("avg loss = %.4f , acc = %.2f%% (%d / %d)" % (
			ac_loss / (step+1) , 100 * ac_ghit / (step+1) , ac_ghit , step+1 , 
		))

	E[eval_name + " Loss"][str(run_id)].update(ac_loss / len(dataset) , epoch_id)
	E[eval_name + " Acc" ][str(run_id)].update(ac_ghit / len(dataset) , epoch_id)

	return ac_ghit / len(dataset)