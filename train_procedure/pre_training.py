from entry import E
from prepare import get_data , get_model , get_others
from .train import train
from .evaluate import evaluate


def pretrain(C , data , num_epoch = 10 , pos_lim = 300, neg_lim = 300 , model = None):
	device = 0

	(trainset , devset , testset) , lab_num = get_data  (C , fold = data , files = True , pos_lim = pos_lim , neg_lim = neg_lim)
	if model is None:
		model 								= get_model (C , lab_num)
	optimer , loss_func 					= get_others(C , model)

	for epoch_id in range(num_epoch):
		model = train(C, model, trainset, loss_func, optimer, epoch_id, "PT", device)

	return model
