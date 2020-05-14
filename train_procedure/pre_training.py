from config import E
from prepare import get_data , get_model , get_others
from .train import train
from .evaluate import evaluate


def pretrain(C , data , num_epoch = 10):
	device = 0

	(trainset , devset , testset) , lab_num = get_data  (C , fold = data)
	model 									= get_model (C , lab_num)
	optimer , loss_func 					= get_others(C , model)

	for epoch_id in range(num_epoch):
		model = train(C, model, trainset, loss_func, optimer, epoch_id, "PT", device)
		droc_auc , dprc_auc = evaluate(C, model, devset , loss_func, epoch_id, "PT", device, "Dev")

	return model
