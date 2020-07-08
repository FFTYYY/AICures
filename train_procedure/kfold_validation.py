from entry import E
from prepare import get_data , get_model , get_others
from prepare.dataloader import load_fingers
from .train import train
from .evaluate import evaluate
from utils import copy_param , save_model

def kfold(C , k = 10 , choose_one = [] , p_model = None):

	if C.finger or C.mol2vec:
		finger_dict = load_fingers(C , C.data)

	device = 0

	roc_aucs 	= []
	prc_aucs 	= []
	for run_id in range(k):

		if len(choose_one) > 0 and run_id not in choose_one: #只跑选择的那一个
			continue

		(trainset , devset , testset) , lab_num = get_data  (C , fold = run_id)

		model = get_model (C , lab_num)
		if p_model is not None:
			copy_param(model , p_model)
		model = model.to(device)

		optimer , loss_func = get_others(C , model)

		E.log("%d th run starts on device %d\n" % (run_id , device))

		best_epoch	= -1
		best_metric = -1
		tes_roc_auc = -1
		tes_prc_auc = -1
		for epoch_id in range(C.num_epoch):
			model , train_loss = train(C, model, trainset, loss_func, optimer, epoch_id, run_id, device , finger_dict)
			droc_auc , dprc_auc = evaluate(C, model, devset , loss_func, epoch_id, run_id, device, "Dev" , finger_dict)
			troc_auc , tprc_auc = evaluate(C, model, testset, loss_func, epoch_id, run_id, device, "Test" , finger_dict)

			E.log("Epoch %d of run %d ended." % (epoch_id , run_id))
			E.log("Dev  Roc-Auc = %.4f Prc-Auc = %.4f" % (droc_auc , dprc_auc))
			E.log("Test Roc-Auc = %.4f Prc-Auc = %.4f" % (troc_auc , tprc_auc))
			E.log()

			if C.train_loss_val:
				metric_val = -train_loss
			else:
				metric_val = dprc_auc

			if (best_epoch < 0 or metric_val > best_metric) or C.no_valid:
				best_epoch 	= epoch_id
				best_metric = metric_val
				tes_roc_auc = troc_auc
				tes_prc_auc = tprc_auc
				save_model(model , C.save_path , E.core.id , str(run_id))

		E.log("%d th run ends. best epoch = %d" % (run_id , best_epoch))
		E.log("Best metric = %.4f"                     % (best_metric))
		E.log("Got Test Roc-Auc = %.4f Prc-Auc = %.4f" % (tes_roc_auc , tes_prc_auc))
		E.log()

		E["Test ROC-AUC"]["Best"].update(tes_roc_auc , run_id)
		E["Test PRC-AUC"]["Best"].update(tes_prc_auc , run_id)

		roc_aucs.append(tes_roc_auc)
		prc_aucs.append(tes_prc_auc)

		E.log("model saved.")

		E.log("--------------------------------------------------------------")

	roc_auc_avg = sum(roc_aucs) / len(roc_aucs)
	roc_auc_std = (sum([(x - roc_auc_avg) ** 2 for x in roc_aucs]) / len(roc_aucs)) ** 0.5
	prc_auc_avg = sum(prc_aucs) / len(prc_aucs)
	prc_auc_std = (sum([(x - prc_auc_avg) ** 2 for x in prc_aucs]) / len(prc_aucs)) ** 0.5

	E["Test ROC-AUC"].update("%.4f ± %.4f" % (roc_auc_avg , roc_auc_std))
	E["Test PRC-AUC"].update("%.4f ± %.4f" % (prc_auc_avg , prc_auc_std))
	E.log ("got avg test Roc-Auc = %.4f ± %.4f Prc-Auc = %.4f ± %.4f" % (
		roc_auc_avg , roc_auc_std , prc_auc_avg , prc_auc_std)
	)

	
	E.log("All run end!")
