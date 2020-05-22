from entry import E
from prepare import get_data , get_model , get_others
from .train import train
from .evaluate import evaluate
from utils import copy_param

def kfold(C , k = 10 , choose_one = [] , p_model = None):
	device = 0

	metric = "PRC-AUC"

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
			model = train(C, model, trainset, loss_func, optimer, epoch_id, run_id, device)
			droc_auc , dprc_auc = evaluate(C, model, devset , loss_func, epoch_id, run_id, device, "Dev")
			troc_auc , tprc_auc = evaluate(C, model, testset, loss_func, epoch_id, run_id, device, "Test")

			E.log("Epoch %d of run %d ended." % (epoch_id , run_id))
			E.log("Dev  Roc-Auc = %.4f Prc-Auc = %.4f" % (droc_auc , dprc_auc))
			E.log("Test Roc-Auc = %.4f Prc-Auc = %.4f" % (troc_auc , tprc_auc))
			E.log()

			if best_epoch < 0 or dprc_auc > best_metric or C.no_valid: #no_valid：总是更新最佳epoch
				best_epoch 	= epoch_id
				best_metric = dprc_auc
				tes_roc_auc = troc_auc
				tes_prc_auc = tprc_auc

		E.log("%d th run ends. best epoch = %d" % (run_id , best_epoch))
		E.log("Best Dev %s = %.4f"                     % (metric , best_metric))
		E.log("Got Test Roc-Auc = %.4f Prc-Auc = %.4f" % (tes_roc_auc , tes_prc_auc))
		E.log()

		E["Dev %s" % metric]["Best"].update(best_metric , run_id)
		E["Test ROC-AUC"]["Best"].update(tes_roc_auc , run_id)
		E["Test PRC-AUC"]["Best"].update(tes_prc_auc , run_id)

		roc_aucs.append(tes_roc_auc)
		prc_aucs.append(tes_prc_auc)
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
