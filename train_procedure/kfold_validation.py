from config import E
from prepare import get_data , get_model , get_others
from .train import train
from .evaluate import evaluate

def kfold(C , k = 10 , choose_one = -1):
	device = 0

	ac_best_acc 	= 0.
	ac_best_e_tacc 	= 0.
	useful_run 		= 0
	for run_id in range(k):

		if choose_one >= 0 and run_id != choose_one: #只跑选择的那一个
			continue

		(trainset , devset , testset) , lab_num = get_data  (C , fold = run_id)
		model 									= get_model (C , lab_num)
		optimer , loss_func 					= get_others(C , model)


		E.log("%d th run starts on device %d\n" % (run_id , device))

		best_epoch	= -1
		best_acc 	= -1
		best_e_tacc = -1
		for epoch_id in range(C.num_epoch):
			model = train   (C , model , trainset , loss_func , optimer , epoch_id , run_id , 
				device = device , )
			acc   = evaluate(C , model , devset   , loss_func ,           epoch_id , run_id , 
				device = device , eval_name = "Dev")
			tacc  = evaluate(C , model , testset  , loss_func ,           epoch_id , run_id , 
				device = device , eval_name = "Test")

			E.log("Epoch %d of run %d ended. Dev acc = %.2f%% Test acc = %.2f%%\n" % (
				epoch_id , run_id , 100 * tacc , 100 * acc , 
			))

			if best_epoch < 0 or acc > best_acc:
				best_epoch 	= epoch_id
				best_acc 	= acc
				best_e_tacc = tacc

		E.log("%d th run ends. best epoch = %d , best dev acc = %.2f%% , got test acc = %.2f%%" % (
			run_id , best_epoch , best_acc , best_e_tacc , 
		))

		E["Test Acc"]["Best"].update(best_e_tacc , run_id)
		E["Dev Acc" ]["Best"].update(best_acc    , run_id)
		ac_best_acc 	+= best_acc
		ac_best_e_tacc 	+= best_e_tacc
		useful_run 		+= 1
		E.log("--------------------------------------------------------------")

	E["Dev Acc" ].update(ac_best_acc    / useful_run)
	E["Test Acc"].update(ac_best_e_tacc / useful_run)
	E.log ("avg best dev acc is %.2f%%" % (ac_best_acc    / useful_run))
	E.log ("avg got test acc is %.2f%%" % (ac_best_e_tacc / useful_run))

	
	E.log("All run end!")
