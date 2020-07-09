from entry import C , E
from train_procedure.kfold_validation import kfold , eval_run
from train_procedure.pre_training import pretrain
from YTools.experiment_helper.logger import Logger
import pdb

def main():

	logger = Logger(mode = [print , E.add_line])
	E.log = logger.log

	E.new_variable("Dev ROC-AUC")
	E.new_variable("Dev PRC-AUC")
	E.new_variable("Test ROC-AUC")
	E.new_variable("Test PRC-AUC")
	E.new_variable("Train Loss")
	E.new_variable("Dev Loss")
	E.new_variable("Test Loss")

	model = None
	if C.pretrain:
		model = pretrain(C , ["AID1706_binarized_sars_scaffold/train.csv"] , C.pt_epoch , 300 , int(300 * (2 ** C.pos_aug)))

	if C.eval:
		eval_run(C , p_model = model)
	else:
		kfold(C , p_model = model)


if __name__ == "__main__":
	main()

	E.finish()