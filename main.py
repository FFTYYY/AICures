from entry import C , E
from train_procedure.kfold_validation import kfold
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
		# model = pretrain(C , ["ecoli_scaffold/train.csv"] , C.pt_epoch , 90 , 90 , model = model)
		# model = pretrain(C , ["the_data/bace.csv"] , C.pt_epoch , model = model)
		# model = pretrain(C , "the_data/bbbp.csv" , C.pt_epoch , 500 , 500 , model = model)
		# model = pretrain(C , "the_data/hiv.csv" , C.pt_epoch , 1400 , 1400 , model = model)
	kfold(C , p_model = model)


if __name__ == "__main__":
	main()

	E.finish()