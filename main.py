from config import C , E
from train_procedure.kfold_validation import kfold
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


	kfold(C)


if __name__ == "__main__":
	main()

	E.finish()