from config import C , E
from prepare import get_data , get_model , get_others
import pdb

def main():
	(trainset , devset , testset) , lab_num = get_data  (C)
	model 									= get_model (C , lab_num)
	optimer , loss_func 					= get_others(C , model)

	pdb.set_trace()


if __name__ == "__main__":
	main()

	E.finish()