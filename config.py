import argparse
from fitterlog.arg_proxy.arg_proxy import ArgProxy
import sys

def get_arg_proxy():
	prox = ArgProxy()

	# data
	prox.add_argument("data" 		, type = str   , default = "pseudomonas")

	# train & test
	prox.add_argument("dev_prop"	, type = float , default = 0.1)
	prox.add_argument("lr"			, type = float , default = 1e-2)
	prox.add_argument("num_epoch"	, type = int   , default = 20)
	prox.add_argument("bs"			, type = int   , default = 10)

	prox.add_store_true("no_valid")
	prox.add_store_true("train_loss_val")

	# model
	prox.add_argument("model" 		, type = str   , default = "gcn")
	prox.add_argument("d" 	 		, type = int   , default = 128)
	prox.add_argument("num_layers" 	, type = int   , default = 2)
	prox.add_store_true("residual")
	prox.add_store_true("reinit")

	# others
	prox.add_argument("info" 		, type = str   , default = "" , editable = True)
	prox.add_argument("group" 		, type = str   , default = "default")
	prox.add_argument("seed" 		, type = int   , default = 2333)
	prox.add_argument("pt_epoch" 	, type = int   , default = 1)
	prox.add_store_true("pretrain")

	return prox

