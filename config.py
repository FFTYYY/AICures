import argparse
from fitterlog.arg_proxy.arg_proxy import ArgProxy
import sys

def get_arg_proxy():
	prox = ArgProxy()

	# data
	prox.add_argument("data" 		, type = str , default = "pseudomonas")
	prox.add_argument("pos_aug" 	, type = int , default = 0)

	# features
	prox.add_store_true("finger")
	prox.add_store_true("mol2vec")

	# train & test
	prox.add_argument("dev_prop"	, type = float , default = 0.1)
	prox.add_argument("lr"			, type = float , default = 1e-3)
	prox.add_argument("num_epoch"	, type = int   , default = 20)
	prox.add_argument("bs"			, type = int   , default = 10)
	prox.add_store_true("uniform_sample")

	prox.add_store_true("no_valid")
	prox.add_store_true("train_loss_val")

	# model
	prox.add_argument("model" 		, type = str   	, default = "gcn")
	prox.add_argument("d" 	 		, type = int   	, default = 128)
	prox.add_argument("num_layers" 	, type = int   	, default = 2)
	prox.add_store_true("residual")
	prox.add_store_true("reinit")
	prox.add_store_true("layer_norm")
	prox.add_argument("ensemble" 	, type = int   	, default = 1)

	# regularization
	prox.add_argument("weight_decay", type = float	, default = 1e-8)
	prox.add_argument("grad_clip" 	, type = float	, default = -1)
	prox.add_argument("dropout" 	, type = float	, default = 0.0)
	prox.add_argument("activate" 	, type = str  	, default = "relu")
	prox.add_store_true("loss_weight")


	# others
	prox.add_argument("info" 		, type = str  	, default = "" , editable = True)
	prox.add_argument("group" 		, type = str  	, default = "default")
	prox.add_argument("seed" 		, type = int  	, default = 2333)
	prox.add_argument("pt_epoch" 	, type = int  	, default = 0)
	prox.add_store_true("pretrain")
	prox.add_argument("save_path" 	, type = str 	, default = "save_model/")

	return prox

