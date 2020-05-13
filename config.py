from fitterlog.interface import new_or_load_experiment
from fitterlog.arg_proxy.arg_proxy import ArgProxy


def get_prox():
	prox = ArgProxy()

	# data
	prox.add_argument("data" 		, type = str   , default = "pseudomonas")

	# train & test
	prox.add_argument("dev_prop"	, type = float , default = 0.1)
	prox.add_argument("lr"			, type = float , default = 1e-2)
	prox.add_argument("num_epoch"	, type = int   , default = 10)
	prox.add_argument("bs"			, type = int   , default = 10)

	# model
	prox.add_argument("model" 		, type = str   , default = "gcn")
	prox.add_argument("d" 	 		, type = int   , default = 128)
	prox.add_argument("num_layers" 	, type = int   , default = 2)

	# others
	prox.add_argument("info" 		, type = str   , default = "" , editable = True)
	prox.add_argument("group" 		, type = str   , default = "default")

	return prox

prox = get_prox()

C = prox.assign_from_cmd()

E = new_or_load_experiment(project_name = "PRML" , group_name = C.group)
E.use_argument_proxy(prox)
