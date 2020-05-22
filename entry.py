from fitterlog.interface import new_or_load_experiment
from YTools.experiment_helper import set_random_seed
from config import get_arg_proxy

prox = get_arg_proxy()

C = prox.assign_from_cmd()

E = new_or_load_experiment(project_name = "PRML" , group_name = C.group)
E.use_argument_proxy(prox)

if C.seed > 0:
	set_random_seed(C.seed)
