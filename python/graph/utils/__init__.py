from .loop import add_remaining_self_loops, contains_self_loops
from .loop import remove_self_loops, add_self_loops
from .scatter import scatter_add, scatter_mean
from .inits import glorot, zeros, uniform
from .sparse import spmm

__all__ = [
	'scatter_add',
	'scatter_mean',
	'add_remaining_self_loops',
	"contains_self_loops",
	"remove_self_loops",
	"add_self_loops",
	'glorot',
	'zeros', 
	'uniform',
	'spmm'
]