from .. import backend as F
from .. import ndarray
from .. import utils
from .._ffi.function import _init_api


def PinOMPThreads(x):
    x = F.to_dgl_nd(F.tensor(x, dtype=F.int64))
    _CAPI_SetOMPThreadAffinity(x)

def FakeGompAffinity(x):
    x = F.to_dgl_nd(F.tensor(x, dtype=F.int64))
    _CAPI_FakeGompAffinity(x)

_init_api("dgl.dataloading.cpu_affinity")