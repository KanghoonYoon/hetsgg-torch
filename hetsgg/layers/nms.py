
from hetsgg import _C

from apex import amp

nms = amp.float_function(_C.nms)


