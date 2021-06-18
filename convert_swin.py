from models import build_model
from config import get_config
import torch
import sys
sys.path.append('torch2trt')
from torch2trt import torch2trt, tensorrt_converter, add_missing_trt_tensors, trt

def load_config(path):
    class Tmp():
        def __init__(self, path):
            self.cfg = path

        def __getattribute__(self, name):
            try:
                return object.__getattribute__(self, name)
            except:
                return None
    return get_config(Tmp(path))

data = torch.randn(1,3,224,224).cuda()
model = build_model(load_config('configs/swin_tiny_patch4_window7_224.yaml')).cuda().eval()
model.load_state_dict(torch.load('swin_tiny_patch4_window7_224.pth')['model'])

@tensorrt_converter('torch.matmul')
@tensorrt_converter('torch.Tensor.__matmul__')
def convert_matmul(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    output = ctx.method_return
    layer = ctx.network.add_matrix_multiply(input_a_trt, trt.MatrixOperation.NONE, input_b_trt, trt.MatrixOperation.NONE)
    output._trt = layer.get_output(0)

model_trt = torch2trt(model, [data], log_level=trt.Logger.INFO)