import os
import time
from argparse import Namespace

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed

from src_files.helper_functions.bn_fusion import fuse_bn_recursively
from src_files.models import create_model

from src_files.models.tresnet.tresnet import InplacABN_to_ABN

from PIL import Image
import numpy as np

model_args = Namespace(
    num_classes=80,  # Default: 80
    model_path='./models/tresnet_l_COCO__448_90_0.pth',  # Default model path
    # pic_path='./pics/000000000885.jpg',  # Default image path
    model_name='tresnet_l',  # Default model name
    # image_size=448,  # Default image size (448)
    # th=0.75,  # Default threshold (0.75)
    # top_k=20,  # Default top-k value (20)
    use_ml_decoder=1,  # Default: 1 (ML decoder enabled)
    num_of_groups=-1,  # Default: -1 (full-decoding)
    decoder_embedding=768,  # Default decoder embedding (768)
    zsl=0  # Default: 0 (zero-shot learning disabled)
)

image_args = Namespace(
    image_size=448,  # Default image size (448)
)

model = None
classes_list = None
th = 0.75  # Default threshold (0.75)
top_k = 20  # Default top-k value (20)


def init():
    global model, classes_list
    # Setup model
    print('creating model {}...'.format(model_args.model_name))
    model = create_model(model_args)
    state = torch.load(model_args.model_path, map_location='cpu')
    classes_list = np.array(list(state['idx_to_class'].values()))
    model.load_state_dict(state['model'], strict=True)
    ########### eliminate BN for faster inference ###########
    model = model.cpu()
    model = InplacABN_to_ABN(model)
    model = fuse_bn_recursively(model)
    model = model.cuda().half().eval()
    #######################################################
    print('done')


def serve(pic_path):
    if model is None:
        init()

    t0 = time.time()

    im = Image.open(pic_path)
    im_resize = im.resize((image_args.image_size, image_args.image_size))
    np_img = np.array(im_resize, dtype=np.uint8)
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
    tensor_batch = torch.unsqueeze(tensor_img, 0).cuda().half()  # float16 inference
    output = torch.squeeze(torch.sigmoid(model(tensor_batch)))
    np_output = output.cpu().detach().numpy()

    ## Top-k predictions
    # detected_classes = classes_list[np_output > args.th]
    idx_sort = np.argsort(-np_output)
    detected_classes = np.array(classes_list)[idx_sort][: top_k]
    scores = np_output[idx_sort][: top_k]

    t1 = time.time()

    cost_time = t1 - t0

    return [{cls, score} for cls, score in zip(detected_classes, scores)], cost_time
    # idx_th = scores > th
    # detected_classes = detected_classes[idx_th]
