import sys
import os
import json
sys.path.append('../T2F/implementation')
from django.shortcuts import render
import torch as th
import numpy as np
#import data_processing.DataLoader as dl
import yaml
from networks.TextEncoder import Encoder
from networks.ConditionAugmentation import ConditionAugmentor
#from pro_gan_pytorch.PRO_GAN import ConditionalProGAN
from networks.TextEncoder import PretrainedEncoder
from .extractor import pro_gan

def create_grid(samples, scale_factor, img_file, real_imgs=False):
    from torchvision.utils import save_image
    from torch.nn.functional import interpolate

    samples = th.clamp((samples / 2) + 0.5, min=0, max=1)

    # upsample the image
    if not real_imgs and scale_factor > 1:
        samples = interpolate(samples,
                              scale_factor=scale_factor)

    # save the images:
    save_image(samples, img_file, nrow=int(np.sqrt(len(samples))))

# Create your views here.
def get_config(conf_file):
    from easydict import EasyDict as edict
    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/" + conf_file
    print("PATH")
    print(dir_path)
    with open(os.path.dirname(os.path.realpath(__file__)) + "/" + conf_file, "r") as file_descriptor:
        data = yaml.load(file_descriptor)

    # convert the data into an easyDictionary
    return edict(data)

def home(request):
    context = {}
    if(request.method=="POST"):
        caption = request.POST['description']

        try:
            # current_depth = 4
            # # define the device for the training script
            # device = th.device("cpu")
            #
            # ############################################################################
            # # load my generator.
            #
            # config = get_config('configs/colab.conf')
            #
            # dataset = dl.Face2TextDataset(
            #     pro_pick_file=os.path.dirname(os.path.realpath(__file__)) + '/saved_models/processed_text.pkl',
            #     img_dir=os.path.dirname(os.path.realpath(__file__)) + '/lfw',
            #     img_transform=dl.get_transform(config.img_dims),
            #     captions_len=config.captions_length
            # )
            # # dataset = dl.RawTextFace2TextDataset(
            # #     annots_file=os.path.dirname(os.path.realpath(__file__)) + '/saved_models/clean.json',
            # #     img_dir=config.images_dir,
            # #     img_transform=dl.get_transform(config.img_dims)
            # # )
            # text_encoder = PretrainedEncoder(
            #     model_file=os.path.dirname(os.path.realpath(__file__)) + '/saved_models/infersent2.pkl',
            #     embedding_file=os.path.dirname(os.path.realpath(__file__)) + '/saved_models/glove.840B.300d.txt',
            #     device='cpu'
            # )
            #
            # c_pro_gan = ConditionalProGAN(
            #     embedding_size=config.hidden_size,
            #     depth=config.depth,
            #     latent_size=config.latent_size,
            #     learning_rate=config.learning_rate,
            #     beta_1=config.beta_1,
            #     beta_2=config.beta_2,
            #     eps=config.eps,
            #     drift=config.drift,
            #     n_critic=config.n_critic,
            #     device=device
            # )
            # dir_path = os.path.dirname(os.path.realpath(__file__)) + "/" + 'saved_models/GAN_GEN_4.pth'
            # print("Loading Generator from:", dir_path)
            #
            # dic = th.load(dir_path, map_location=th.device(device))
            # print(dic.keys())
            #
            # dic["initial_block.conv_2.conv.bias"] = dic["initial_block.conv_2.bias"]
            # dic["layers.0.conv_1.conv.bias"] = dic["layers.0.conv_1.bias"]
            # dic["layers.0.conv_2.conv.bias"] = dic["layers.0.conv_2.bias"]
            # dic["layers.1.conv_1.conv.bias"] = dic["layers.1.conv_1.bias"]
            # dic["layers.1.conv_2.conv.bias"] = dic["layers.1.conv_2.bias"]
            # dic["layers.2.conv_1.conv.bias"] = dic["layers.2.conv_1.bias"]
            # dic["layers.2.conv_2.conv.bias"] = dic["layers.2.conv_2.bias"]
            # dic["layers.3.conv_1.conv.bias"] = dic["layers.3.conv_1.bias"]
            # dic["layers.3.conv_2.conv.bias"] = dic["layers.3.conv_2.bias"]
            # dic["layers.4.conv_1.conv.bias"] = dic["layers.4.conv_1.bias"]
            # dic["layers.4.conv_2.conv.bias"] = dic["layers.4.conv_2.bias"]
            # dic["layers.5.conv_1.conv.bias"] = dic["layers.5.conv_1.bias"]
            # dic["layers.5.conv_2.conv.bias"] = dic["layers.5.conv_2.bias"]
            # dic["rgb_converters.0.conv.bias"] = dic["rgb_converters.0.bias"]
            # dic["rgb_converters.1.conv.bias"] = dic["rgb_converters.1.bias"]
            # dic["rgb_converters.2.conv.bias"] = dic["rgb_converters.2.bias"]
            # dic["rgb_converters.3.conv.bias"] = dic["rgb_converters.3.bias"]
            # dic["rgb_converters.4.conv.bias"] = dic["rgb_converters.4.bias"]
            # dic["rgb_converters.5.conv.bias"] = dic["rgb_converters.5.bias"]
            # dic["rgb_converters.6.conv.bias"] = dic["rgb_converters.6.bias"]
            #
            # # del dic["module.initial_block.conv_2.bias"]
            # # del dic["module.layers.0.conv_1.bias"]
            # # del dic["module.layers.0.conv_2.bias"]
            # # del dic["module.layers.1.conv_1.bias"]
            # # del dic["module.layers.1.conv_2.bias"]
            # # del dic["module.layers.2.conv_1.bias"]
            # # del dic["module.layers.2.conv_2.bias"]
            # # del dic["module.layers.3.conv_1.bias"]
            # # del dic["module.layers.3.conv_2.bias"]
            # # del dic["module.layers.4.conv_1.bias"]
            # # del dic["module.layers.4.conv_2.bias"]
            # # del dic["module.layers.5.conv_1.bias"]
            # # del dic["module.layers.5.conv_2.bias"]
            # # del dic["module.rgb_converters.0.bias"]
            # # del dic["module.rgb_converters.1.bias"]
            # # del dic["module.rgb_converters.2.bias"]
            # # del dic["module.rgb_converters.3.bias"]
            # # del dic["module.rgb_converters.4.bias"]
            # # del dic["module.rgb_converters.5.bias"]
            # # del dic["module.rgb_converters.6.bias"]
            #
            # c_pro_gan.gen.load_state_dict(dic)
            #
            # ###################################################################################
            # # load my embedding and conditional augmentor
            #
            # condition_augmenter = ConditionAugmentor(
            #     input_size=config.hidden_size,
            #     latent_size=config.ca_out_size,
            #     use_eql=config.use_eql,
            #     device=device
            # )
            #
            # ca_file = os.path.dirname(os.path.realpath(__file__)) + '/saved_models/Condition_Augmentor_4.pth'
            #
            # print("Loading conditioning augmenter from:", ca_file)
            # ca_dict = th.load(ca_file, map_location=lambda storage, loc: storage)
            # print(ca_dict.keys())
            # ca_dict["transformer.linear.weight"] = ca_dict["transformer.weight"]
            # del ca_dict["transformer.weight"]
            # ca_dict["transformer.linear.bias"] = ca_dict["transformer.bias"]
            # condition_augmenter.load_state_dict(ca_dict)
            #
            # seq = []
            # for word in caption.split(' '):
            #     print(word)
            #     seq.append(dataset.rev_vocab[word])
            # for i in range(len(seq), 100):
            #     seq.append(0)
            #
            # seq = th.LongTensor(seq)
            # seq = seq.cpu()
            # print(type(seq))
            # print('\nInput : ', caption)
            #
            # list_seq = [seq for i in range(16)]
            # print(list_seq)
            # print(len(list_seq))
            # list_seq = th.stack(list_seq)
            # list_seq = list_seq.cpu()
            #
            # embeddings = text_encoder(caption)
            #
            # fixed_embeddings = th.from_numpy(embeddings).to(device)
            # print(fixed_embeddings)
            #
            # c_not_hats, mus, sigmas = condition_augmenter(fixed_embeddings)
            #
            # z = th.zeros(len(caption),
            #              c_pro_gan.latent_size - c_not_hats.shape[-1]).to(device)
            #
            # print(z)
            # print(len(caption))
            context['result'] = pro_gan(caption)
            return render(request, "index.html", context)

            #
            # gan_input = th.cat((c_not_hats, z), dim=-1)
            #
            # alpha = 0.007352941176470588
            #
            # samples = c_pro_gan.gen(gan_input,
            #                         current_depth,
            #                         alpha)
            #
            # from torchvision.utils import save_image
            # from torch.nn.functional import upsample
            # # from train_network import create_grid
            #
            # img_file = "static\\" + caption + '.png'
            # samples = (samples / 2) + 0.5
            # if int(np.power(2, c_pro_gan.depth - current_depth - 1)) > 1:
            #     samples = upsample(samples, scale_factor=current_depth)
            #
            # # save image to the disk, the resulting image is <caption>.png
            # save_image(samples, img_file, nrow=int(np.sqrt(20)))
            #
            # ###################################################################################
            # # #output the image.
            #
            # result = "\\static\\" + caption + ".png"
        except Exception as e:
            print(e)

    return render(request,"index.html",context)
