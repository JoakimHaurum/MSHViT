import os
from argparse import ArgumentParser
from collections import OrderedDict
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

def extractWeights(args):
    model_path = args["model_path"]
    output_dir = args["output_dir"]

    ckpt_path = os.path.join(model_path, "last.ckpt")

    if not os.path.isfile(ckpt_path):
        raise ValueError("The provided directory path does not contain a 'last.ckpt' file: {}".format(model_path))
    
    model_last_ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

    model_ckpt_hparams = vars(model_last_ckpt["hyper_parameters"]["args"])
    sigmoid_loss = model_last_ckpt["hyper_parameters"]["sigmoid_loss"]
    num_classes = model_last_ckpt["hyper_parameters"]["num_classes"]

    backbone_hparams =  {"backbone_model": model_ckpt_hparams["backbone_model"],
                        "img_size": model_ckpt_hparams["img_size"],
                        "backbone_feature_maps": model_ckpt_hparams["backbone_feature_maps"],
                        "pretrained_backbone": model_ckpt_hparams["pretrained_backbone"]}

    head_model = model_ckpt_hparams["head_model"]
    if head_model == "BaseHead":
        head_hparams = {"head_model":model_ckpt_hparams["head_model"],
                        "num_classes": num_classes,
                        "sigmoid_loss": sigmoid_loss,
                        "global_pool": model_ckpt_hparams["global_pool"],
                        "tresnet_init": model_ckpt_hparams["tresnet_init"]}
        
    elif "MultiScaleViTHead" in head_model:
        head_hparams = {"head_model": "MultiScaleViTHead",
                        "num_classes": num_classes,
                        "sigmoid_loss": sigmoid_loss,
                        "tresnet_init": model_ckpt_hparams["tresnet_init"],
                        "backbone_feature_maps": model_ckpt_hparams["backbone_feature_maps"],
                        "patch_size": model_ckpt_hparams["patch_size"],
                        "num_clusters": model_ckpt_hparams["num_clusters"],
                        "l2_normalize": model_ckpt_hparams["l2_normalize"],
                        "sinkhorn_eps": model_ckpt_hparams["sinkhorn_eps"],
                        "sinkhorn_iters": model_ckpt_hparams["sinkhorn_iters"],
                        "tokenizer_layer": model_ckpt_hparams["tokenizer_layer"],
                        "num_heads": model_ckpt_hparams["num_heads"],
                        "qkv_bias": model_ckpt_hparams["qkv_bias"],
                        "mlp_ratio": model_ckpt_hparams["mlp_ratio"],
                        "proj_drop": model_ckpt_hparams["proj_drop"],
                        "attn_drop": model_ckpt_hparams["attn_drop"],
                        "cross_num_heads": model_ckpt_hparams["cross_num_heads"],
                        "cross_qkv_bias": model_ckpt_hparams["cross_qkv_bias"],
                        "cross_mlp_ratio": model_ckpt_hparams["cross_mlp_ratio"],
                        "cross_proj_drop": model_ckpt_hparams["cross_proj_drop"],
                        "cross_attn_drop": model_ckpt_hparams["cross_attn_drop"],
                        "cross_block_drop": model_ckpt_hparams["cross_block_drop"],
                        "cross_block_type": model_ckpt_hparams["cross_block_type"],
                        "cross_scale_all": model_ckpt_hparams["cross_scale_all"],
                        "shared_tower": model_ckpt_hparams["shared_tower"],
                        "shared_tokenizer": model_ckpt_hparams["shared_tokenizer"],
                        "late_fusion": model_ckpt_hparams["late_fusion"],
                        "multiscale_method": model_ckpt_hparams["multiscale_method"],
                        "no_vit_layers": model_ckpt_hparams["no_vit_layers"],
                        "token_dim": model_ckpt_hparams["token_dim"],
                        "representation_size": model_ckpt_hparams["representation_size"],
                        "transformer_depth": model_ckpt_hparams["transformer_depth"],
                        "block_type": model_ckpt_hparams["block_type"],
                        "block_drop": model_ckpt_hparams["block_drop"],
                        "use_pos_embed": model_ckpt_hparams["use_pos_embed"],
                        "pos_embed_drop": model_ckpt_hparams["pos_embed_drop"],
                        "norm_layer": model_ckpt_hparams["norm_layer"],
                        "act_layer": model_ckpt_hparams["act_layer"],
                        "use_mean_token": model_ckpt_hparams["use_mean_token"]}

    # Load best checkpoint
    best_model_path = model_last_ckpt["callbacks"][ModelCheckpoint]["best_model_path"]
    best_model = torch.load(best_model_path, map_location="cpu")
    model_state_dict = best_model["state_dict"]

    updated_backbone_state_dict = OrderedDict()
    updated_head_state_dict = OrderedDict()

    for k,v in model_state_dict.items():
        if "criterion" in k:
            continue
        if "head" in k:
            if "head_ema" in k:
                continue
            name = k.replace("head.", "")
            updated_head_state_dict[name] = v
        if "backbone" in k:
            if "backbone_cls" in k:
                continue
            if "backbone_ema" in k:
                continue
            elif "backbone.backbone." in k:
                name = k.replace("backbone.backbone.", "backbone.")
            else:
                name = k.replace("backbone.", "")
            updated_backbone_state_dict[name] = v

    updated_backbone_state_dict_ema = OrderedDict()
    updated_head_state_dict_ema = OrderedDict()
    for k,v in model_state_dict.items():
            if "criterion" in k:
                continue
            if "backbone_ema" in k:
                name = k.replace("backbone_ema.module.", "")
                updated_backbone_state_dict_ema[name] = v
            if "head_ema" in k:
                name = k.replace("head_ema.module.", "")
                updated_head_state_dict_ema[name] = v


    new_dir = os.path.join(os.path.basename(model_path))

    tmp_dict_pytorch = {"state_dict_backbone": updated_backbone_state_dict,
                        "state_dict_head": updated_head_state_dict,
                        "state_dict_backbone_ema": updated_backbone_state_dict_ema,
                        "state_dict_head_ema": updated_head_state_dict_ema,
                        "hyper_parameters_backbone": backbone_hparams,
                        "hyper_parameters_head":head_hparams}

    torch.save(tmp_dict_pytorch, os.path.join(output_dir, "Pytorch", new_dir))

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--output_dir", type=str)

    args = vars(parser.parse_args())

    output_dir = args["output_dir"]
    if not os.path.isdir(os.path.join(output_dir, "Pytorch")):
        os.makedirs(os.path.join(output_dir, "Pytorch"))

    extractWeights(args)

    