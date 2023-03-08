import os
import pandas as pd
import numpy as np
from argparse import Namespace
from argparse import ArgumentParser

import torch
import torch.nn as nn

import backbones
import heads
import blocks
import dataloader
import transforms

def evaluate(dataloader, backbone, head, act_func, device):
    backbone.eval()
    head.eval()

    predictions = None
    first = True

    imgPathsList = []

    dataLen = len(dataloader)
    print(dataLen)
    
    with torch.no_grad():
        for i, (images, _, imgPaths) in enumerate(dataloader):
            if i % 100 == 0:
                print("{} / {}".format(i, dataLen))

            images = images.to(device)

            feat_map = backbone(images)
            outputs = head(feat_map, True, True)

            classOutput = act_func(outputs).detach().cpu().numpy()

            if first:
                predictions = classOutput
                first = False	
            else:
                predictions = np.vstack((predictions, classOutput))

            imgPathsList.extend(list(imgPaths))
    return predictions, imgPathsList


def process_predictions(dataset, predictions, image_paths, label_names = None, top_k = None):

    prediction_dict = {}
    prediction_dict["Filename"] = image_paths

    if dataset in dataloader.ML_Datasets and label_names is not None:
        for idx, header in enumerate(label_names):
            prediction_dict[header] = predictions[:,idx]
    elif dataset in dataloader.MC_Datasets and top_k is not None:
        predictions = predictions.argsort(axis=1)[:, ::-1]
        for k in range(top_k):
            prediction_dict["Top-{}".format(k+1)] = predictions[:, k]
    else:
        raise ValueError("Incorrect eval parameters - Dataset: {} - label_names: {} - top_k: {}".format(dataset, label_names, top_k))

    return prediction_dict



def inference(args):

    ann_root = args["ann_root"]
    data_root = args["data_root"]
    model_path = args["model_path"]
    outputPath = args["results_output"]
    eval_val = args["eval_val"]
    eval_test = args["eval_test"]
    use_ema = args["use_ema"]
    filter_novit = args["filter_novit"]

    if not os.path.isfile(model_path):
        raise ValueError("The provided path is not a file: {}".format(model_path))
    
    os.makedirs(outputPath, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    try:
        best_model = torch.load(model_path, map_location=device)
    except Exception as e:
        print(str(e))
        return None

    print(model_path)

    bargs = Namespace(**best_model["hyper_parameters_backbone"])
    hargs = Namespace(**best_model["hyper_parameters_head"])

    if bargs.backbone_model in backbones.VALID_BACKBONES:
        backbone = backbones.get_backbone(backbone_model = bargs.backbone_model,
                                            img_size = bargs.img_size,
                                            return_stages = bargs.backbone_feature_maps,
                                            pretrained_backbone = bargs.pretrained_backbone,
                                            custom_pretrained_path="")

    if hargs.head_model in heads.HEADS:
        if hargs.head_model == "BaseHead":
            head = heads.BaseHead(num_classes = hargs.num_classes,
                                        backbone_feature_map_sizes = backbone.feature_map_sizes,
                                        last_stage = backbone.last_stage,
                                        global_pool = hargs.global_pool,
                                        sigmoid_loss = hargs.sigmoid_loss,
                                        tresnet_init = hargs.tresnet_init)

        elif hargs.head_model == "MultiScaleViTHead":
            tokenizer_kwargs = {"patch_size":hargs.patch_size,
                                "num_clusters":hargs.num_clusters,
                                "l2_normalize": hargs.l2_normalize,
                                "eps": hargs.sinkhorn_eps,
                                "iters": hargs.sinkhorn_iters}

            transformer_block_kwargs = {"num_heads":hargs.num_heads,
                                        "qkv_bias": hargs.qkv_bias,
                                        "mlp_ratio":hargs.mlp_ratio,
                                        "proj_drop":hargs.proj_drop,
                                        "attn_drop": hargs.attn_drop}

            cross_attention_kwargs = {"num_heads":hargs.cross_num_heads,
                                        "qkv_bias":hargs.cross_qkv_bias,
                                        "mlp_ratio":hargs.cross_mlp_ratio,
                                        "proj_drop": hargs.cross_proj_drop,
                                        "attn_drop": hargs.cross_attn_drop,
                                        "block_drop": hargs.cross_block_drop}

            if filter_novit and hargs.multiscale_method in ["CrossScale-Token", "CrossScale-CNN"]:
                no_vit_layers = [m for m in hargs.backbone_feature_maps if m != "layer4"]
            else:
                no_vit_layers = hargs.no_vit_layers
            
            head = heads.MultiScaleViTHead(num_classes = hargs.num_classes,
                                                token_dim = hargs.token_dim,
                                                representation_size = hargs.representation_size,
                                                tokenizer_layer_name = hargs.tokenizer_layer,
                                                block_type = blocks.__dict__[hargs.block_type],
                                                block_drop = hargs.block_drop,
                                                use_pos_embed = hargs.use_pos_embed,
                                                pos_embed_drop = hargs.pos_embed_drop,
                                                backbone_feature_map_sizes = backbone.feature_map_sizes,
                                                backbone_feature_maps = hargs.backbone_feature_maps,
                                                transformer_depths = hargs.transformer_depth,
                                                sigmoid_loss = hargs.sigmoid_loss,
                                                norm_layer = hargs.norm_layer,
                                                act_layer = hargs.act_layer,
                                                tokenizer_kwargs_base = tokenizer_kwargs,
                                                transformer_block_kwargs_base = transformer_block_kwargs,
                                                cross_attention_kwargs = cross_attention_kwargs,
                                                cross_block_type = blocks.__dict__[hargs.cross_block_type],
                                                shared_tower=hargs.shared_tower,
                                                multiscale_method=hargs.multiscale_method,
                                                late_fusion=hargs.late_fusion,
                                                cross_scale_all=hargs.cross_scale_all,
                                                no_vit_layers=no_vit_layers,
                                                shared_tokenizer=hargs.shared_tokenizer,
                                                tresnet_init = hargs.tresnet_init,
                                                use_mean_token=hargs.use_mean_token)

        else:
            raise ValueError("Got head {}, but no such head is in this codebase".format(hargs.head_model))
    else:
        raise ValueError("Got head {}, but no such head is in this codebase".format(hargs.head_model))


    model_version = args["model_version"]
    if model_version == "":
        model_version = os.path.splitext(os.path.basename(model_path))[0]
    if use_ema:
        model_version += "_EMA"
    if filter_novit:
        model_version += "_NOViT"

    # Load best checkpoint
    if use_ema:
        print("EMA")
        updated_backbone_state_dict = best_model["state_dict_backbone_ema"]
        updated_head_state_dict = best_model["state_dict_head_ema"]
    else:
        print("NO EMA")
        updated_backbone_state_dict = best_model["state_dict_backbone"]
        updated_head_state_dict = best_model["state_dict_head"]
    
    backbone.load_state_dict(updated_backbone_state_dict)
    head.load_state_dict(updated_head_state_dict)

    print("STATE DICTS LOADED")
    print()
    
    # initialize dataloaders
    eval_transform = transforms.create_sewerml_eval_transformations({"img_size": bargs.img_size, "model_name": bargs.backbone_model})

    act_func = nn.Sigmoid()  
    
    backbone = backbone.to(device)
    head = head.to(device)

    if eval_val:
        val_dataloader, label_names = dataloader.get_dataloader(args["dataset"], args["batch_size"], args["workers"], ann_root, data_root, "Val", eval_transform)
        val_predictions, val_imgPaths = evaluate(val_dataloader, backbone, head, act_func, device)
        
        val_prediction_dict = process_predictions(args["dataset"], val_predictions, val_imgPaths, label_names, top_k=10)
       
        val_prediction_df = pd.DataFrame(val_prediction_dict)
        val_prediction_df.to_csv(os.path.join(outputPath, "{}_prediction_val.csv".format(model_version)), sep=",", index=False)
                        
    
    if eval_test:
        test_dataloader, label_names = dataloader.get_dataloader(args["dataset"], args["batch_size"], args["workers"], ann_root, data_root, "Test", eval_transform)
        test_predictions, test_imgPaths = evaluate(test_dataloader, backbone, head, act_func, device)

        test_prediction_dict = process_predictions(args["dataset"], test_predictions, test_imgPaths, label_names, top_k=10)

        test_prediction_df = pd.DataFrame(test_prediction_dict)
        test_prediction_df.to_csv(os.path.join(outputPath, "{}_prediction_test.csv".format(model_version)), sep=",", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--conda_env', type=str, default='Pytorch-Lightning')
    parser.add_argument('--notification_email', type=str, default='')
    parser.add_argument('--ann_root', type=str, default='./annotations_sewerml')
    parser.add_argument('--dataset', type=str,  choices=["SewerML"], default="SewerML")
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=512, help="Size of the batch per GPU")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_version", type=str, default="")
    parser.add_argument("--results_output", type=str, default = "")
    parser.add_argument("--eval_val", action='store_true')
    parser.add_argument("--eval_test", action='store_true')
    parser.add_argument("--use_ema", action='store_true')
    parser.add_argument("--filter_novit", action='store_true')


    args = vars(parser.parse_args())

    inference(args)