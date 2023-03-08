# Multi-Scale Hybrid Vision Transformer and Sinkhorn Tokenizer for Sewer Defect Classification

This repository is the official implementation of [Multi-Scale Hybrid Vision Transformer and Sinkhorn Tokenizer for Sewer Defect Classification](https://doi.org/10.1016/j.autcon.2022.104614). 


The MSHViT project page can be found [here](http://vap.aau.dk/mshvit).

## Requirements

The main packages needed are listed below:

- Pytorch 1.8.1
- Torchvision 0.9.1
- Pytorch-Lightning 1.3.7
- torchmetrics 0.3.2
- Pandas 1.3.5
- Numpy 1.20.2
- inplace-abn 1.1.0
- wandb 0.12.9
- einops 0.4.1


The Sewer-ML dataset can be accessed after filling out this [Google Form](https://forms.gle/hBaPtoweZumZAi4u9).
The Sewer-ML dataset is licensed under the Creative Commons BY-NC-SA 4.0 license.


## Training

When training the images are normalized with the following mean and standard deviation:
- mean = [0.523, 0.453, 0.345]
- std = [0.210, 0.199, 0.154]

The models can be trained in two variations: with and without the MSHViT Head. The variation share trainer script, and is selected through the head_model argument

```
trainer.py --precision 16 --batch_size 256 --max_epochs 40 --gpus 1 --img_size 224 --ann_root <path_to_annotations> --data_root <path_to_data> --deterministic --backbone_model resnet50 --head_model BaseHead --sigmoid_loss --model_ema --monitor_metric --wandb_project <wandb_project> --wandb_group <wandb_group> --progress_bar_refresh_rate 1000 --flush_logs_every_n_steps 1000 --log_every_n_steps 1 --log_save_dir <path_to_model_logs>
```

```
trainer.py --precision 16 --batch_size 256 --max_epochs 40 --gpus 1 --img_size 224 --ann_root <path_to_annotations> --data_root <path_to_data>  --deterministic --backbone_model resnet50 --head_model MultiScaleViTHead --token_dim 512 --transformer_depth 2 --block_drop 0. --tokenizer_layer Sinkhorn --block_type FNetBlock --use_pos_embed --num_heads 8 --qkv_bias --mlp_ratio 4. --proj_drop 0. --attn_drop 0. --pos_embed_drop 0. --cross_num_heads 8 --cross_qkv_bias --cross_proj_drop 0. --cross_attn_drop 0. --cross_block_drop 0. --num_cluster 64 --sinkhorn_eps 0.25 --sinkhorn_iters 5 --l2_normalize --backbone_feature_maps layer3 layer4 --shared_tower --multiscale_method CrossScale-Token --sigmoid_loss --model_ema --monitor_metric --wandb_project <wandb_project> --wandb_group <wandb_group> --progress_bar_refresh_rate 1000 --flush_logs_every_n_steps 1000 --log_every_n_steps 1 --log_save_dir <path_to_model_logs>
```

## Evaluation

To evaluate a set of models on the validation set of the Sewer-ML dataset, first the raw predictions for each image should be generated, which is subsequently compared to the ground truth. The raw predictions should be probabilities.

When the predictions have been obtained the performance of the model can be determined using the calculate_results.py script.

```
python calculate_results.py --output_path <path_to_metric_results> --split <dataset_split_to_use> --score_path <path_to_predictions> --gt_path <path_to_annotations>
```

Raw predictions are obtained with the inference.py script, assuming the weights have been transformed from the pytorch lightning format to a simplified format using the extract_weights.py script. EMA weights can be used by passing the use_ema flag.

```
python inference.py --ann_root <path_to_annotations> --data_root <path_to_data> --results_output <path_to_results> --model_path <path_to_models> --split <dataset_split_to_use>
```


## Pre-trained Models

You can download pretrained models here:

- [Model Repository](https://sciencedata.dk/shared/mshvit_aic2022_models) trained on Sewer-ML using the parameters described in the paper.

Each model weight file consists of a dict with the model state_dicts and the necessary model hyper_parameters.


## Code Credits

Parts of the code builds upon prior work:

- The Sinkhorn Tokenizer builds upon the SuperGlue Sinkhorn-Knopp implementation. Found at: [https://github.com/magicleap/SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork)
- The BotNet backbone model implementation is from the bottleneck-transformer-pytorch repo by Phil Wang. Found at:[https://github.com/lucidrains/bottleneck-transformer-pytorch](https://github.com/lucidrains/bottleneck-transformer-pytorch)
- The CoAtNet backbone model implementation is form the 
coatnet-pytorch repo by Justin Wu: [https://github.com/chinhsuanwu/coatnet-pytorch](https://github.com/chinhsuanwu/coatnet-pytorch)
- The TResNet backbone model implementation is form the 
TResNet repo by Alibaba-MIIL: [https://github.com/Alibaba-MIIL/TResNet](https://github.com/Alibaba-MIIL/TResNet)
- Parts of the training code and large part of the ViT implementation is based and inspired by the timm framework by Ross Wrightman: [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)



## Contributing

The Code is licensed under an MIT License, with exceptions of the BoTNet, TResNet, CoAtNet, Sinkhorn, and timm code which follows the license of the original authors.

The Sewer-ML Dataset follows the Creative Commons Attribute-NonCommerical-ShareAlike 4.0 (CC BY-NC-SA 4.0) International license.



## Bibtex
```bibtex
@article{Haurum_2022_AiC,
author = {Joakim Bruslund Haurum and Meysam Madadi and Sergio Escalera and Thomas B. Moeslund},
title = {Multi-scale hybrid vision transformer and Sinkhorn tokenizer for sewer defect classification},
journal = {Automation in Construction},
volume = {144},
pages = {104614},
year = {2022},
issn = {0926-5805},
doi = {https://doi.org/10.1016/j.autcon.2022.104614},
}
```