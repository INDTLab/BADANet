# -*- coding: utf-8 -*-

import argparse
import json
import os
import os.path

import numpy as np
import torch
from tqdm import tqdm

from utils import builder, configurator, io, misc, ops, pipeline, recorder
# from torchstat import stat
# from torchsummary import summary
# from thop import profile
# from thop import clever_format

from thop import profile
from thop import clever_format
import time

def get_model_complexity(model, input_size=(1, 3, 512, 512)):
    B, C, H, W = input_size
    dummy_batch = {
        "image1.0": torch.randn(B, 3, H, W).to(model.device),
        "mask": torch.randn(B, 1, H, W).to(model.device),
        "edge": torch.randn(B, 1, H, W).to(model.device),
        "depth": torch.randn(B, 1, H, W).to(model.device),
    }

    # 注意 profile 需要 inputs 是 tuple，所以包一层
    flops, params = profile(model, inputs=(dummy_batch,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    return flops, params

def benchmark_fps(model, input_size=(1, 3, 512, 512), num_warmup=10, num_iters=50):
    B, C, H, W = input_size
    dummy_batch = {
        "image1.0": torch.randn(B, 3, H, W).to(model.device),
        "mask": torch.randn(B, 1, H, W).to(model.device),
        "edge": torch.randn(B, 1, H, W).to(model.device),
        "depth": torch.randn(B, 1, H, W).to(model.device),
    }
    model.eval()

    # 预热
    for _ in range(num_warmup):
        _ = model(dummy_batch)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        _ = model(dummy_batch)
    torch.cuda.synchronize()
    end = time.time()

    fps = num_iters / (end - start)
    return fps




def parse_config():
    parser = argparse.ArgumentParser("Training and evaluation script")
    parser.add_argument("--config", default="./configs/zoomnet/cod_zoomnet.py", type=str)
    parser.add_argument("--datasets-info", default="./configs/_base_/dataset/dataset_configs.json", type=str)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--load-from", default="/data1/wanglin/ZoomNet-main/ZoomNet-main/output/ZoomNet_BS12_LR5e-05_E50_H384_W384_OPMadamw_OPGMall_SCf3_AMP_INFObad_back_baseline_depth_edge_3/pth/state_final.pth", type=str)
    parser.add_argument("--save-path", default="/data1/wanglin/ZoomNet-main/ZoomNet-main/output/ZoomNet_BS12_LR5e-05_E50_H384_W384_OPMadamw_OPGMall_SCf3_AMP_INFObad_back_baseline_depth_edge_3/pre2", type=str)
    parser.add_argument("--minmax-results", action="store_true")
    parser.add_argument("--info", type=str)
    args = parser.parse_args()

    config = configurator.Configurator.fromfile(args.config)
    config.use_ddp = False
    if args.model_name is not None:
        config.model_name = args.model_name
    if args.batch_size is not None:
        config.test.batch_size = args.batch_size
    if args.load_from is not None:
        config.load_from = args.load_from
    if args.info is not None:
        config.experiment_tag = args.info
    # if args.save_path is not None:
    #     if os.path.exists(args.save_path):
    #         if len(os.listdir(args.save_path)) != 0:
    #             raise ValueError(f"--save-path is not an empty folder.")
    #     else:
    #         print(f"{args.save_path} does not exist, create it.")
    #         os.makedirs(args.save_path)
    config.save_path = args.save_path
    config.test.to_minmax = args.minmax_results

    with open(args.datasets_info, encoding="utf-8", mode="r") as f:
        datasets_info = json.load(f)

    te_paths = {}
    for te_dataset in config.datasets.test.path:
        if te_dataset not in datasets_info:
            raise KeyError(f"{te_dataset} not in {args.datasets_info}!!!")
        te_paths[te_dataset] = datasets_info[te_dataset]
    config.datasets.test.path = te_paths

    config.proj_root = os.path.dirname(os.path.abspath(__file__))
    config.exp_name = misc.construct_exp_name(model_name=config.model_name, cfg=config)
    return config


def test_once(
    model,
    data_loader,
    save_path,
    tta_setting,
    clip_range=None,
    show_bar=False,
    desc="[TE]",
    to_minmax=False,
):
    model.is_training = False
    cal_total_seg_metrics = recorder.CalTotalMetric()

    pgr_bar = enumerate(data_loader)
    if show_bar:
        pgr_bar = tqdm(pgr_bar, total=len(data_loader), ncols=79, desc=desc)
    for batch_id, batch in pgr_bar:
        batch_images = misc.to_device(batch["data"], device=model.device)
        # if tta_setting.enable:
        #     logits, edge = pipeline.test_aug(
        #         model=model, data=batch_images, strategy=tta_setting.strategy, reducation=tta_setting.reduction
        #     )
        # else:
        #     logits, edge = model(data=batch_images)

        if tta_setting.enable:
            logits = pipeline.test_aug(
                model=model, data=batch_images, strategy=tta_setting.strategy, reducation=tta_setting.reduction
            )
        else:
            logits = model(data=batch_images, batch=batch)
        probs = logits.squeeze(1).cpu().detach().numpy()
        # print(probs.shape)


        # probs_list = logits
        # for j in range(len(probs_list)):
        #     probs = probs_list[j].sigmoid().squeeze(1).cpu().detach().numpy()
        #     save_path2 = save_path + "_" + str(j)
        #     # print(f"Note, Now, Results will be saved into {save_path2}")
        #
        #     for i, pred in enumerate(probs):
        #         mask_path = batch["info"]["mask_path"][i]
        #         mask_array = io.read_gray_array(mask_path, dtype=np.uint8)
        #         mask_h, mask_w = mask_array.shape
        #
        #         # here, sometimes, we can resize the prediciton to the shape of the mask's shape
        #         if j == 0:
        #             pred = ops.imresize(pred, target_h=mask_h, target_w=mask_w, interp="linear")
        #
        #         if clip_range is not None:
        #             pred = ops.clip_to_normalize(pred, clip_range=clip_range)
        #
        #         if save_path2:  # 这里的save_path包含了数据集名字
        #             ops.save_array_as_image(
        #                 data_array=pred, save_name=os.path.basename(mask_path), save_dir=save_path2, to_minmax=to_minmax
        #             )
        #         if j == 0:
        #             pred = (pred * 255).astype(np.uint8)
        #             cal_total_seg_metrics.step(pred, mask_array, mask_path)
        # save_path3 = save_path + "_edge"
        # edge = edge.sigmoid().squeeze(1).cpu().detach().numpy()
        # for i, pred in enumerate(edge):
        #     mask_path = batch["info"]["mask_path"][i]
        #     mask_array = io.read_gray_array(mask_path, dtype=np.uint8)
        #     mask_h, mask_w = mask_array.shape
        #     # here, sometimes, we can resize the prediciton to the shape of the mask's shape
        #     pred = ops.imresize(pred, target_h=mask_h, target_w=mask_w, interp="linear")
        #     if save_path3:  # 这里的save_path包含了数据集名字
        #         ops.save_array_as_image(
        #             data_array=pred, save_name=os.path.basename(mask_path), save_dir=save_path3, to_minmax=to_minmax
        #         )


        for i, pred in enumerate(probs):
            mask_path = batch["info"]["mask_path"][i]
            mask_array = io.read_gray_array(mask_path, dtype=np.uint8)
            mask_h, mask_w = mask_array.shape

            # here, sometimes, we can resize the prediciton to the shape of the mask's shape
            pred = ops.imresize(pred, target_h=mask_h, target_w=mask_w, interp="linear")

            if clip_range is not None:
                pred = ops.clip_to_normalize(pred, clip_range=clip_range)

            if to_minmax:
                pred = ops.minmax(pred)

            if save_path:  # 这里的save_path包含了数据集名字
                ops.save_array_as_image(data_array=pred, save_name=os.path.basename(mask_path), save_dir=save_path)

            pred = (pred * 255).astype(np.uint8)
            # print(pred.shape)
            cal_total_seg_metrics.step(pred, mask_array, mask_path)

        # name_ls = ['_x', '_e4', '_e3', '_r1', '_b4', '_b3', '_r2', '_f4', '_f3', '_r3']
        # name_ls = ['_x', '_e', '_b0', '_b1', '_b2', '_b3', '_b4', '_f0', '_f1', '_f2', '_f3', '_f4', '_r3']
        # for j in range(13):
        #     probs = out3[j].sigmoid().squeeze(1).cpu().detach().numpy()
        #     for i, pred in enumerate(probs):
        #         mask_path = batch["info"]["mask_path"][i]
        #         mask_array = io.read_gray_array(mask_path, dtype=np.uint8)
        #         mask_h, mask_w = mask_array.shape
        #
        #         # here, sometimes, we can resize the prediciton to the shape of the mask's shape
        #         pred = ops.imresize(pred, target_h=mask_h, target_w=mask_w, interp="linear")
        #
        #         if clip_range is not None:
        #             pred = ops.clip_to_normalize(pred, clip_range=clip_range)
        #
        #         if to_minmax:
        #             pred = ops.minmax(pred)
        #
        #         if save_path:  # 这里的save_path包含了数据集名字
        #             ops.save_array_as_image(data_array=pred, save_name=os.path.basename(mask_path.split('.')[0] + name_ls[j] + '.png'),
        #                                     save_dir=save_path)

    fixed_seg_results = cal_total_seg_metrics.get_results()
    return fixed_seg_results


@torch.no_grad()
def testing(model, cfg):
    # ---- 模型复杂度统计 ----
    # flops, params = get_model_complexity(model, input_size=(1, 3, 512, 512))
    # fps = benchmark_fps(model, input_size=(1, 3, 512, 512))
    # print(f"Model Params: {params}, FLOPs: {flops}, FPS: {fps:.2f}")
    pred_save_path = None
    for data_name, data_path, loader in pipeline.get_te_loader(cfg):

        if cfg.save_path:
            pred_save_path = os.path.join(cfg.save_path, data_name)
            print(f"Results will be saved into {pred_save_path}")
        seg_results = test_once(
            model=model,
            save_path=pred_save_path,
            data_loader=loader,
            tta_setting=cfg.test.tta,
            clip_range=cfg.test.clip_range,
            show_bar=cfg.test.get("show_bar", False),
            to_minmax=cfg.test.get("to_minmax", False),
        )
        print(f"Results on the testset({data_name}): {misc.mapping_to_str(data_path)}\n{seg_results}")


def main():
    cfg = parse_config()

    model, model_code = builder.build_obj_from_registry(
        registry_name="MODELS", obj_name=cfg.model_name, return_code=True
    )
    io.load_weight(model=model, load_path=cfg.load_from)

    model.device = "cuda:4"
    # model.device = "cuda:7"
    model.to(model.device)
    model.eval()

    # input1 = torch.randn(1, 3, 384, 384)
    # # model = MODEL()  # your model here
    # macs, params = profile(model, inputs=(input1,))  # “问号”内容使我们之后要详细讨论的内容，即是否为FLOPs
    # macs, params = clever_format([flops, params], "%.3f")

    testing(model=model, cfg=cfg)


if __name__ == "__main__":
    main()
