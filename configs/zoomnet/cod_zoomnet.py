_base_ = [
    "../_base_/common.py",
    "../_base_/train.py",
    "../_base_/test.py",
]

has_test = True
deterministic = True
use_custom_worker_init = False
# model_name = "ZoomNet"
model_name = "CAMO"
# base_seed=0
# base_seed=42
base_seed=5

train = dict(
    batch_size=6,
    # batch_size=4,
    # batch_size=8,
    num_workers=8,
    use_amp=True,
    # num_epochs=50,
    num_epochs=40,
    # num_epochs=200,
    epoch_based=True,
    lr=0.000035,
    # lr=0.00001,
    # lr=0.00005,
    # lr=0.00008,
    # lr=0.0001,
    # lr=0.00001,
    # lr=0.00003,
    test_from_epoch=35, # 自定义的，从第几个epoch开始，每训练一个epoch就计算一次指标
    # test_from_epoch=40, # 自定义的，从第几个epoch开始，每训练一个epoch就计算一次指标

    optimizer=dict(
        mode="adamw",
        set_to_none=True,
        group_mode="all",
        cfg=dict(
            # momentum=0.9,
            # weight_decay=5e-4,
            # nesterov=False,
        ),
    ),
    sche_usebatch=True,
    scheduler=dict(
        warmup=dict(
            num_iters=336 * 0,
            initial_coef=0.01,
            mode="linear",
        ),
        mode="cos",
        cfg=dict(
            lr_decay=0.9,
            min_coef=0.001,
        ),
    ),
)

test = dict(
    batch_size=8,
    num_workers=0,
    # show_bar=False,
    show_bar=True,
)

datasets = dict(
    train=dict(
        dataset_type="msi_cod_tr",
        # dataset_type="cross_val_msi_cod_tr",
        # shape=dict(h=384, w=384),
        shape=dict(h=512, w=512),
        path=["cod10k_camo_tr"],
        # path=["cod10k_tr"], # 只用 COD10K训练
        interp_cfg=dict(),
    ),
    test=dict(
        dataset_type="msi_cod_te",
        # shape=dict(h=384, w=384),
        shape=dict(h=512, w=512),
        # shape=dict(h=480, w=480),
        # path=["camo_te", "cod10k_te", "nc4k"],
        path=["chameleon", "camo_te", "cod10k_te", "nc4k"],
        interp_cfg=dict(),
    ),

    # train=dict(
    #     dataset_type="msi_cod_tr",
    #     shape=dict(h=256, w=256),
    #     path=["USOD10K_Train"],
    #     interp_cfg=dict(),
    #     ),
    # test=dict(
    #     dataset_type="msi_cod_te",
    #     shape=dict(h=256, w=256),
    #     path=["USOD10K_Test"],
    #     interp_cfg=dict(),
    # ),

    # train=dict(
    #     dataset_type="msi_cod_tr",
    #     shape=dict(h=512, w=512),
    #     path=["USOD10K_Train"],
    #     interp_cfg=dict(),
    # ),
    # test=dict(
    #     dataset_type="msi_cod_te",
    #     shape=dict(h=512, w=512),
    #     path=["USOD10K_Test"],
    #     interp_cfg=dict(),
    # ),
)
