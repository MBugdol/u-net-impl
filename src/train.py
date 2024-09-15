from aiutils import CamouflagedAnimalsModel
from dataset import CamouflagedAnimalsDataset, colorMaskToOneHot

import torch as T
from torchvision.transforms import v2 as TV
import lightning as L


NoopTransform = TV.RandomHorizontalFlip(p=0.0)

# common transformations - applied to both the input & mask images
common_data_augment = TV.RandomChoice(
    [
        TV.RandomHorizontalFlip(p=1.0),
        TV.RandomRotation((-15, +15)),
        TV.RandomCrop(size=(256, 256), padding_mode="constant", pad_if_needed=True),
        NoopTransform
    ]
)
common_normalize = [
    TV.Resize((512, 512), interpolation=TV.InterpolationMode.NEAREST),
    TV.ToDtype(T.float, scale=True),
]
common_transform = TV.Compose([common_data_augment] + common_normalize)

# image transformations - applied only to the input images
# use IMAGENET transformations for input
image_transform = TV.RandomChoice(
    [
        TV.Grayscale(num_output_channels=3),
        TV.ColorJitter(brightness=0.5, hue=0.3),
        TV.GaussianBlur(kernel_size=(5, 9)),
        TV.RandomPosterize(p=1, bits=3),
        TV.RandomEqualize(p=1),
        TV.RandomSolarize(threshold=0.8, p=1),
        NoopTransform
    ]
)

# mask transformations - applied only to the mask images
# convert mask from RGB to one-hot mask with 4 channels
mask_transform = TV.Compose([colorMaskToOneHot])

dataset = CamouflagedAnimalsDataset(
    images_path="images",
    masks_path="masks",
    common_transform=common_transform,
    image_transform=image_transform,
    mask_transform=mask_transform,
)

# enable TPU
T.set_float32_matmul_precision("medium")

# configure checkpoint
checkpoint_path = "src/lightning_logs/version_1/checkpoints/epoch=1-step=98.ckpt"
# checkpoint_path: str | None = None

seed = T.Generator().manual_seed(42)
train_set, valid_set = T.utils.data.random_split(dataset, [0.9, 0.1], generator=seed)

train_loader = T.utils.data.DataLoader(
    train_set,
    batch_size=5,
    shuffle=True,
)
valid_loader = T.utils.data.DataLoader(
    valid_set,
    batch_size=5,
    shuffle=False,
)

if checkpoint_path is None:
    model_lightning = CamouflagedAnimalsModel()
else:
    model_lightning = CamouflagedAnimalsModel.load_from_checkpoint(checkpoint_path)

trainer = L.Trainer(
    limit_train_batches=250,
    accelerator="auto",
    devices="auto",
    strategy="auto",
    max_epochs=1,
    log_every_n_steps=5,
    # fast_dev_run=True
)

trainer.fit(
    model=model_lightning,
    train_dataloaders=train_loader,
    val_dataloaders=valid_loader,
)