import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
import torchvision

from model import UNet
# from unet.model import UNet

from matplotlib import pyplot as plt
from dataset import getColors

import lightning as L
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy


def print_dataset_masks(data: tuple[Tensor, Tensor, Tensor], title: str):
    plt.close()

    fig = plt.figure(figsize=(10, 12))
    fig.suptitle(title)

    colors = getColors()
    input, predictions, truth = data

    truth = torch.stack(
        [
            torchvision.utils.draw_segmentation_masks(
                img, mask.bool(), alpha=1, colors=colors
            )
            for img, mask in zip(input, truth)
        ]
    )
    predictions = torch.stack(
        [
            torchvision.utils.draw_segmentation_masks(
                img, mask.bool(), alpha=1, colors=colors
            )
            for img, mask in zip(input, predictions)
        ]
    )

    toimg = ToPILImage()

    fig.add_subplot(3, 1, 1)
    plt.axis("off")
    plt.imshow(toimg(make_grid(input.float(), nrow=5)))
    plt.title("Input")

    fig.add_subplot(3, 1, 2)
    plt.axis("off")
    plt.imshow(toimg(make_grid(predictions.float(), nrow=5)))
    plt.title("Predictions")

    fig.add_subplot(3, 1, 3)
    plt.axis("off")
    plt.imshow(toimg(make_grid(truth.float(), nrow=5)))
    plt.title("Truth")

    plt.show()


class CamouflagedAnimalsModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet(
            kernel_size=5,
            # num_classes=4,
        )
        self.criterion = nn.CrossEntropyLoss(Tensor([0.0, 0.1, 0.8, 0.1]))

        self.example_input_array = Tensor(20, 3, 512, 512)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        prediction = self.model(x)
        loss = self.criterion(prediction, y.float())
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        prediction = self.model(x)
        loss = self.criterion(prediction, y.float())
        self.log("validation_loss", loss)

        # calculate IOU
        jaccard = MulticlassJaccardIndex(
            num_classes=4, ignore_index=0, validate_args=False
        ).to(self.device)
        jaccard_idx = jaccard(prediction, y.float())
        self.log("validation_iou", jaccard_idx)

        prediction_classes = torch.argmax(prediction, dim=1)
        prediction_masks = torch.nn.functional.one_hot(prediction_classes, 4).permute(
            0, 3, 1, 2
        )

        maccuracy = MulticlassAccuracy(
            num_classes=4, ignore_index=0, average="micro"
        ).to(self.device)
        accuracy = maccuracy(prediction_masks, y)
        self.log("validation_accuracy", accuracy)

        print_dataset_masks(
            (x, prediction_masks, y),
            f"[{batch_idx:03d}] Validation accuracy: {accuracy*100:.4f}%",
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
