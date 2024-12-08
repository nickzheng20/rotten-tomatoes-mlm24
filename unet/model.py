import torch
import segmentation_models_pytorch as smp

class TomatoLeafModel(torch.nn.Module):
    def __init__(
            self,
            encoder_name: str="resnet18",
            encoder_weights: str="imagenet",
            in_channels: int=3,
            out_classes: int=1,
            **kwargs
    ):
        super().__init__()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

    def forward(self, image):
        mask = self.model(image)
        return mask


class TomatoLeafMaskModel(torch.nn.Module):
    def __init__(
            self,
            encoder_name: str="resnet18",
            encoder_weights: str="imagenet",
            in_channels: int=1,
            out_classes: int=1,
            **kwargs
    ):
        super().__init__()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

    def forward(self, image):
        mask = self.model(image)
        return mask


class DoubleTomatoLeafModel(torch.nn.Module):
    def __init__(
            self,
            encoder_name: str="resnet18",
            encoder_weights: str="imagenet",
            in_channels: int=3,
            inner_channels: int=1,
            out_classes: int=1,
            **kwargs
    ):
        super().__init__()

        self.image_model = TomatoLeafModel(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            out_classes=out_classes,
            **kwargs
        )

        self.mask_model = TomatoLeafMaskModel(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=inner_channels,
            out_classes=out_classes,
            **kwargs
        )

    def forward(self, image):
        mask = self.image_model(image)
        enhanced_mask = self.mask_model(mask)
        return enhanced_mask
