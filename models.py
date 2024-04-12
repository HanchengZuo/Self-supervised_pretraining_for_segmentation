import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vit_b_16
from torchvision.transforms import functional as F


class BaseModel(nn.Module):
    """
    Base model using pre-trained visual transformer (ViT) as backbone network

    Args:
        pretrained (bool): true indicates loading pre-trained weights

    Attributes:
        vit: vistion transformer without classification head
    """

    def __init__(self, pretrained=True):
        super().__init__()
        # Initialize vision transformer model, optionally load pre-trained weights
        self.vit = vit_b_16(pretrained=pretrained)
        # Remove categorization header as standard categorization tasks may not be required
        self.vit.heads = nn.Identity()

    def forward(self, x):
        outputs = self.vit(x)

        return outputs


class SelfSuperviseHead(nn.Module):
    """
    Head for self-supervised learning. Convert input to the original image shape

    Args:
        dim (int): imput dimension
        image_size (tuple[int]): (width,height) of input image
    """

    def __init__(self, dim=768, image_size=(224, 224)):
        super().__init__()
        self.image_size = image_size
        self.projection = nn.Linear(dim, 3 * image_size[0] * image_size[1])

    def forward(self, x):
        batch_size = x.size(0)
        outputs = self.projection(x)
        outputs = outputs.reshape(
            batch_size, 3, self.image_size[0], self.image_size[1])

        return outputs


class SelfSuperviseHeadNonelinear(nn.Module):
    """
    Head for self-supervised learning. Convert input to the original image shape

    Args:
        dim (int): imput dimension
        image_size (tuple[int]): (width,height) of input image
    """

    def __init__(self, dim=768, image_size=(224, 224)):
        super().__init__()
        self.image_size = image_size
        self.projection = nn.Linear(dim, 3 * image_size[0] * image_size[1])

    def forward(self, x):
        batch_size = x.size(0)
        outputs = self.projection(x)
        outputs = outputs.reshape(
            batch_size, 3, self.image_size[0], self.image_size[1])

        return outputs


class SegmentationHead(nn.Module):
    """
    Segmentation head

    Args:
        num_class (int): number of classes
        dim (int): imput dimension
        image_size (tuple[int]): (width,height) of input image
        c1, c2, c3 (int): channels in upsampling, c1(input) -> c2 -> c3 -> num_classes(output)
    """

    def __init__(self, num_classes=3, dim=768, image_size=(224, 224), c1=64, c2=32, c3=16):
        super().__init__()
        self.image_size = image_size
        self.c1 = c1
        self.projection = nn.Linear(dim, c1 * image_size[0] * image_size[1])

        # Decoder to upsample the patch representations to the original image resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                c3, num_classes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        batch_size = x.size(0)
        outputs = self.projection(x)
        outputs = outputs.reshape(
            batch_size, self.c1, self.image_size[0], self.image_size[1])
        outputs = self.decoder(outputs)

        return outputs


class BaseSelfSupervise(nn.Module):
    """
    Base model with selfsupervied head, i.e. projection for selfsupervie learning

    Args:
        word_dim (int): word vector dimension in ViT
        image_size (tuple[int]): (width, height) of imput images
        base_model_weights (None|'pretrained'|str|dict), can be:
            'pretrained': load the pretrained vit weights automatically
            base model path (str): load the weights from model path
            base model class (BaseModel): integrate the base model
            base model weights (dict): load the weight from this dict
            None: do not load pre-trained weights
    """

    def __init__(self, word_dim=768, image_size=(224, 224), base_model_weights=None):
        super().__init__()

        # save attribute
        if type(base_model_weights) is BaseModel:
            self.base_model = base_model_weights
        elif base_model_weights == 'pretrain':
            self.base_model = BaseModel(True)
        else:
            self.base_model = BaseModel(False)
            if type(base_model_weights) == dict:
                self.base_model.load_state_dict(base_model_weights)
            elif type(base_model_weights) == str:
                self.base_model.load_state_dict(torch.load(base_model_weights))
            elif base_model_weights is not None:
                raise ValueError('unknown base_model_weights')

        self.heads = SelfSuperviseHead(dim=word_dim, image_size=image_size)

    def forward(self, x):

        outputs = self.base_model(x)
        outputs = self.heads(outputs)

        return outputs


class BaseSegmentation(nn.Module):
    """
    Base model with segmentation head for segmentation task.

    Args:
        num_classes (int): Number of classes.
        word_dim (int): Word vector dimension in ViT.
        image_size (tuple): (width, height) of input images.
        c1, c2, c3 (int): Channels in upsampling, c1(input) -> c2 -> c3 -> num_classes(output).
        base_model_weights (None|'pretrained'|str|dict): Can be:
            'pretrained': Load the pretrained ViT weights automatically.
            str: Load the weights from a model path.
            dict: Load the weight from this dict.
            None: Do not load pre-trained weights.
    """

    def __init__(self, num_classes=3, word_dim=768, image_size=(224, 224), 
                 c1=64, c2=32, c3=16, 
                 base_model_weights=None):
        super().__init__()

        # Initialize the base model
        self.base_model = BaseModel(pretrained=(
            base_model_weights == 'pretrained'))

        # Load the base model weights if provided
        if isinstance(base_model_weights, dict):
            self.base_model.load_state_dict(base_model_weights)
        elif isinstance(base_model_weights, str):
            # Load the model weights from a file and handle 'module.' prefix
            state_dict = torch.load(base_model_weights)
            new_state_dict = {
                k.replace('module.', ''): v for k, v in state_dict.items()}
            self.base_model.load_state_dict(new_state_dict, strict=False)
        elif base_model_weights is not None and base_model_weights != 'pretrained':
            raise ValueError('unknown base_model_weights')

        # Initialize the segmentation head
        self.heads = SegmentationHead(
            num_classes, word_dim, image_size, c1, c2, c3)

    def forward(self, x):
        # Forward pass through the base model and the segmentation head
        outputs = self.base_model(x)
        outputs = self.heads(outputs)
        return outputs


class ViTSegmentation(nn.Module):
    """
    ViT with decoder for segmentation task or self-supervised learning (with trimap as labels)

    Args:
        num_classes: number of classes
        word_dim: word vector dimension in ViT
        image_size: (width, height) of imput images
        c1, c2, c3: channels in upsampling, c1(input) -> c2 -> c3 -> num_classes(output)
        pretrained: bool, true indicates pre-trained weights for vit_b_16
    """

    def __init__(self, num_classes=3, word_dim=768, image_size=(224, 224), 
                 c1=64, c2=32, c3=16, 
                 pretrained=True):
        super().__init__()

        # save attribute
        self.image_size = image_size
        self.c1 = c1

        # Load a pretrained ViT model
        self.vit = vit_b_16(pretrained=pretrained)
        self.vit.heads = nn.Identity()  # Remove the classification head
        self.projection = nn.Linear(
            word_dim, c1 * image_size[0] * image_size[1])

        # Decoder to upsample the patch representations to the original image resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                c3, num_classes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        # ViT outputs one feature vector per patch, plus a class token
        # Here, we reshape the patch features to a feature map and pass it through the decoder
        batch_size = x.size(0)

        vit_features = self.vit(x)
        x = self.projection(vit_features)

        x = x.reshape(batch_size, self.c1,
                      self.image_size[0], self.image_size[1])
        out = self.decoder(x)

        return out


class Decoder(nn.Module):
    """
    MLP decoder for segmentation task

    Args:
        latent_dim (int): embedding vector dimension in ViT
        hidden_dim (int): hidden non-linear dimension
        output_dim (int): task output dimension
    """

    def __init__(self, latent_dim:int, hidden_dim:int, output_dim:int):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class MaskedAutoEncoder(nn.Module):
    """
    Masked autoencoder for segmentation task

    Args:
        encoder (nn.Moudule): embedding vector encoder
        decoder (nn.Moudule): task output decoder
    """

    def __init__(self, encoder:nn.Module, decoder:nn.Module):
        super(MaskedAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, mask):
        x_masked = mask * x
        encode_patch = self.encoder(x_masked)
        reconstruct = self.decoder(encode_patch)

        return reconstruct
