from .composition import Compose
from .augmentor import DataAugment
from .test_augmentor import TestAugmentor

# augmentation methods
from .warp import Elastic
from .grayscale import Grayscale
from .flip import Flip
from .rotation import Rotate
from .rescale import Rescale
from .misalign import MisAlignment
from .missing_section import MissingSection
from .missing_parts import MissingParts
from .motion_blur import MotionBlur
from .cutblur import CutBlur
from .cutnoise import CutNoise
from .mixup import MixupAugmentor

__all__ = ['Compose',
           'DataAugment', 
           'Elastic',
           'Grayscale',
           'Rotate',
           'Rescale',
           'MisAlignment',
           'MissingSection',
           'MissingParts',
           'Flip',
           'MotionBlur',
           'CutBlur',
           'CutNoise',
           'MixupAugmentor',
           'TestAugmentor']


def build_train_augmentor(opt, keep_uncropped=False, keep_non_smoothed=False):
    # The two arguments, keep_uncropped and keep_non_smoothed, are used only
    # for debugging, which are False by defaults and can not be adjusted
    # in the config files.
    aug_list = []
    # #1. rotate
    # if opt.ROTATE:
    #     aug_list.append(Rotate())

    # #2. rescale
    # if opt.RESCALE:
    #     aug_list.append(Rescale())

    #3. flip
    if opt.FLIP:
        aug_list.append(Flip())

    # #4. grayscale
    # if opt.GRAYSCALE:
    #     aug_list.append(Grayscale())


    augmentor = Compose(aug_list, input_size=(32,256,256), smooth=False,
                        keep_uncropped=keep_uncropped, keep_non_smoothed=keep_non_smoothed)

    return augmentor
