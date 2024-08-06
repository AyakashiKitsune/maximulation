import albumentations as A
import numpy as np
import os
from PIL import Image



def generate_motion_blur(img,dst):
    motion_blur = A.Compose(
        [
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=(15, 15), p=1, always_apply=True),
                ],
                p=1,
            )
        ]
    )
    image = resize_image(img)
    transformed = motion_blur(image=image)["image"]
    save_path = dst
    Image.fromarray(transformed).save(save_path)


def generate_iso_noise(img, dst):
    iso_noise = A.Compose(
        [
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(2000, 2000), p=1, always_apply=True),
                ],
                p=1,
            )
        ]
    )
    image = resize_image(img)
    transformed = iso_noise(image=image)["image"]
    save_path = dst
    Image.fromarray(transformed).save(save_path)


def generate_randombrightnesscontrast(img,dst):
    random_brightness_contrast = A.Compose(
        [
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.5, -0.5),
                        contrast_limit=(0.1, 0.1),
                        brightness_by_max=True,
                        always_apply=True,
                        p=1,
                    )
                ],
                p=1,
            )
        ]
    )
    image = resize_image(img)
    transformed = random_brightness_contrast(image=image)["image"]
    save_path = dst
    Image.fromarray(transformed).save(save_path)


def resize_image(img, height=720, save=False,dist =""):
    image = Image.open(img)
    resize = A.Compose(
        [A.Resize(height=height, width=image.width * height // image.height)]
    )
    image = np.array(image)
    transformed = resize(image=image)["image"]
    if save:
        save_path = dist
        Image.fromarray(transformed).save(save_path)

    return transformed
