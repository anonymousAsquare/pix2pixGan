import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.input_image_dir = self.root_dir + "/" + os.listdir(self.root_dir)[0]
        self.output_image_dir = self.root_dir + "/" + os.listdir(self.root_dir)[1]
        self.list_input_images = os.listdir(self.input_image_dir)
        self.list_output_images = os.listdir(self.output_image_dir)

    def __len__(self):
        return len(self.list_input_images)

    def __getitem__(self, index):
        in_img_file = self.list_input_images[index]
        out_img_file = self.list_output_images[index]
        in_img_path = os.path.join(self.input_image_dir, in_img_file)
        out_img_path = os.path.join(self.output_image_dir, out_img_file)
        in_image = np.array(Image.open(in_img_path))
        out_image = np.array(Image.open(out_img_path))
        input_image = in_image[:, :, :]
        target_image = out_image[:, :, :]
        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]
        return input_image, target_image
