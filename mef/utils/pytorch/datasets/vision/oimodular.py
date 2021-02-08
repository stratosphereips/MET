from collections import defaultdict as dd
from concurrent import futures
from pathlib import Path
from typing import Callable, Dict, List, Optional

import boto3
import botocore
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


def _download(url: str, fname: Path):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(desc=str(fname),
                                         total=total,
                                         unit='iB',
                                         unit_scale=True,
                                         unit_divisor=1024,
                                         ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


class OIModular(Dataset):
    _openimages_classes = {
        "Animal": {
            "Fish": ["Rays and skates", "Shark", "Goldfish"],
            "Reptile & Amphibian": ["Frog", "Crocodile", "Snake"],
            "Mammal": ["Cat", "Dog", "Elephant"],
            "Invertebrate": ["Snail", "Butterfly", "Spider"]
        },
        "Clothing": {
            "Luggage & bags": ["Handbag", "Suitcase", "Backpack"],
            "Footwear": ["High heels", "Boot", "Roller skates"],
            "Fashion accessory": ["Scarf", "Earrings", "Necklace"],
            "Hat": ["Sombrero", "Fedora", "Cowboy hat"]
        },
        "Vehicle": {
            "Aerial vehicle": ["Rocket", "Airplane", "Helicopter"],
            "Watercraft": ["Jet ski", "Canoe", "Barge"],
            "Land vehicle": ["Train", "Tank", "Motorcycle"]
        },
        "Plant": {
            "Flower": ["Lily", "Lavender", "Rose"],
            "Tree": ["Maple", "Palm tree", "Christmas tree"]
        },
        "Food": {
            "Dessert": ["Muffin", "Candy", "Cake"],
            "Fruit": ["Apple", "Orange", "Strawberry"],
            "Baked goods": ["Bread", "Cookie", "Pastry"],
            "Vegetable": ["Cucumber", "Pumpkin", "Tomato"]
        },

    }
    _OID_v5 = "https://storage.googleapis.com/openimages/v5/"
    _OID_v6 = "https://storage.googleapis.com/openimages/v6/"

    def __init__(self,
                 root: str,
                 num_classes: int = 5,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 seed: int = 0,
                 download=False):
        if num_classes not in [5, 17, 51]:
            raise ValueError(f"ModularOpenImages dataset has following "
                             f"number of classes 5, 17, 51 but {num_classes} "
                             f"was given.")

        self._root = Path(root)
        self._split = "train" if train else "test"
        self._transform = transform
        self._seed = seed
        # Number of samples per class that should be used as test samples

        if download:
            if not self._root.joinpath("complete").exists():
                self._labels_to_codes = self._class_label_codes()
                self._download_images()

                # Create empty file as flag that the dataset is downloaded
                self._root.joinpath("complete").touch()

        if num_classes == 5:
            self._test_size = 1020
            self._class_type = "Superclass"
            class_names = list(self._openimages_classes.keys())
        elif num_classes == 17:
            self._test_size = 300
            self._class_type = "Subclass"
            class_names = []
            for _, subclasses in self._openimages_classes.items():
                class_names.extend(list(subclasses.keys()))
        else:
            self._test_size = 100
            self._class_type = "Subsubclass"
            class_names = []
            for _, subclasses in self._openimages_classes.items():
                for _, subsubclasses in subclasses.items():
                    class_names.extend(subsubclasses)

        self._classes_to_idx = {name: idx for name, idx in
                                zip(class_names, range(len(class_names)))}

        self._df_img_ids_classes = pd.read_csv(
                self._root.joinpath("img_ids_and_classes.csv"))

        partition_to_imgs = self._get_split_idxs()[self._split]
        self.samples, self.targets = map(list, zip(*partition_to_imgs))
        self.targets = self.targets

        print(f"Loaded {self.__class__.__name__} ({self._split}) with "
              f"{len(self.samples)} samples")

        super().__init__()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filepath = self.samples[idx]
        label = self.targets[idx]

        sample = Image.open(filepath)

        if self._transform:
            sample = self._transform(sample)

        return sample, label

    def _class_label_codes(self) -> Dict:
        classes_csv = "oidv6-class-descriptions.csv"

        if not self._root.exists():
            self._root.mkdir(parents=True)

        # download the class descriptions CSV file to the specified
        # directory if not present
        descriptions_csv_file_path = self._root.joinpath(classes_csv)
        if not descriptions_csv_file_path.exists():
            # get the annotations CSV for the section
            url = self._OID_v6 + classes_csv
            _download(url, descriptions_csv_file_path)

        df_classes = pd.read_csv(descriptions_csv_file_path,
                                 header=0)

        # build dictionary of class labels to OpenImages class codes
        labels_to_codes = {}
        for _, subclasses in self._openimages_classes.items():
            for _, subsubclasses in subclasses.items():
                for subsubclass in subsubclasses:
                    # There is also color class Lavender
                    if subsubclass == "Lavender":
                        labels_to_codes[subsubclass.lower()] = \
                            df_classes.loc[df_classes["DisplayName"] ==
                                           "Lavender (Plant)"].values[0][0]
                    else:
                        labels_to_codes[subsubclass.lower()] = \
                            df_classes.loc[df_classes["DisplayName"] ==
                                           subsubclass].values[0][0]

        # return the labels to OpenImages codes dictionary
        return labels_to_codes

    def _download_one_image(self,
                            bucket,
                            img_id: str,
                            download_dir: Path):
        try:
            filepath = download_dir.joinpath(f"{img_id}.jpg")
            if not filepath.exists():
                bucket.download_file(f'train/{img_id}.jpg', str(filepath))
        except botocore.exceptions.ClientError:
            return False

        return True

    def _download_subsubclass_images(self,
                                     subsubclass_img_ids: np.array,
                                     subsubclass_dir: Path):
        if not subsubclass_dir.exists():
            subsubclass_dir.mkdir(parents=True)

        superclass = subsubclass_dir.parent.parent.name
        subclass = subsubclass_dir.parent.name
        subsubclass = subsubclass_dir.name

        # If susubclass has less samples than limit, we take all samples
        num_samples = len(subsubclass_img_ids)
        limit = 500
        if num_samples >= limit:
            num_samples = limit

        bucket = boto3.resource('s3', config=botocore.config.Config(
                signature_version=botocore.UNSIGNED)).Bucket(
                "open-images-dataset")
        progress_bar = tqdm(total=num_samples,
                            desc=f"Downloading images: Subsubclass: "
                                 f"{subsubclass_dir.stem}")
        downloaded_images = 0
        dfs_imgs = []
        with futures.ThreadPoolExecutor(max_workers=10) as executor:
            for img_id in subsubclass_img_ids:
                future = executor.submit(self._download_one_image, bucket,
                                         img_id, subsubclass_dir)
                downloaded = future.result()

                if downloaded or num_samples < limit:
                    progress_bar.update()

                if downloaded:
                    dfs_imgs.append(pd.DataFrame({"Superclass": [superclass],
                                                  "Subclass": [subclass],
                                                  "Subsubclass": [subsubclass],
                                                  "ImageID": [img_id]}))
                    downloaded_images += 1
                    if downloaded_images == limit:
                        break
        progress_bar.close()

        return pd.concat(dfs_imgs)

    def _download_images(self):
        if not self._root.exists():
            self._root.mkdir()

        # Download human verified labels
        # if split == "train":
        hv_labels_csv = f"oidv6-train-annotations-human-imagelabels.csv"
        # else:
        #     hv_labels_csv = f"{split}-annotations-human-imagelabels.csv"

        hv_labels_csv_file_path = self._root.joinpath(hv_labels_csv)
        if not hv_labels_csv_file_path.exists():
            # if split == "train":
            url = self._OID_v6 + hv_labels_csv
            # else:
            #     url = self._OID_v5 + hv_labels_csv

            print("Downloading human-verified labels for train data split.")
            _download(url, hv_labels_csv_file_path)

        df_hv_labels = pd.read_csv(hv_labels_csv_file_path, header=0)

        dfs_imgs = []
        for superclass, subclasses in self._openimages_classes.items():
            for subclass, subsubclasses in subclasses.items():
                for subsubclass in subsubclasses:
                    subsubclass_dir = self._root.joinpath(superclass, subclass,
                                                          subsubclass)
                    subsubclass_code = self._labels_to_codes[
                        subsubclass.lower()]

                    subsubclass_h_imgs = (df_hv_labels["LabelName"] ==
                                          subsubclass_code) & \
                                         (df_hv_labels["Confidence"] == 1)
                    subsubclass_img_ids = df_hv_labels.loc[
                                              subsubclass_h_imgs].loc[
                                          :, "ImageID"].values

                    dfs_imgs.append(self._download_subsubclass_images(
                            subsubclass_img_ids, subsubclass_dir))

        df_all_imgs = pd.concat(dfs_imgs)
        df_all_imgs.to_csv(self._root.joinpath("img_ids_and_classes.csv"),
                           index=False)

        return

    def _get_split_idxs(self) -> Dict[str, List[int]]:
        partition_to_imgs = {
            "train": [],
            "test": []
        }

        # Use this random seed to make partition consistent
        before_state = np.random.get_state()
        np.random.seed(self._seed)

        # ----------------- Create mapping: class -> (filepath, class_idx)
        class_to_imgs = dd(list)
        for idx, row in self._df_img_ids_classes.iterrows():
            filepath = str(self._root.joinpath(row.Superclass, row.Subclass,
                                               row.Subsubclass,
                                               f"{row.ImageID}.jpg"))

            class_name = row[self._class_type]
            class_to_imgs[class_name].append(
                    (filepath, self._classes_to_idx[class_name]))

        # Shuffle class_to_imgs
        for _, imgs in class_to_imgs.items():
            np.random.shuffle(imgs)

        for _, imgs in class_to_imgs.items():
            # A constant no. kept aside for evaluation
            partition_to_imgs["test"] += imgs[:self._test_size]
            # Train on remaining
            partition_to_imgs["train"] += imgs[self._test_size:]

        np.random.set_state(before_state)

        return partition_to_imgs
