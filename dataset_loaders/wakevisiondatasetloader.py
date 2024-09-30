# A datasetloader to load the wake vision dataset
# Preprocessing code is adapted from the original wake vision preprocessing code found at https://github.com/harvard-edge/Wake_Vision.

# Standard Library Imports
from typing import Tuple, Any

# Third Party Imports
import datasets
import tensorflow as tf

# Local Imports
import datasetloader
from data import Data

batch_size = 128  # TODO: Make this a configurable option


class WakeVisionDatasetLoader(datasetloader.DatasetLoader):
    def __init__(
        self,
    ) -> None:
        self.hf_dataset = datasets.load_dataset(
            path="Harvard-Edge/Wake-Vision",
            cache_dir="/work3/emjn/huggingface/datasets",  # TODO: Make this a configurable option
        )
        self.hf_dataset["original_train_quality"] = self.hf_dataset["train_quality"]
        self.hf_dataset["original_validation"] = self.hf_dataset["validation"]
        self.hf_dataset["original_test"] = self.hf_dataset["test"]

    def configure_dataset(self, **kwargs: Any) -> Any:
        dataset = {}
        dataset["train_quality"] = self.hf_dataset[
            "original_train_quality"
        ].to_tf_dataset(columns=["image", "person"], shuffle=True)
        dataset["validation"] = self.hf_dataset["original_validation"].to_tf_dataset(
            columns=["image", "person"]
        )
        dataset["test"] = self.hf_dataset["original_test"].to_tf_dataset(
            columns=["image", "person"]
        )

        dataset["train_quality"] = self._preprocessing(
            dataset["train_quality"],
            resolution=kwargs["resolution"],
            color=kwargs["color"],
            train=True,
        )
        dataset["validation"] = self._preprocessing(
            dataset["validation"],
            resolution=kwargs["resolution"],
            color=kwargs["color"],
            train=False,
        )
        dataset["test"] = self._preprocessing(
            dataset["test"],
            resolution=kwargs["resolution"],
            color=kwargs["color"],
            train=False,
        )

        return dataset

    def supervised_dataset(self, input_data: Any, test_size: float = None) -> Data:
        if test_size != None:
            print(
                "Test size set for wake vision dataset. This option has no effect on the wake vision dataset due to an explicitly defined test set."
            )

        input_data["train_quality"] = input_data["train_quality"].map(
            self._prepare_supervised, num_parallel_calls=tf.data.AUTOTUNE
        )
        input_data["validation"] = input_data["validation"].map(
            self._prepare_supervised, num_parallel_calls=tf.data.AUTOTUNE
        )
        input_data["test"] = input_data["test"].map(
            self._prepare_supervised, num_parallel_calls=tf.data.AUTOTUNE
        )

        return Data(
            X_train=input_data["train_quality"]
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE),
            X_test=input_data["test"].batch(batch_size).prefetch(tf.data.AUTOTUNE),
            X_val=input_data["validation"].batch(batch_size).prefetch(tf.data.AUTOTUNE),
        )

    def _preprocessing(self, ds_split, resolution, color, train=False):
        ds_split = ds_split.map(
            self._cast_images_to_float32, num_parallel_calls=tf.data.AUTOTUNE
        )

        input_shape = (resolution, resolution, 3)

        if train:
            # Repeat indefinitely and shuffle the dataset
            ds_split = ds_split.shuffle(1000, reshuffle_each_iteration=True)
            # inception crop
            ds_split = ds_split.map(
                self._inception_crop, num_parallel_calls=tf.data.AUTOTUNE
            )
            # resize
            resize = lambda ds_entry: self._resize(ds_entry, input_shape)
            ds_split = ds_split.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
            # flip
            ds_split = ds_split.map(
                self._random_flip_lr, num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            # resize small
            resize_small = lambda ds_entry: self._resize_small(ds_entry, input_shape)
            ds_split = ds_split.map(resize_small, num_parallel_calls=tf.data.AUTOTUNE)
            # center crop
            center_crop = lambda ds_entry: self._center_crop(ds_entry, input_shape)
            ds_split = ds_split.map(center_crop, num_parallel_calls=tf.data.AUTOTUNE)

        if color == "monochrome":
            ds_split = ds_split.map(
                self._grayscale, num_parallel_calls=tf.data.AUTOTUNE
            )

        # Use the official mobilenet preprocessing to normalize images
        ds_split = ds_split.map(
            self._mobilenet_preprocessing_wrapper, num_parallel_calls=tf.data.AUTOTUNE
        )

        ds_split = ds_split.map(
            self._to_one_hot_encoding, num_parallel_calls=tf.data.AUTOTUNE
        )

        return ds_split

    def _to_one_hot_encoding(self, ds_entry):
        ds_entry["person"] = tf.one_hot(ds_entry["person"], 2)
        return ds_entry

    def _cast_images_to_float32(self, ds_entry):
        ds_entry["image"] = tf.cast(ds_entry["image"], tf.float32)
        return ds_entry

    def _inception_crop(self, ds_entry):
        """
        Inception-style crop is a random image crop (its size and aspect ratio are
        random) that was used for training Inception models, see
        https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf.
        """
        image = ds_entry["image"]
        begin, crop_size, _ = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            tf.zeros([0, 0, 4], tf.float32),
            area_range=(0.6, 1.0),  # TODO look into if this is too small of a min area
            min_object_covered=0,  # Don't enforce a minimum area.
            use_image_if_no_bounding_boxes=True,
        )
        crop = tf.slice(image, begin, crop_size)
        # Unfortunately, the above operation loses the depth-dimension. So we need
        # to restore it the manual way.
        crop.set_shape([None, None, image.shape[-1]])
        ds_entry["image"] = crop
        return ds_entry

    def _center_crop(self, ds_entry, input_shape):
        # crop image to desired size
        image = ds_entry["image"]
        h, w = input_shape[0], input_shape[1]
        dy = (tf.shape(image)[0] - h) // 2
        dx = (tf.shape(image)[1] - w) // 2
        ds_entry["image"] = tf.image.crop_to_bounding_box(image, dy, dx, h, w)
        return ds_entry

    def _resize(self, ds_entry, input_shape):
        ds_entry["image"] = tf.image.resize(ds_entry["image"], input_shape[:2])
        return ds_entry

    def _resize_small(self, ds_entry, input_shape):
        # Resizes the smaller side to `smaller_size` keeping aspect ratio.
        image = ds_entry["image"]
        smaller_size = input_shape[0]  # Assuming target shape is square

        h, w = tf.shape(image)[0], tf.shape(image)[1]

        # Figure out the necessary h/w.
        ratio = tf.cast(smaller_size, tf.float32) / tf.cast(
            tf.minimum(h, w), tf.float32
        )
        h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
        w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)

        dtype = image.dtype
        image = tf.image.resize(image, (h, w), method="area", antialias=False)
        ds_entry["image"] = tf.cast(image, dtype)
        return ds_entry

    def _random_flip_lr(self, ds_entry):
        ds_entry["image"] = tf.image.random_flip_left_right(ds_entry["image"])
        return ds_entry

    def _grayscale(self, ds_entry):
        ds_entry["image"] = tf.image.rgb_to_grayscale(ds_entry["image"])
        return ds_entry

    def _mobilenet_preprocessing_wrapper(self, ds_entry):
        ds_entry["image"] = tf.keras.applications.mobilenet_v2.preprocess_input(
            ds_entry["image"]
        )
        return ds_entry

    def _prepare_supervised(self, ds_entry):
        return (ds_entry["image"], ds_entry["person"])


if __name__ == "__main__":
    dataset_loader = WakeVisionDatasetLoader()
    dataset = dataset_loader.configure_dataset(resolution=400, color="monochrome")
    s_dataset = dataset_loader.supervised_dataset(dataset, test_size=0.4)
