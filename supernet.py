import tensorflow as tf
import dataset_loaders.wakevisiondatasetloader

SUPERNET_NUM_EPOCHS_PRETRAIN = 30
SUPERNET_STEPS_PER_EPOCH = 1000


class SuperNet:
    def __init__(
        self,
        data,
        num_target_classes: int,
        model_optimizer: tf.keras.optimizers.Optimizer,
        model_loss_function: tf.keras.losses.Loss,
    ):
        self.num_target_classes = num_target_classes
        self.data = data
        self.model_optimizer = model_optimizer
        self.model_loss_function = model_loss_function

        input_shape = data.X_train.element_spec[0].shape[1:]
        print(f"Training new MobileNetV2 Backbone for input shape {input_shape}")

        supernet = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            weights=None,
            include_top=False,
        )

        x = tf.keras.layers.GlobalAveragePooling2D()(supernet.output)
        x = tf.keras.layers.Dense(self.num_target_classes, activation="softmax")(x)

        supernet_w_head = tf.keras.Model(inputs=supernet.input, outputs=x)

        supernet_w_head.compile(
            loss=model_loss_function,
            optimizer=model_optimizer,
            metrics=["Accuracy", "Precision", "Recall"],
        )

        supernet_w_head.fit(
            data.X_train,
            epochs=SUPERNET_NUM_EPOCHS_PRETRAIN,
            steps_per_epoch=SUPERNET_STEPS_PER_EPOCH,
            validation_data=data.X_val,
            verbose=2,
        )

        self.supernet = supernet_w_head

    def sample_subnet(
        self,
        stage3depth: int,
        stage4depth: int,
        stage5depth: int,
        stage6depth: int,
        stage7depth: int,
    ):
        assert 0 <= stage3depth <= 3
        assert 0 <= stage4depth <= 4
        assert 0 <= stage5depth <= 3
        assert 0 <= stage6depth <= 3
        assert 0 <= stage7depth <= 1

        if stage3depth == 0 and (
            stage4depth != 0 or stage5depth != 0 or stage6depth != 0 or stage7depth != 0
        ):
            return None
        if stage4depth == 0 and (
            stage5depth != 0 or stage6depth != 0 or stage7depth != 0
        ):
            return None
        if stage5depth == 0 and (stage6depth != 0 or stage7depth != 0):
            return None
        if stage6depth == 0 and stage7depth != 0:
            return None

        x = self.supernet.input

        # Stage 1 - Block "0" - keep all
        x = self.supernet.get_layer("expanded_conv_project_BN").output

        # Stage 2 - Block 1-2 - keep all
        x = self.supernet.get_layer("block_1_project_BN").output
        x = self.supernet.get_layer("block_2_project_BN").output

        # Stage 3 - Block 3-5 - Vary 0-3
        for i in range(3, 3 + stage3depth):
            x = self._include_block(x, i)

        # Stage 4 - Block 6-9 - Vary 0-4
        for i in range(6, 6 + stage4depth):
            x = self._include_block(x, i)

        # Stage 5 - Block 10-12 - Vary 0-3
        for i in range(10, 10 + stage5depth):
            x = self._include_block(x, i)

        # Stage 6 - Block 13-15 - Vary 0-3
        for i in range(13, 13 + stage6depth):
            x = self._include_block(x, i)

        # Stage 7 - Block 16 - Vary 0-1
        for i in range(16, 16 + stage7depth):
            x = self._include_block(x, i)
            # Final Conv
            x = self.supernet.get_layer("Conv_1")(x)
            x = self.supernet.get_layer("Conv_1_bn")(x)
            x = self.supernet.get_layer("out_relu")(x)

        # New classification head
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = tf.keras.layers.Dense(self.num_target_classes, activation="softmax")(x)

        subnetwork = tf.keras.Model(inputs=self.supernet.input, outputs=x)

        subnetwork.compile(
            loss=self.model_loss_function,
            optimizer=self.model_optimizer,
            metrics=["Accuracy", "Precision", "Recall"],
        )

        return subnetwork

    def _include_block(self, input, block_number: int):
        block_expand = self.supernet.get_layer(f"block_{block_number}_expand")(input)
        block_expand_BN = self.supernet.get_layer(f"block_{block_number}_expand_BN")(
            block_expand
        )
        block_expand_relu_or_pad = self.supernet.get_layer(
            f"block_{block_number}_expand_relu"
        )(block_expand_BN)
        if block_number in [1, 3, 6, 13]:
            block_expand_relu_or_pad = self.supernet.get_layer(
                f"block_{block_number}_pad"
            )(block_expand_relu_or_pad)
        block_depthwise = self.supernet.get_layer(f"block_{block_number}_depthwise")(
            block_expand_relu_or_pad
        )
        block_depthwise_BN = self.supernet.get_layer(
            f"block_{block_number}_depthwise_BN"
        )(block_depthwise)
        block_depthwise_relu = self.supernet.get_layer(
            f"block_{block_number}_depthwise_relu"
        )(block_depthwise_BN)
        block_project = self.supernet.get_layer(f"block_{block_number}_project")(
            block_depthwise_relu
        )
        block_project_BN = self.supernet.get_layer(f"block_{block_number}_project_BN")(
            block_project
        )
        return block_project_BN


if __name__ == "__main__":
    dataset = dataset_loaders.wakevisiondatasetloader.WakeVisionDatasetLoader()
    configured_dataset = dataset.configure_dataset(resolution=224, color="rgb")
    supervised_dataset = dataset.supervised_dataset(configured_dataset)

    x = SuperNet(supervised_dataset, 2, "adam", "categorical_crossentropy")
    y = x.sample_subnet(1, 1, 1, 1, 1)
