import argparse
import sys
import time

def generate_yolov1_dataloader(grid_size, class_size, batch_size):
    dataset_ls = tf.data.Dataset.list_files('.annotations/*.anot')

    def parse_annotation_and_image(anot):
        annotation = tf.io.read_file(anot)
        annotation = tf.strings.split(annotation, "\n" if sys.platform == 'linux' else '\r\n')

        filename = annotation[0]
        image_size = tf.strings.split(annotation[1], " ")
        image_size = tf.strings.to_number(image_size, out_type=tf.int32)

        # Step 1: Generate y_true
        object_list = annotation[2:]

        def create_label_box(object_list):
            y_true = tf.zeros(shape=(grid_size * grid_size, 5 + class_size))
            for obj in object_list:
                segments = tf.strings.split(obj, " ")
                if segments[0] == "":
                    continue
                segments = tf.strings.to_number(segments, out_type=tf.float32)

                x_center, y_center, width, height = segments[:4]
                class_one_hot = segments[4:]

                # dynamic range (0, 6] falls into grid!
                grid_x, grid_y = int(x_center * grid_size), int(y_center * grid_size)
                grid_scope_x_center = x_center * grid_size - grid_x
                grid_scope_y_center = y_center * grid_size - grid_y
                if y_true[grid_y * grid_size + grid_x, 4] == 1.:
                    # print("Duplicate grid values! skipping ...")
                    continue
                # tf.print('grid_x', grid_x, 'grid_y', grid_y, 'target_width', width, 'target_height', height)

                single_grid_slice = [grid_scope_x_center, grid_scope_y_center, width, height, 1.]
                single_grid_slice = tf.concat([single_grid_slice, class_one_hot], axis=0)
                single_grid_slice = tf.expand_dims(single_grid_slice, axis=0)

                # tf.print(single_grid_slice)
                # tf.print(tf.shape(single_grid_slice))

                y_true = tf.tensor_scatter_nd_update(y_true, [[grid_y * grid_size + grid_x]], single_grid_slice)

            return y_true

        y_true = tf.py_function(func=create_label_box, inp=[object_list], Tout=tf.float32)
        y_true = tf.reshape(y_true, shape=(grid_size, grid_size, 5 + class_size))

        # Step 2: Load image
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [448, 448])

        return image, y_true

    voc2012_yolo_ds = dataset_ls.map(parse_annotation_and_image)
    voc2012_yolo_ds_batch = voc2012_yolo_ds.batch(batch_size)
    voc2012_yolo_ds_batch_valid = voc2012_yolo_ds_batch.take(2)
    voc2012_yolo_ds_batch_x = voc2012_yolo_ds_batch.skip(2)
    return voc2012_yolo_ds_batch_x, voc2012_yolo_ds_batch_valid


def generate_mobilenet_model(metrics, loss_function=None, verbose=False):
    import tensorflow.keras.layers as layers

    class YoloFinalLayer(tf.keras.layers.Layer):
        def __init__(self, num_bbox):
            super(YoloFinalLayer, self).__init__()
            self.bbox_offset = num_bbox * 4

        def call(self, input):
            y_pred_bbox = input[:, :, :, :self.bbox_offset]
            y_pred_conf = input[:, :, :, self.bbox_offset:(self.bbox_offset + 1)]
            y_pred_class = input[:, :, :, (self.bbox_offset + 1):]

            y_pred_bbox = tf.sigmoid(y_pred_bbox)
            # y_pred_bbox = tf.nn.leaky_relu(y_pred_class, alpha=.1)
            y_pred_conf = tf.sigmoid(y_pred_conf)
            # y_pred_class = tf.nn.leaky_relu(y_pred_class, alpha=.1)
            # y_pred_class = tf.sigmoid(y_pred_class)
            y_pred_class = tf.nn.softmax(y_pred_class, axis=3)

            return tf.concat([y_pred_bbox, y_pred_conf, y_pred_class], axis=3)

    model = tf.keras.Sequential([
        layers.Input(shape=(448, 448, 3)),

        layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same'),
        layers.LeakyReLU(alpha=.1),
        layers.MaxPool2D(pool_size=(2, 2), strides=2),
        layers.LeakyReLU(alpha=.1),

        layers.Conv2D(filters=192, kernel_size=(3, 3), padding='same'),
        layers.LeakyReLU(alpha=.1),
        layers.MaxPool2D(pool_size=(2, 2), strides=2),
        layers.LeakyReLU(alpha=.1),

        layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same'),
        layers.LeakyReLU(alpha=.1),
        layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same'),
        layers.LeakyReLU(alpha=.1),
        layers.Conv2D(filters=256, kernel_size=(1, 1), padding='same'),
        layers.LeakyReLU(alpha=.1),
        layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
        layers.MaxPool2D(pool_size=(2, 2), strides=2),
        layers.LeakyReLU(alpha=.1),

        layers.Conv2D(filters=256, kernel_size=(1, 1), padding='same'),
        layers.LeakyReLU(alpha=.1),
        layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
        layers.LeakyReLU(alpha=.1),
        layers.Conv2D(filters=256, kernel_size=(1, 1), padding='same'),
        layers.LeakyReLU(alpha=.1),
        layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
        layers.LeakyReLU(alpha=.1),
        layers.Conv2D(filters=256, kernel_size=(1, 1), padding='same'),
        layers.LeakyReLU(alpha=.1),
        layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
        layers.LeakyReLU(alpha=.1),
        layers.Conv2D(filters=256, kernel_size=(1, 1), padding='same'),
        layers.LeakyReLU(alpha=.1),
        layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same'),
        layers.LeakyReLU(alpha=.1),

        layers.Conv2D(filters=512, kernel_size=(1, 1), padding='same'),
        layers.LeakyReLU(alpha=.1),
        layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same'),
        layers.LeakyReLU(alpha=.1),
        layers.MaxPool2D(pool_size=(2, 2), strides=2),
        layers.LeakyReLU(alpha=.1),

        layers.Conv2D(filters=512, kernel_size=(1, 1), padding='same'),
        layers.LeakyReLU(alpha=.1),
        layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same'),
        layers.LeakyReLU(alpha=.1),
        layers.Conv2D(filters=512, kernel_size=(1, 1), padding='same'),
        layers.LeakyReLU(alpha=.1),
        layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same'),
        layers.LeakyReLU(alpha=.1),

        layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same'),
        layers.LeakyReLU(alpha=.1),
        layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=2, padding='same'),
        layers.LeakyReLU(alpha=.1),

        layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same'),
        layers.LeakyReLU(alpha=.1),
        layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same'),
        layers.LeakyReLU(alpha=.1),

        layers.Flatten(),
        layers.Dense(units=4096),
        layers.LeakyReLU(alpha=.1),
        layers.Dense(units=1225),
        layers.Reshape(target_shape=(7, 7, 25)),
        YoloFinalLayer(num_bbox=1)  # does Sigmoid or Softmax for each bboxes
    ])

    if loss_function is None:
        print("WARNING: No loss function applied!")
        loss_function = 'mean_squared_error'

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)  #TODO 고질적인 문제는 Learning rate였다
    
    model.compile(
        loss=loss_function,
        optimizer=optimizer,
        metrics=metrics
    )  # ! Q1: Adam을 쓰는 이유?!
    
    if verbose:
        model.summary()

    return model


def generate_yolov1_loss():
    def convert_cwh_to_xyminmax(bbox, cell_size):
        xy_center = bbox[:, :, :, :2]
        width_height = bbox[:, :, :, 2:]

        index_table_x = tf.repeat(tf.expand_dims(tf.range(cell_size), -1), cell_size, -1)
        index_table_y = tf.reshape(tf.tile(tf.range(cell_size), [cell_size]), (cell_size, cell_size))
        index_table = tf.cast(tf.concat([tf.expand_dims(index_table_x, -1), tf.expand_dims(index_table_y, -1)], -1), tf.float32)
        index_table_include_batch = tf.expand_dims(index_table, 0)

        xy_global_perspective = xy_center + index_table_include_batch
        xy_min = xy_global_perspective - (width_height / 2)
        xy_max = xy_global_perspective + (width_height / 2)

        return xy_min, xy_max

    def iou_between(a_xy_min, a_xy_max, b_xy_min, b_xy_max):
        x_max_of_mins = tf.reduce_max(tf.concat([a_xy_min[:, :, :, 0:1], b_xy_min[:, :, :, 0:1]], -1), -1)
        y_max_of_mins = tf.reduce_max(tf.concat([a_xy_min[:, :, :, 1:2], b_xy_min[:, :, :, 1:2]], -1), -1)
        x_min_of_maxes = tf.reduce_min(tf.concat([a_xy_max[:, :, :, 0:1], b_xy_max[:, :, :, 0:1]], -1), -1)
        y_min_of_maxes = tf.reduce_min(tf.concat([a_xy_max[:, :, :, 1:2], b_xy_max[:, :, :, 1:2]], -1), -1)

        # tf.print('x_max_of_mins', tf.shape(x_max_of_mins))

        intersaction_width = x_min_of_maxes - x_max_of_mins
        intersaction_width = tf.where(intersaction_width < 0., 0., intersaction_width)
        intersaction_height = y_min_of_maxes - y_max_of_mins
        intersaction_height = tf.where(intersaction_height < 0., 0., intersaction_height)

        # tf.print('intersaction_width', tf.shape(intersaction_width))
        # tf.print('intersaction_height', tf.shape(intersaction_height))

        intersaction_area = tf.multiply(intersaction_width, intersaction_height)

        a_area = (a_xy_max[:, :, :, 0] - a_xy_min[:, :, :, 0]) * (a_xy_max[:, :, :, 1] - a_xy_min[:, :, :, 1])
        b_area = (b_xy_max[:, :, :, 0] - b_xy_min[:, :, :, 0]) * (b_xy_max[:, :, :, 1] - b_xy_min[:, :, :, 1])

        iou = intersaction_area / (a_area + b_area - intersaction_area)
        # tf.print('a_area', tf.shape(a_area))
        # tf.print('b_area', tf.shape(b_area))
        # tf.print('intersaction_area', tf.shape(intersaction_area))

        # b_area shows [16, 7, 7] and intersaction_area shows [2, 16, 7] ??

        return iou

    def loss(y_true, y_pred):  # Loss inputs with batch size!
        lambda_coord = 5.
        lambda_noobj = .5

        y_pred_bbox = y_pred[:, :, :, :4]
        y_pred_conf = y_pred[:, :, :, 4:5]
        y_pred_class = y_pred[:, :, :, 5:]

        y_true_bbox = y_true[:, :, :, :4]
        y_true_conf = y_true[:, :, :, 4:5]  # always 1 !
        y_true_class = y_true[:, :, :, 5:]

        y_pred_bbox_xy_min, y_pred_bbox_xy_max = convert_cwh_to_xyminmax(y_pred_bbox, 7)
        y_true_bbox_xy_min, y_true_bbox_xy_max = convert_cwh_to_xyminmax(y_true_bbox, 7)
        bbox_iou = iou_between(y_pred_bbox_xy_min, y_pred_bbox_xy_max, y_true_bbox_xy_min, y_true_bbox_xy_max)

        object_exist_mask = tf.cast(y_true_conf == 1., tf.float32)
        object_non_exist_mask = 1. - object_exist_mask
        # object_responsible_mask = tf.expand_dims(tf.cast(bbox_iou > 0.5, tf.float32), -1)  #! TODO Check each cell's IoU with GT - highest returns Responsible Flag!
        # object_non_responsible_mask = 1. - object_responsible_mask
        # cell 내에 bbox가 1개밖에 없으므로 object_responsible_mask == object_exist_mask이다.
        # 그렇지 않은 경우 Cell n개와 GT BBox를 비교해서 n개의 셀중 가장 GT와 IOU가 높은 bbox를 선택한다.
        object_responsible_mask = object_exist_mask
        object_non_responsible_mask = object_non_exist_mask

        bbox_xy_loss = tf.square(y_pred_bbox[:, :, :, :2] - y_true_bbox[:, :, :, :2])
        bbox_wh_loss = tf.square(tf.sqrt(y_pred_bbox[:, :, :, 2:4] + 1e-15) - tf.sqrt(y_true_bbox[:, :, :, 2:4] + 1e-15))

        conf_obj_loss = tf.square(y_true_conf - y_pred_conf)
        conf_noobj_loss = tf.square(y_true_conf - y_pred_conf)

        bbox_loss = lambda_coord * object_responsible_mask * (bbox_xy_loss + bbox_wh_loss)
        conf_loss = (object_responsible_mask * conf_obj_loss) + \
                    (lambda_noobj * object_non_responsible_mask * conf_noobj_loss)
        class_loss = object_exist_mask * tf.square(y_pred_class - y_true_class)

        return tf.reduce_sum(bbox_loss) + tf.reduce_sum(conf_loss) + tf.reduce_sum(class_loss)

    return loss

def tf_v1_compat_memory_management():
    # Memory Pre-configuration
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9))
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--width-multiplier', '-w', type=float, default=1.,
                        help='Width multiplier of MobileNet (could be one of 1.0, 0.75, 0.5, 0.24)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging (Model summary and Tensorflow logics, etc)')
    parser.add_argument('--batch-size', '-b', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--input-resolution', '-i', type=int, default=224,
                        help='Input resolution (applied with resolution multiplier, one of 224, 192, 160, 128)')
    args = parser.parse_args()

    import os

    if not args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    print("Begin TF Library Load")
    import tensorflow as tf
    tf_v1_compat_memory_management()

    print("End TF Library Load")

    if args.width_multiplier not in [1., 0.75, 0.5, 0.25]:
        print(f"Invalid argument: args.width_multiplier={args.width_multiplier}")
        exit(-1)

    if args.input_resolution not in [224, 192, 160, 128]:
        print(f"Invalid argument: args.input_resolution={args.input_resolution}")
        exit(-1)

    print("Model Init")
    loss_function = generate_yolov1_loss()
    dataloader, dataloader_valid = generate_yolov1_dataloader(grid_size=7, class_size=20, batch_size=args.batch_size)
    model = generate_mobilenet_model([], loss_function, args.verbose)
    print("Done Model Init")

    # model.summary()

    def lr_scheduler(epoch, lr):
        # return lr * tf.math.exp(-0.1)
        return lr * tf.math.exp(-0.05)

    history = model.fit(
        x=dataloader,
        validation_data=dataloader_valid,
        epochs=600,
        shuffle=True,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath='S:\\.checkpoints\\weights.epoch{epoch:02d}-loss{loss:.2f}.ckpt',
                save_weights_only=True,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.LearningRateScheduler(
                lr_scheduler, verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f'.logs/run-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))}',
                update_freq=8,
                profile_batch=0,
            )
        ]
    )
