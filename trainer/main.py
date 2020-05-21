import datetime

import tensorflow as tf
from tensorflow_io.bigquery import BigQueryClient

PROJECT_BUCKET = "YOUR_PROJECT_ID"
REGION = "us-central1"
EXPORT_PATH = "gs://{}/model".format(PROJECT_BUCKET)
CHECK_POINT_PATH = "gs://{}/checkpoint/checkpoint".format(PROJECT_BUCKET)
# EXPORT_PATH = "../model"


def main():
    def transfrom_row(row_dict):
        datadict = {column: tensor
                    for (column, tensor) in row_dict.items()
                    }
        weight_pounds = datadict.pop("weight_pounds")
        datadict["is_male"] = tf.cast(datadict["is_male"], tf.int8)
        return (datadict, weight_pounds)

    def read_bigquery():
        # publicdata.samples.natality
        tensorflow_io_bigquery_client = BigQueryClient()
        read_session = tensorflow_io_bigquery_client.read_session(
            "projects/{}".format(PROJECT_BUCKET),
            "bigquery-public-data", "natality", "samples",
            [
                "weight_pounds",
                "is_male",
                "mother_age",
                "plurality",
                "gestation_weeks"
            ],
            row_restriction="""
                year > 2000 AND
                weight_pounds > 0 AND
                gestation_weeks > 0 AND
                plurality > 0
            """,
            output_types=[
                tf.dtypes.double,
                tf.dtypes.bool,
                tf.dtypes.int64,
                tf.dtypes.int64,
                tf.dtypes.int64
            ],
        )

        dataset = read_session.parallel_read_rows()
        transformed_ds = dataset.map(transfrom_row)
        return transformed_ds

    dataset = read_bigquery()

    print(dataset)

    # def preprocess(df):
    #     # clean up data we don't want to train on
    #     # in other words, users will have to tell us the mother's age
    #     # otherwise, our ML service won't work.
    #     # these were chosen because they are such good predictors
    #     # and because these are easy enough to collect
    #     df = df[df.weight_pounds > 0]
    #     df = df[df.mother_age > 0]
    #     df = df[df.gestation_weeks > 0]
    #     df = df[df.plurality > 0]
    #
    #     # modify plurality field to be a string
    #     twins_etc = dict(zip([1, 2, 3, 4, 5],
    #                          ['Single(1)', 'Twins(2)', 'Triplets(3)', 'Quadruplets(4)', 'Quintuplets(5)']))
    #     df['plurality'].replace(twins_etc, inplace=True)
    #
    #     # now create extra rows to simulate lack of ultrasound
    #     nous = df.copy(deep=True)
    #     nous.loc[nous['plurality'] != 'Single(1)', 'plurality'] = 'Multiple(2+)'
    #     nous['is_male'] = 'Unknown'
    #
    #     df["is_male"] = df["is_male"].astype(str)
    #     return pd.concat([df, nous])

    ## Build a simple Keras DNN using its Functional API
    def rmse(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

    def build_dnn_model():
        # input layer
        inputs = {
            "mother_age": tf.keras.layers.Input(name="mother_age", shape=(), dtype='int64'),
            "gestation_weeks": tf.keras.layers.Input(name="gestation_weeks", shape=(), dtype='int64'),
            "is_male": tf.keras.layers.Input(name="is_male", shape=(), dtype='int8'),
            "plurality": tf.keras.layers.Input(name="plurality", shape=(), dtype='int64')
        }

        # feature columns from inputs
        feature_columns = {
            "mother_age": tf.feature_column.numeric_column("mother_age"),
            "gestation_weeks": tf.feature_column.numeric_column("gestation_weeks"),
            "is_male": tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list("is_male", [0, 1])
            ),
            "plurality": tf.feature_column.numeric_column("plurality")
        }

        # the constructor for DenseFeatures takes a list of numeric columns
        # The Functional API in Keras requires that you specify: LayerConstructor()(inputs)
        dnn_inputs = tf.keras.layers.DenseFeatures(feature_columns.values())(inputs)

        # two hidden layers of [64, 32] just in like the BQML DNN
        h1 = tf.keras.layers.Dense(64, activation='relu', name='h1')(dnn_inputs)
        h2 = tf.keras.layers.Dense(32, activation='relu', name='h2')(h1)

        # final output is a linear activation because this is regression
        output = tf.keras.layers.Dense(1, activation='linear', name='babyweight')(h2)

        model = tf.keras.models.Model(inputs, output)
        model.compile(optimizer='adam', loss='mse', metrics=[rmse, 'mse'])
        return model

    print("Here is our DNN architecture so far:\n")

    # note how to use strategy to do distributed training
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    model = build_dnn_model()
    # model.load_weights(CHECK_POINT_PATH)
    # model = tf.keras.models.load_model(EXPORT_PATH)
    print(model.summary())

    # query = """
    # SELECT
    #   weight_pounds,
    #   is_male,
    #   mother_age,
    #   plurality,
    #   gestation_weeks,
    #   FARM_FINGERPRINT(CONCAT(CAST(YEAR AS STRING), CAST(month AS STRING))) AS hashmonth
    # FROM
    #   publicdata.samples.natality
    # WHERE year > 1990
    # """

    # trainQuery = "SELECT * FROM (" + query + ") WHERE ABS(MOD(hashmonth, 4)) < 3 AND RAND() < 0.01"
    # evalQuery = "SELECT * FROM (" + query + ") WHERE ABS(MOD(hashmonth, 4)) = 3 AND RAND() < 0.01"
    # traindf = bigquery.Client().query(trainQuery).to_dataframe()
    # evaldf = bigquery.Client().query(evalQuery).to_dataframe()
    # traindf = preprocess(traindf).reset_index()
    # evaldf = preprocess(evaldf).reset_index()

    TRAIN_BATCH_SIZE = 512
    EPOCH = 10  # how many times to evaluate
    NUM_TRAIN_EXAMPLES = 100000 * EPOCH  # training dataset repeats, so it will wrap around
    NUM_EVAL_EXAMPLES = 10000  # enough to get a reasonable sample, but not so much that it slows down

    # trainds = tf.data.Dataset.from_tensor_slices(
    #     (dict(traindf[["mother_age", "gestation_weeks", "is_male", "plurality"]]),
    #      traindf["weight_pounds"]))
    # evaldf = tf.data.Dataset.from_tensor_slices(
    #     (dict(evaldf[["mother_age", "gestation_weeks", "is_male", "plurality"]]),
    #      evaldf["weight_pounds"]))

    # trainds = trainds.shuffle(1000).batch(TRAIN_BATCH_SIZE).repeat()
    # evaldf = evaldf.batch(1000).take(NUM_EVAL_EXAMPLES // 1000)

    trainds = read_bigquery().shuffle(10000).batch(TRAIN_BATCH_SIZE).repeat()
    evaldf = read_bigquery().batch(1000).take(NUM_EVAL_EXAMPLES // 1000)

    # evalds = load_dataset('eval*', 1000, tf.estimator.ModeKeys.EVAL).take(NUM_EVAL_EXAMPLES // 1000)

    log_dir = "../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    steps_per_epoch = NUM_TRAIN_EXAMPLES // (TRAIN_BATCH_SIZE * EPOCH)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        verbose=1,
        filepath=CHECK_POINT_PATH,
        save_weights_only=True,
        mode='auto',
        save_best_only=True)

    model.fit(trainds,
              validation_data=evaldf,
              epochs=EPOCH,
              steps_per_epoch=steps_per_epoch,
              verbose=2,
              callbacks=[tensorboard_callback, model_checkpoint_callback])

    # Serving function that passes through keys
    @tf.function(input_signature=[{
        'is_male': tf.TensorSpec([None, ], dtype=tf.int8, name='is_male'),
        'mother_age': tf.TensorSpec([None, ], dtype=tf.int64, name='mother_age'),
        'plurality': tf.TensorSpec([None, ], dtype=tf.int64, name='plurality'),
        'gestation_weeks': tf.TensorSpec([None, ], dtype=tf.int64, name='gestation_weeks'),
        'key': tf.TensorSpec([None, ], dtype=tf.string, name='key')
    }])
    def serve(inputs):
        feats = inputs.copy()
        key = feats.pop('key')
        output = model(feats)
        return {'key': key, 'babyweight': output}

    model.save(EXPORT_PATH, signatures={'serving_default': serve})


if __name__ == "__main__":
    main()
