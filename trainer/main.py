import datetime
from os import system

import pandas as pd
import tensorflow as tf
from google.cloud import bigquery

BUCKET = 'cloud-training-demos-ml'
PROJECT = 'qwiklabs-gcp-00-2f8873fbee83'
REGION = 'us-central1'
EXPORT_PATH = "../model"


def main():
    def preprocess(df):
        # clean up data we don't want to train on
        # in other words, users will have to tell us the mother's age
        # otherwise, our ML service won't work.
        # these were chosen because they are such good predictors
        # and because these are easy enough to collect
        df = df[df.weight_pounds > 0]
        df = df[df.mother_age > 0]
        df = df[df.gestation_weeks > 0]
        df = df[df.plurality > 0]

        # modify plurality field to be a string
        twins_etc = dict(zip([1, 2, 3, 4, 5],
                             ['Single(1)', 'Twins(2)', 'Triplets(3)', 'Quadruplets(4)', 'Quintuplets(5)']))
        df['plurality'].replace(twins_etc, inplace=True)

        # now create extra rows to simulate lack of ultrasound
        nous = df.copy(deep=True)
        nous.loc[nous['plurality'] != 'Single(1)', 'plurality'] = 'Multiple(2+)'
        nous['is_male'] = 'Unknown'

        df["is_male"] = df["is_male"].astype(str)
        return pd.concat([df, nous])

    ## Build a simple Keras DNN using its Functional API
    def rmse(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

    # Helper function to handle categorical columns
    def categorical_fc(name, values):
        return tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(name, values))

    def build_dnn_model():
        # input layer
        inputs = {
            "mother_age": tf.keras.layers.Input(name="mother_age", shape=(), dtype='float32'),
            "gestation_weeks": tf.keras.layers.Input(name="gestation_weeks", shape=(), dtype='float32'),
            "is_male": tf.keras.layers.Input(name="is_male", shape=(), dtype='string'),
            "plurality": tf.keras.layers.Input(name="plurality", shape=(), dtype='string')
        }

        # feature columns from inputs
        feature_columns = {
            "mother_age": tf.feature_column.numeric_column("mother_age"),
            "gestation_weeks": tf.feature_column.numeric_column("gestation_weeks"),
            "is_male": tf.feature_column.indicator_column(
                tf.feature_column
                    .categorical_column_with_vocabulary_list('is_male',
                                                             ['True', 'False', 'Unknown'])),
            "plurality": tf.feature_column.indicator_column(
                tf.feature_column
                    .categorical_column_with_vocabulary_list('plurality',
                                                             ['Single(1)', 'Twins(2)', 'Triplets(3)',
                                                              'Quadruplets(4)', 'Quintuplets(5)', 'Multiple(2+)']))
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
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_dnn_model()
    print(model.summary())

    query = """
    SELECT
      weight_pounds,
      is_male,
      mother_age,
      plurality,
      gestation_weeks,
      FARM_FINGERPRINT(CONCAT(CAST(YEAR AS STRING), CAST(month AS STRING))) AS hashmonth
    FROM
      publicdata.samples.natality
    WHERE year > 2000
    """

    trainQuery = "SELECT * FROM (" + query + ") WHERE ABS(MOD(hashmonth, 4)) < 3 AND RAND() < 0.0005"
    evalQuery = "SELECT * FROM (" + query + ") WHERE ABS(MOD(hashmonth, 4)) = 3 AND RAND() < 0.0005"
    traindf = bigquery.Client().query(trainQuery).to_dataframe()
    evaldf = bigquery.Client().query(evalQuery).to_dataframe()
    traindf = preprocess(traindf).reset_index()
    evaldf = preprocess(evaldf).reset_index()

    TRAIN_BATCH_SIZE = 32
    EPOCH = 10  # how many times to evaluate
    NUM_TRAIN_EXAMPLES = len(traindf) * EPOCH  # training dataset repeats, so it will wrap around
    NUM_EVAL_EXAMPLES = 10000  # enough to get a reasonable sample, but not so much that it slows down

    trainds = tf.data.Dataset.from_tensor_slices(
        (dict(traindf[["mother_age", "gestation_weeks", "is_male", "plurality"]]),
         traindf["weight_pounds"]))
    evaldf = tf.data.Dataset.from_tensor_slices(
        (dict(evaldf[["mother_age", "gestation_weeks", "is_male", "plurality"]]),
         evaldf["weight_pounds"]))

    trainds = trainds.shuffle(1000).batch(TRAIN_BATCH_SIZE).repeat()
    evaldf = evaldf.batch(1000).take(NUM_EVAL_EXAMPLES // 1000)

    # evalds = load_dataset('eval*', 1000, tf.estimator.ModeKeys.EVAL).take(NUM_EVAL_EXAMPLES // 1000)

    log_dir = "../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    steps_per_epoch = NUM_TRAIN_EXAMPLES // (TRAIN_BATCH_SIZE * EPOCH)

    history = model.fit(trainds,
                        validation_data=evaldf,
                        epochs=EPOCH,
                        steps_per_epoch=steps_per_epoch,
                        verbose=2,
                        callbacks=[tensorboard_callback])

    # Serving function that passes through keys
    @tf.function(input_signature=[{
        'is_male': tf.TensorSpec([None, ], dtype=tf.string, name='is_male'),
        'mother_age': tf.TensorSpec([None, ], dtype=tf.float32, name='mother_age'),
        'plurality': tf.TensorSpec([None, ], dtype=tf.string, name='plurality'),
        'gestation_weeks': tf.TensorSpec([None, ], dtype=tf.float32, name='gestation_weeks'),
        'key': tf.TensorSpec([None, ], dtype=tf.string, name='key')
    }])
    def serve(inputs):
        feats = inputs.copy()
        key = feats.pop('key')
        output = model(feats)
        return {'key': key, 'babyweight': output}

    tf.saved_model.save(model, EXPORT_PATH, signatures={'serving_default': serve})

    system("gsutil cp -R ../model gs://{}/".format(PROJECT))


if __name__ == "__main__":
    main()
