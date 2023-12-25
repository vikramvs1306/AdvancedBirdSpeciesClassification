import tensorflow as tf

def parse_examples(serialized_examples, feature_description):
    examples = tf.io.parse_example(serialized_examples, feature_description)
    images = examples.pop("image")
    resized_images = tf.image.resize_with_pad(
        tf.cast(tf.io.decode_jpeg(images, channels=3), tf.float32), 299, 299
    )
    targey_key = list(examples.keys())[0]
    targets = examples[targey_key]
    return resized_images, targets