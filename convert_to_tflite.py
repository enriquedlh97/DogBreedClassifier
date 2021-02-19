import tensorflow as tf


saved_model_dir = "/content/gdrive/My Drive/Tec/Semestre 9.5 - Invierno/Evaluacion y Admin de proyectos/Proyecto/" +\
                  "MascotitasApp/efficientb3_acc_86_7"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

file_name = "/content/gdrive/My Drive/Tec/Semestre 9.5 - Invierno/Evaluacion y Admin de proyectos/Proyecto/" + \
            "MascotitasApp/b3_acc_86_7.tflite"

with open(file_name, "wb") as f:
    f.write(tflite_model)
