Facial Emotion Recognition using Deep Learning

DATA SET:
---------
- Download Data Set: `fer2013.bin` (63M) and `test_batch.bin` (7.9M) from the ‘Datasets’ folder in the repository

  Image Properties: `Size of an image` - 48 x 48 pixels (2304 bytes), `Size of a label` - number in (0..6) (1 byte) (0=Angry, 1=Fear, 2=Happy, 3=Sad, 4=Disgust, 5=Surprise, 6=Neutral).


-------------
- Install `TensorFlow`: https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#pip-installation
- Run `python fer2013_train.py`
- Run `python fer2013_eval.py` on fer2013.bin data (Training Precision)
- Run `python fer2013_eval.py` on test_batch.bin data (Evaluation Precision)