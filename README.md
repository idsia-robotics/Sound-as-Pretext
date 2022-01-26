# Learning Visual Localization of a Quadrotor using its Noise as Self-Supervision

*Mirko Nava, Antonio Paolillo, Jerome Guzzi, Luca Maria Gambardella, and Alessandro Giusti*

Dalle Molle Institute for Artificial Intelligence, USI-SUPSI, Lugano (Switzerland)

### Abstract
We introduce an approach to train neural network models for visual object localization using a small training set, labeled with ground truth object positions, and a large unlabeled one.
We assume that the object to be localized emits sound, which is perceived by a microphone rigidly affixed to the camera.
This information is used as the target of a cross-modal pretext task: predicting sound features from camera frames.
By solving the pretext task, the model draws self-supervision from visual and auditory data. 
The approach is well suited to robot learning: we instantiate it to localize a small quadrotor from 128x80 pixel images acquired by a ground robot. 
Experiments on a separate testing set show that introducing the auxiliary pretext task yields large performance improvements:
the Mean Absolute Error (MAE) of the estimated image coordinates of the target is reduced from 7 to 4 pixels; the MAE of the estimated distance is reduced from 28 cm to 14 cm.
A model that has access to labels for the entire training set yields a MAE of 2 pixels and 11 cm, respectively.


![Sound as Pretext](https://github.com/idsia-robotics/Sound-as-Pretext/blob/main/code/data/out/Intro.png)
Figure 1: *Given an image from the ground robot camera, the model estimates the relative position of the drone; this is the **end task**, learned by minimizing a regression end loss on few training frames for which the true relative position is known.
We show that simultaneously learning to predict audio features (**pretext task**), which are known in all training frames, yields dramatic performance improvements for the end task.*


![Regression Performance on the testing set](https://github.com/idsia-robotics/Sound-as-Pretext/blob/main/code/data/out/results.png)
Figure 2: *End Task Regression Performance on the testing set.
On the left side, we compare ground truth (x axis) and predictions (y axis) for different models (columns) and variables (rows).
On the right, predictions on 30s of the testing set.
Between seconds 17 and 20 the drone exits of the camera FOV, causing all models to temporarily fail.*

The PDF of the article is available in Open Access [here]( https://doi.org/10.1109/LRA.2022.3143565).

### Bibtex will be displayed here later

```properties
@article{nava2022learning,
  author={M. {Nava} and A. {Paolillo} and J. {Guzzi} and L. M. {Gambardella} and A. {Giusti}},
  journal={IEEE Robotics and Automation Letters}, 
  title={Learning Visual Localization of a Quadrotor Using its Noise as Self-Supervision}, 
  year={2022},
  volume={7},
  number={2},
  pages={2218-2225},
  doi={10.1109/LRA.2022.3143565}
}
```

### Video

A video of the approach can be downloaded by clicking [here](https://github.com/idsia-robotics/Sound-as-Pretext/raw/main/code/data/out/sap.mp4).
<!--[![Learning Visual Object Localization from Few Labeled Examples using Sound Prediction as a Pretext Task](https://github.com/idsia-robotics/Sound-as-Pretext/blob/main/code/data/out/video.gif)](https://youtu.be/XXXXXXX)-->

### Code

The entire codebase, training scripts and pre-trained models are avaliable [here](https://github.com/idsia-robotics/Sound-as-Pretext/tree/main/code).

### Dataset

The dataset divided into [unlabeled training-set](https://drive.switch.ch/index.php/s/RSz7jRiHrSwf54p), [labeled training-set](https://drive.switch.ch/index.php/s/BfQwbzCf4gTGJ7T), [validation-set](https://drive.switch.ch/index.php/s/qN4NO9296K6ry1t), and [test-set](https://drive.switch.ch/index.php/s/7myEJA7E4zYQlVz) is avaiable through the relative links.
