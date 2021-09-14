# Learning Visual Object Localization from Few Labeled Examples using Sound Prediction as a Pretext Task

*Mirko Nava, Antonio Paolillo, Jerome Guzzi, Luca Maria Gambardella, and Alessandro Giusti*

Dalle Molle Institute for Artificial Intelligence, USI-SUPSI, Lugano (Switzerland)

### Abstract
We introduce an approach to train neural network models for visual object localization, leveraging a dataset largely composed of unlabeled examples.
We assume that the object to be localized emits sound, which is picked up by a microphone rigidly affixed to the body of the robot.
This information is used as the target of a cross-modal pretext task: predicting sound features from a camera frame.
By solving the pretext task, a model draws supervision from both visual and auditorial information, enhancing its performance.
This approach is well suited to many practical applications in self-supervised robot learning: we instantiate it to localize a small quadrotor from low-resolution images acquired by a ground robot.
We show that learning to predict sound features as an auxiliary pretext task yields large performance improvements on the visual localization task.
Extensive experiments show that the approach significantly outperforms a supervised baseline, reducing the Mean Absolute Error from 14 to 7 cm, whereas a model that has access to labels for the entire training set yields an error of 5 cm.

<!---
### Bibtex

```properties
@article{nava2021uncertainty,
  author={M. {Nava} and A. {Paolillo} and J. {Guzzi} and L. M. {Gambardella} and A. {Giusti}},
  journal={IEEE Robotics and Automation Letters}, 
  title={Uncertainty-Aware Self-Supervised Learning of Spatial Perception Tasks}, 
  year={2021},
  volume={6},
  number={4},
  pages={6693-6700},
  doi={10.1109/LRA.2021.3095269}
}
```
-->

### Video
*Soon to be available.*
<!--[![Learning Visual Object Localization from Few Labeled Examples using Sound Prediction as a Pretext Task](https://github.com/idsia-robotics/Sound-as-Pretext/blob/main/video/video.gif)](https://youtu.be/G3cIDRrkfZY)-->
