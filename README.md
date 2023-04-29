
# Controlling ROS Based Robot Movements with Hand Gestures Using  Multi-Layer Perceptron Classifier

This project is focussed on creation of a ROS node using Multi Layer Perceptron(MLP) to identify specfic hand gestures to give movement commands for a robot compatible with ROS.

This can be usefull in situations like,
- when working alongside with collaborative robots.
- to give commands to a robot without using a physical controller.
- to be used as a backup communication medium in case of noisy communication channels and etc.

The project uses google's mediapipe hand landmark detection impementations to detect hands and 21 landmarks on the hand. This projects focused only on single hand commands.

## Quick Execution Instructions 
All the python python files can be executed from the terminal using following command.
```shell
  python filename.py
```

To execute the ROS node in ROS workspace
```shell
  rosrun hand_signal hand_signal.py
```
NOTE: make sure to give execution permission for the `hand_signal.py` file using the command `chmod`

- `HandDetectionLogging.py` is used to generate the dataset.
- `BuildAndTrainMLP.py` is used to train and dave the MLP classifier. The came content is also available in a Jupiter Notebook format.
- `quickTest.py` is used to quickly test if the trained model can identify the gestures correctly


## Detailed Instructions for execution

The repository contains 4 main component.
- `HandDetectionLogging.py` 
- `BuildAndTrainMLP.py`
- `quickTest.py`
- `hand_signal` folder


### `HandDetectionLogging.py`

When `HandDetectionLogging.py` is executed, it will show a preview of the webcam and it will overlay a stick skeleton of the hand on top of the preview. It uses the mediapipe library to detect hand and draw the lanmarks on it. This window can be used to capture hand landmark information to build a dataset.

Press and hold number 0 on the keyboard to capture the current hand gesture as the 1st gesture to be included in the data set. The current implementation supports upto 5 different gestures. The landmark details corresponding to the preview will be stored in the `keypoint.csv` file in `./dataset` folder.  Pressing `Esc` will terminate the program.

### `BuildAndTrainMLP.py`
This python file contains the implementation of the MLP. Ths same content is available in a Jupyter Notebook version aswell. This will access the `keypoint.csv` file in the dataset folder and train an MLP classifier on the training dataset. The final trained model will be save in the `model` folder as `finalized_model.sav`.

### `quickTest.py`
This python will access and load the saved MLP model from model folder. Then the hand landmarks will be calculated from the webcam frame and the list will be sent to the model to predict the hand gesture id. The predicted hand gesture will be displayed on the preview window.

### `hand_signal Folder`
This contains the trained mode as a ROS package which can be used in ROS workspace. The ROS Node will take the webcam input, predict the hand gesture using the trained model and send the relevent velocity commands to the `/cmd_vel` topic.


