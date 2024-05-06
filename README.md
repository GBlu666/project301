source ./setup.sh
# Calligraphy Classifier

This project implements a Convolutional Neural Network (CNN) for classifying calligraphy images. The model is trained on a dataset of calligraphy images and can predict the class of a given input image.

## Dataset

The dataset used for training the calligraphy classifier be organized in the following directory structure:
data:
    caoshu:
        img1.jpg
        img2.jpg
        ...
    kaishu:
        img1.jpg
        img2.jpg
        ...
    lishu:
        img1.jpg
        img2.jpg
        ...
    zhuanshu:
        img1.jpg
        img2.jpg
        ...
Where caoshu, kaishu, lishu, zhuanshu represent a type of Chinese calligrapgy style and each class should have its own subdirectory containing the corresponding calligraphy images.

The dataset should be split into training and testing sets, with separate directories for each set:
    train_dir = "/Users/garybluedemac/Desktop/advance_topic/project/project301/project301/data"
    test_dir = "/Users/garybluedemac/Desktop/advance_topic/project/project301/project301/data_test"

## Requirements

To run the code, you need to have the following dependencies installed:

- Python
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-learn
- tqdm

You can install the required packages using the following command:
pip install ...

## Usage

1. Prepare your dataset by organizing it in the directory structure mentioned above.

2. Update the `load_dir` variable in the `main` function of `main.py` to point to the directory containing your dataset.

3. Run the `main.py` script to train the calligraphy classifier: python path_to/main.py

The script will load the dataset, train the CNN model, and evaluate its performance.

4. After training, the script will display the training loss and accuracy plots, as well as the confusion matrix and classification report.

## Model Architecture

The calligraphy classifier uses a Convolutional Neural Network (CNN) architecture defined in the `CalligraphyCNN` class in `src/model.py`. The architecture consists of convolutional layers, max pooling layers, and fully connected layers.

## Data Augmentation

Data augmentation techniques are applied to the training dataset to improve the model's generalization ability. The following augmentations are used:

- Conversion to grayscale
- Random rotation (up to 10 degrees)
- Random horizontal flip
- Normalization

## Evaluation

The model's performance is evaluated using accuracy as the metric. The training and validation losses and accuracies are plotted over the epochs to visualize the model's learning progress.

Additionally, a confusion matrix is generated to provide insights into the model's performance for each class.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- The dataset used in this project is sourced from [website cropping: https://www.cidianwang.com/shufazuopin/xiandai/269297_10.htm].
- The CNN architecture is inspired by [Huang, Q.; Li, M.; Agustin, D.; Li, L.; Jha, M. A Novel CNN Model for Classification of Chinese Historical Calligraphy Styles in Regular Script Font. Sensors 2024, 24, 197. https://doi.org/10.3390/s24010197
Chen, Yu-Sheng.; Su, Guangjun.; Li, Haihong. Machine Learning for Calligraphy Styles Recognition.].
