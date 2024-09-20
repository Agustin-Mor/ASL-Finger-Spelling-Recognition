# ASL Finger Spelling Recognition

### Overview
This project focuses on recognizing American Sign Language (ASL) finger-spelling hand poses. It extracts and preprocesses images of ASL hand gestures and trains a machine learning model to recognize and translate these images into letters of the alphabet. With this project, users can create, train, and test their own models or use the pre-trained model provided, which achieves 95% testing accuracy.

---

### Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Technologies](#technologies)
- [Testing](#testing)
- [Credits](#credits)
- [License](#license)
- [Disclaimer](#disclaimer)

---

### Installation
To get started with this project, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/Agustin-Mor/ASL-Finger-Spelling-Recognition
   cd asl-finger-spelling-recognition
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
---

### Usage
This project consists of several notebooks and scripts for different stages of the ASL finger-spelling recognition process:

1. **Data Extraction**:  
   Run the `data-extraction.ipynb` notebook to extract the raw ASL hand gesture data. This step pulls data from images and converts them into a usable format for the model.
   
2. **Data Preprocessing**:  
   Once the data is extracted, preprocess it by running `data-preprocess.ipynb`. This step cleans, scales, and formats the data for model training.
   
3. **Model Training**:  
   Train the machine learning model by running the `generate-model.ipynb` notebook. This will create and train a model using the preprocessed data. If you don’t want to train your own model, a pre-trained model with 95% accuracy is provided.

4. **Capture New Images**:  
   If you want to create new test images, use the `capture_image.py` script. It captures new ASL hand images for testing the model.

5. **Testing the Model**:  
   To test the model on new images, run the `use-model.ipynb` notebook. You can load the pre-trained model or your own trained model and pass in new images to evaluate its performance.

---

### Features
- Full data extraction, preprocessing, and model training pipeline.
- The ability to capture and test custom ASL hand gestures.
- A pre-trained model with 95% accuracy, so you don’t need to start from scratch.

---

### Technologies
This project uses the following technologies and frameworks:
- **Python**: Primary programming language.
- **MediaPipe**: Used for extracting hand landmarks from images.
- **PyTorch**: Deep learning framework used for model creation and training.

---

### Testing
To test your model:
1. Use `capture_image.py` to capture new ASL images.
2. Run the `use-model.ipynb` notebook to evaluate the model on the captured images.

---

### Credits
The image dataset used for this project was provided by the "[ASL Sign Language Alphabet Pictures [Minus J, Z]](https://www.kaggle.com/datasets/signnteam/asl-sign-language-pictures-minus-j-z)" dataset on Kaggle. Special thanks to the dataset contributors.

---

### License
This project is open-source, and anyone is free to use, modify, or distribute the code. Please provide appropriate credit when using any aspect of the project.

---

### Disclaimer
The model is not trained to recognize the ASL letters 'J' and 'Z', as these letters require motion to express, which is beyond the scope of this image-based model.
