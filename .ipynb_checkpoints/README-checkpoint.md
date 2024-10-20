# **Evaluation of Metrics and Models for Automated Essay Scoring and Feedback Generation**

This repository contains code and resources for evaluating various models and metrics for automated essay scoring (AES) and feedback generation. The project focuses on fine-tuning language models on the PeerRead dataset to generate qualitative feedback similar to human reviewers.

## **Table of Contents**

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## **Project Overview**

Automated Essay Scoring systems have advanced significantly but often lack the ability to provide detailed, qualitative feedback. This project aims to:

- Evaluate different NLP models for feedback generation.
- Analyze the effectiveness of various evaluation metrics.
- Compare model-generated feedback with human judgments to assess alignment and reliability.

## **Dataset**

We use the [PeerRead](https://github.com/allenai/PeerRead) dataset, which contains academic papers and their associated reviews.

## **Prerequisites**

- **Python 3.8 or higher**

## **Installation**

1. **Clone the Repository**

   ```bash
   git clone https://github.com/maryambrj/CSE-842_NLP_project.git
   cd CSE-842_NLP_project
   ```

2. **Create and Activate a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Packages**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## **Usage**

### **Data Preparation**

1. **Download and Prepare the PeerRead Dataset**

   - Clone the PeerRead repository:

     ```bash
     git clone https://github.com/allenai/PeerRead.git
     ```

   - Follow the instructions in the PeerRead repository to set up the data directories.

2. **Run the Data Preparation Script**

   - Modify the `prepare_data.py` script if necessary to point to your data directories.

   - Execute the script:

     ```bash
     python prepare_data.py
     ```

   - This script performs the following:
     - Extracts input-output pairs from the dataset.
     - Preprocesses the text data.
     - Tokenizes the data and saves it for model training.

### **Model Training**

1. **Run the Training Script**

   - Execute the `train.py` script to fine-tune the model:

     ```bash
     python train.py
     ```

   - The script will:
     - Load the prepared datasets.
     - Initialize the selected NLP model (e.g., BART).
     - Fine-tune the model on the training data.
     - Save the fine-tuned model for inference or evaluation.

2. **Adjust Hyperparameters**

   - You can adjust hyperparameters like batch size, learning rate, and number of epochs in the `train.py` script.

## **Project Structure**

```
CSE-842_NLP_project/
├── data/                   # Directory for datasets (not included in the repository)
├── prepare_data.py         # Script for data extraction and preprocessing
├── train.py          # Script for model training
├── requirements.txt        # List of required packages
├── README.md               # Project documentation
└── .gitignore              # Git ignore file
```

**Note:** The `data/` directory is excluded from version control to prevent large files from being uploaded.

## **Contributing**

Contributions are welcome! 

to add sections for:

- **Model Evaluation:** Instructions on how to evaluate the model's performance.
- **Results** 
- **References:** any papers or resources 
