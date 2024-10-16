# AWS SageMaker Autopilot

This repository contains a Python script and workflow that demonstrates how to utilize **AWS SageMaker Autopilot** for automating machine learning model creation. SageMaker Autopilot automatically explores data, selects the best algorithms, and builds the most accurate model based on the input data, saving significant time and effort.

## Project Overview

The purpose of this project is to automate the process of training and tuning machine learning models using AWS SageMaker Autopilot. The workflow includes setting up an AutoML job, monitoring its progress, and deploying the best-performing model. 

Key features of the project:
- Automated model training and tuning.
- Easy deployment of the best model.
- Real-time model performance tracking using Streamlit.

## Requirements

To run this project, you need the following dependencies installed:

- Python 3.8+
- Boto3 (AWS SDK for Python)
- AWS SageMaker Python SDK
- Streamlit (for monitoring and visualization)

You can install the required dependencies with the following command:

```
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

```
git clone https://github.com/Jencheng1/AWSSagemakerAutopilot.git
cd AWSSagemakerAutopilot
```

2. Run the Python script:

```
python sagemaker.streamlit.autopilot.automl.py
```

This script will:
- Set up an AWS SageMaker Autopilot job.
- Track the job’s progress in real-time with a Streamlit dashboard.
- Deploy the best model once the job is complete.

## Data

The data used for training should be provided in CSV format and uploaded to an S3 bucket. The script automatically retrieves the data from S3, prepares it, and initiates the AutoML process using SageMaker Autopilot.

## Credit

This project utilizes the **AWS SageMaker Autopilot** to automate machine learning workflows and Streamlit for interactive monitoring. AWS SageMaker Autopilot provides a comprehensive solution for AutoML, allowing users to quickly build and deploy models without deep machine learning expertise.

## License

This project is licensed under the MIT License - see the LICENSE file at the following link for details: https://www.mit.edu/~amini/LICENSE.md.

## Screenshots

![image](https://github.com/user-attachments/assets/c1a2b199-74b5-402c-a75a-d3b31e318293)

![image](https://github.com/user-attachments/assets/32f8f2a4-786c-44fe-9d0b-39f559c6f3ba)
![image](https://github.com/user-attachments/assets/1ed740da-78eb-46b4-814b-4a46a046d610)


![image](https://github.com/user-attachments/assets/ca70ed13-436a-44dd-baa1-da1575459f8b)


![image](https://github.com/user-attachments/assets/ab8325ed-0a23-4de4-8ed5-a2256b133c4e)

![image](https://github.com/user-attachments/assets/cb67e8bf-d170-419a-bc68-13efd30439ae)




