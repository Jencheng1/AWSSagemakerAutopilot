from tkinter.messagebox import YES
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk
import numpy as np
import flask
import boto3
#from flask import Flask, render_template # for web app
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
#import pandas as pd
import plotly
import plotly.express as px
import json # for graph plotting in website
# NLTK VADER for sentiment analysis
#import nltk
#nltk.downloader.download('vader_lexicon')
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
#import json
import sys
import time
import requests

from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import plotly.figure_factory as ff
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from numpy import radians, cos, sin


def Get_Training_Data(s3_bucket,file_name):
    import boto3
    import numpy as np 
    import pandas as pd

#from package import config
    
    instance_type = 'ml.m5.large'

    session = boto3.Session()
    print(session)

    s3 = session.resource('s3', region_name='us-east-1')
    print(s3)
   
    object = s3.Object(s3_bucket,file_name)
    print(s3_bucket)
    print(file_name)
    print(object)
    download=object.download_file(file_name)
    print(download)


    data = pd.read_csv(file_name, delimiter=',')
    return data

with st.sidebar:
    st.header('Input training data and output S3 for Credit Card Fraud AI Model Training')
    with st.form(key='training_form'):
        s3_bucket=st.text_input(label='Input: Training data S3 bucket')
        file_name=st.text_input(label='Input: Training data file name')
        s3_bucket_for_model=st.text_input(label='Output: S3 bucket for top-performing-AI model for credit card payment pattern fraud detection  ')
        option = st.selectbox(
                "How would you like to train the credit card fraud pattern detection model?",
                ("AutoPiot/AutoML", "Manual: Supervised Learning", "Manual:UnSupervised Learning"))
        #file_name_for_model=st.text_input(label='Please Enter file name for Fraud AI Model')
        submit_button = st.form_submit_button(label='Submit')

if submit_button:


    import numpy as np 
    import pandas as pd

    data= Get_Training_Data(s3_bucket,file_name)
    st.title('Investigate and process the training data')
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image("ml_data.jfif")
    st.dataframe(data)

    nonfrauds, frauds = data.groupby('Class').size()
    st.write('Number of frauds: ', frauds)
    st.write('Number of non-frauds: ', nonfrauds)
    st.write('Percentage of fradulent data:', 100.*frauds/(frauds + nonfrauds))
    
    st.title('AutoPilot/AutoML Training')
    st.image("sagemaker.png")
    st.image("autopilot.png")
 
  

    s3 = boto3.client('s3') 

    obj = s3.get_object(Bucket=s3_bucket, Key=file_name) 
    # get object and file (key) from bucket

    df = pd.read_csv(obj['Body']) # 'Body' is a key word

    import numpy as np 
    import pandas as pd
    import boto3
    import sagemaker
    import os, sys
    import time

    
    bucket= s3_bucket                
    region = boto3.Session().region_name
    prefix = 'sagemaker/fraud-detection-auto-ml'
    # Role when working on a notebook instance
    role = sagemaker.get_execution_role()
    sess   = sagemaker.Session(default_bucket=bucket)
    print(sess.default_bucket())

    sm = boto3.Session().client(service_name='sagemaker',region_name=region)
    sm_rt = boto3.Session().client('runtime.sagemaker', region_name=region)
    
    from sklearn.model_selection import train_test_split

    train_data, test_data = train_test_split(df, test_size=0.2)
    
    # Save to CSV files and upload to S3
    train_file = "automl-train.csv"
    train_data.to_csv(train_file, index=False, header=True, sep=',') # Need to keep column names
    train_data_s3_path = sess.upload_data(path=train_file, key_prefix=prefix + "/train")
    print("Train data uploaded to: " + train_data_s3_path)
    
    test_file = "automl-test.csv"
    test_file_no_target = "automl-test-no-target.csv"
    test_data_no_target = test_data.drop(columns=["Class"])
    test_data.to_csv(test_file, index=False, header=False, sep=',')
    test_data_no_target.to_csv(test_file_no_target, index=False, header=False, sep=',')
    test_data_s3_path = sess.upload_data(path=test_file, key_prefix=prefix + "/test")
    test_data_s3_path_no_target = sess.upload_data(path=test_file_no_target, key_prefix=prefix + "/test")
    print("Test data uploaded to: " + test_data_s3_path)
    print("Test data no target uploaded to: " + test_data_s3_path_no_target)

    import numpy as np 
    import pandas as pd
    import boto3
    import sagemaker
    import os, sys
    import time

    
    #bucket = sess.default_bucket() 
    bucket = s3_bucket                  
    region = boto3.Session().region_name
    prefix = 'sagemaker/fraud-detection-auto-ml'
   
    role='arn:aws:iam::xxx'
    sess   = sagemaker.Session(default_bucket=bucket)
    print(sess.default_bucket())

    sm = boto3.Session().client(service_name='sagemaker',region_name=region)
    sm_rt = boto3.Session().client('runtime.sagemaker', region_name=region)

    print (sagemaker.__version__)
    
    input_data_config = [{
      'DataSource': {
        'S3DataSource': {
          'S3DataType': 'S3Prefix',
          'S3Uri': 's3://{}/{}/input'.format(bucket,prefix)
        }
      },
      'TargetAttributeName': 'Class'  # the column we want to predict
    }
]
    
    output_data_config = { 'S3OutputPath': 's3://{}/{}/output'.format(bucket,prefix) }
    
    # Optional parameters
    
    problem_type = 'BinaryClassification'
    
    job_objective = { 'MetricName': 'F1' }
    
    
    from time import gmtime, strftime, sleep
    timestamp_suffix = strftime('%d-%H-%M-%S', gmtime())
    
    auto_ml_job_name = 'fraud-detection-' + timestamp_suffix
    print('AutoMLJobName: ' + auto_ml_job_name)

    with st.spinner("Waiting:Training the Fraud AI model with AutoPilot/AutoML  ..."):
        st.subheader('Progress of starting training job for  the Fraud AI model with AutoPilot/AutoML...')
   
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_steps = 100  # Example total steps for the process

    for step in range(total_steps):
    # Simulate a step in the process
        time.sleep(0.1)  # Adjust time per step to your actual process
        # Calculate progress percentage
        percent_complete = int((step + 1) / total_steps * 100)
        # Update progress bar and status text
        progress_bar.progress(percent_complete)
        status_text.text(f'Progress: {percent_complete}%')
    
    st.success('Setting up  and launching the Amazon SageMaker AutoPilot/ AutoML job has been completed successfully' )
   
    
    st.title('Tracking Live Job Progress')
 
    sm.create_auto_ml_job(AutoMLJobName=auto_ml_job_name,
                          InputDataConfig=input_data_config,
                          OutputDataConfig=output_data_config,
                          AutoMLJobConfig={"CompletionCriteria": {"MaxCandidates": 20}},
                          AutoMLJobObjective=job_objective,
                          ProblemType=problem_type,
                          RoleArn=role)
   
    job_run_status = sm.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)['AutoMLJobStatus']

    
    st.write(job_run_status)
    
    while job_run_status not in ('Failed', 'Completed', 'Stopped'):
        describe_response = sm.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)
        job_run_status = describe_response['AutoMLJobStatus']
    
        st.write (describe_response['AutoMLJobStatus'] + " - " + describe_response['AutoMLJobSecondaryStatus'])
        sleep(60)


    st.title('Fetching the auto-generated notebooks')
    st.markdown('''
     - Once the 'AnalyzingData' step is complete, SageMaker AutoPilot generates two notebooks: 
        1. Data exploration
        2. Candidate definition.
    ''')
    job = sm.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)
    job_candidate_notebook = job['AutoMLJobArtifacts']['CandidateDefinitionNotebookLocation']
    job_data_notebook = job['AutoMLJobArtifacts']['DataExplorationNotebookLocation']
    
    st.write(job_candidate_notebook)
    st.write(job_data_notebook)

  
    
    st.title('Inspecting the SageMaker Autopilot job with Amazon SageMaker Experiments')
    st.markdown('''
     - Once the 'ModelTuning' step starts, we can use the SageMaker Experiments SDK to list and view all jobs. Data is stored in a pandas dataframe, which makes it easy to filter it, compare it to other experiments, etc.
    ''')
    
    from sagemaker.analytics import ExperimentAnalytics

    analytics = ExperimentAnalytics(
        sagemaker_session=sess, 
        experiment_name=auto_ml_job_name+'-aws-auto-ml-job'
    )
    
    df = analytics.dataframe()
    st.write(df)
    
    st.title('Listing all candidates explored by Amazon SageMaker AutoPilot')
    st.markdown('''
     - Once the 'ModelTuning' step is complete, we can list top candidates that were identified by SageMaker AutoPilot, and sort them by their final performance metric.
    ''')
    
    candidates = sm.list_candidates_for_auto_ml_job(AutoMLJobName=auto_ml_job_name, 
                                                SortBy='FinalObjectiveMetricValue')['Candidates']
    index = 1
    for candidate in candidates:
      st.write (str(index) + "  " 
             + candidate['CandidateName'] + "  " 
             + str(candidate['FinalAutoMLJobObjectiveMetric']['Value']))
      index += 1
      
    best_candidate = sm.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)['BestCandidate']
    best_candidate_name = best_candidate['CandidateName']
    
    st.write("Candidate name: " + best_candidate_name)
    
    timestamp_suffix = strftime("%d-%H-%M-%S", gmtime())
    model_name = best_candidate_name + timestamp_suffix + "-model"
    model_arn = sm.create_model(
        Containers=best_candidate["InferenceContainers"], ModelName=model_name, ExecutionRoleArn=role
    )
    
    st.title('Evaluate on Testset by Hosting an Endpoint')
    st.markdown('''
     - Let's now deploy the model as endpoint and then make predictions on the test set to see how well the model performs on the hold-out test set.
    ''')
    
    epc_name = best_candidate_name + timestamp_suffix + "-epc"
    ep_config = sm.create_endpoint_config(
        EndpointConfigName=epc_name,
        ProductionVariants=[
            {
                "InstanceType": "ml.m5.2xlarge",
                "InitialInstanceCount": 1,
                "ModelName": model_name,
                "VariantName": "main",
            }
        ],
    )
    
    ep_name = best_candidate_name + timestamp_suffix + "-ep"
    create_endpoint_response = sm.create_endpoint(EndpointName=ep_name, EndpointConfigName=epc_name)
    
    sm.get_waiter("endpoint_in_service").wait(EndpointName=ep_name)
    
    st.title('Evaluate')
    st.markdown('''
     - Evaluating the performance for predication of generated AI models for fraud detection by autopiolot/autoML
     ''')
    
    
    tp = tn = fp = fn = count = 0

    with open('automl-test.csv') as f:
        lines = f.readlines()
        for l in lines[1:]:   # Skip header
            l = l.split(',')  # Split CSV line into features
            label = l[-1]     # Store 0/1 label
            l = l[:-1]        # Remove label
            l = ','.join(l)   # Rebuild CSV line without label
                    
            response = sm_rt.invoke_endpoint(EndpointName=ep_name, ContentType='text/csv', Accept='text/csv', Body=l)
    
            response = response['Body'].read().decode("utf-8")
            #print ("label %s response %s" %(label,response))
    
            if '1' in label:
                # Sample is positive
                if '1' in response:
                    # True positive
                    tp=tp+1
                else:
                    # False negative
                    fn=fn+1
            else:
                # Sample is negative
                if '0' in response:
                    # True negative
                    tn=tn+1
                else:
                    # False positive
                    fp=fp+1
            count = count+1
            if (count % 100 == 0):   
                sys.stdout.write(str(count)+' ')
                
    st.write ("Done")
    
     # Confusion matrix
    st.write ("%d %d" % (tn, fp))
    st.write ("%d %d" % (fn, tp))
    
    accuracy  = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall    = tp/(tp+fn)
    f1        = (2*precision*recall)/(precision+recall)
    
    st.write ("Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f" % (accuracy, precision, recall, f1))
    
