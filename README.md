This is the README file for describing the Capstone project "Customer Segmentation".

I: A Proposal File and Proposal Review Link:

A capstone proposal PDF file is included.

The review link is: https://review.udacity.com/#!/reviews/2554658

---------------------------------------------------------------------------------------------------------------------------------------------------------

II: A Project Report: 

A prokect report PDF file including the five major project development stages:

1) Define the problem to solve and investigate potential solutions and performance matrics.
2) Analyze the problem through visualizations and data exploration to have a better understanding of what algorithms and features are appropriate for solving it. 
3) Implement the algorithms and metrics of choice, documenting the preprocessing, refinement, and postprocessing stesp along the way. 
4) Collect results about the performance of the models uses, visualize significant quantiies, and validate/justify these values. 
5) Construct conclusions about the results and discuss whether the above implementation adequately solves the problem. 

---------------------------------------------------------------------------------------------------------------------------------------------------------

II: Python Code Files: 

Three Python development code files are included in the project and should be followed one by one with order: 
1) Data_Analysis_Cleaning.ipynb               --- explornary data analysis, data cleaning, and feature engineering
2) unsupervisedML_PCA_K-Means.ipynb           --- dimensionality reduction using PCA and clustering using K-Means as unsupervised machine learning 
3) supervisedML_XGBoost.ipynb                 --- customer response prediction using trained XGBoost model as supervised machine learning

---------------------------------------------------------------------------------------------------------------------------------------------------------

III: Libaries

All the code files were hosted in AWS notebooks with the kernal: conda_mxnet_p36.

The general libraries used for this project are listed as below: 

"""
import os
import io
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import json

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
"""

The machine learning libraries associated SageMaker and S3 are listed as below:

"""
import boto3
import sagemaker
from sagemaker import get_execution_role
import sagemaker.amazon.common as smac
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.transformer import Transformer
from sagemaker.predictor import csv_serializer, json_deserializer
from sagemaker import KMeans
import mxnet as mx
"""

The instance for running SageMake Notebook is "ml.t2.2xlarge" and the kernal used is "conda_mxnet_p36". 
---------------------------------------------------------------------------------------------------------------------------------------------------------

IV: Data Files: 

The data files were zipped and are included in the prject. Once unzipped, the raw data files provided were all placed inside the "Input" folder. All the treated, cleaned, scaled, and results files were included in other folders accordingly as you walk though Python development code files. 





