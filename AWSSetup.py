import boto3
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
import io
import tarfile

class AWSSetup:
    def __init__(self, region = "eu-north-1"):
        
        self.region = region
        self.s3_client = boto3.client('s3',region_name = region)
        self.s3_resource = boto3.resource('s3',region_name = region)
        self.iam_client = boto3.client('iam',region_name = region)
        self.sagemaker_client = boto3.client('sagemaker',region_name = region)

        try:
            self.sagemaker_session =  sagemaker.Session()
            self.role = get_execution_role()
        except:
            self.sagemaker_session = sagemaker.Session()
            self.role = None
            print(f'Not in SageMaker environment')

        print("AWS Setup Complete")
        print(f"Region: {self.region}")
        print(f"Role: {self.role}")

