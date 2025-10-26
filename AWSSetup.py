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
        self.s3_client = boto3.client('s3',region_name = self.region)
        self.s3_resource = boto3.resource('s3',region_name = self.region)
        self.iam_client = boto3.client('iam',region_name = self.region)
        self.sagemaker_client = boto3.client('sagemaker',region_name = self.region)

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

    def verify_credentials(self):
        try:
            self.sts = boto3.client("STS", region_name = self.region)
            self.identity = self.sts.get_caller_indentity()

            print("AWS Credentials Verified")
            print(f"Account: {self.identity["Account"]}")
            print(f"User ARN: {self.identity["Arn"]}")
            return True
        
        except Exception as e:
            print(f"Credentials failed: {e}")
            return False

    def create_sagemaker_role(self, role_name = "SageMalerExecutionRole"):
        try:
            self.iam_client.get_role(RoleName=role_name)
            print(f"Role {role_name} already exists")
            role_arn = self.iam_client.get_role(RoleName=role_name)['Role']['Arn']
            self.role = role_arn
            return role_arn

        except:
            print(f"Creating a new SageMaker executioner role: {role_name}")

        trust_policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }

        response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='SageMaker execution role for ML training'
            )

        policies = [
            'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
            'arn:aws:iam::aws:policy/AmazonS3FullAccess'
        ]
        
        for policy in policies:
            self.iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn=policy
            )
        
        role_arn = response['Role']['Arn']
        self.role = role_arn
        print(f"Created role: {role_arn}")
        print("IAM propagation...")
        
        return role_arn


            
