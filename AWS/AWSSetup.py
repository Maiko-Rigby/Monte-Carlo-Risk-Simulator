import boto3
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
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

class S3DataManager:
    def __init__(self, bucket_name, region = "eu-north-1"):
        self.bucket_name = bucket_name
        self.region = region

        self.s3_client = boto3.client('s3',region_name = self.region)
        self.s3_resource = boto3.resource('s3',region_name = self.region) 
        
    def create_bucket(self):
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"Bucket {self.bucket_name} already exists")
        except:
            try:
                if self.region == "eu-north-1":
                    self.s3_client.create.bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.region}
                        )
                print(f"Created bucket: {self.bucket_name}")
            except Exception as e:
                print(f"Error creating bucket: {e}")
                raise

        self.s3_client.put_bucket_versioning(
            Bucket=self.bucket_name,
            VersioningConfiguration={'Status': 'Enabled'}
        )
        
        return f"s3://{self.bucket_name}"
    
    def upload_dataframe(self, df, s3_key, file_format = 'parquet'):
        print(f"Uploading {len(df)} records to s3 ://{self.bucket_name}/{s3_key}")

        buffer = io.BytesIO()

        if file_format == "parquet":
            df.to_parquet(buffer, engine = "parquet", compression = "snappy", index = False)
        elif file_format == "csv":
            df.to_csv(buffer, index = False)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        buffer.seek(0)

        self.s3_client.put_object(
            Bucket = self.bucket_name,
            Key = s3_key,
            Body = buffer.getvalue()
        )

        file_size_mb = len(buffer.getvalue()) / (1024 * 1024)
        print(f"Uploaded {file_size_mb}MB")

        return f"s3://{self.bucket_name}/{s3_key}"
    
    def upload_file(self, local_path, s3_key):
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"Uploading {local_path} ({file_size_mb}MB) to S3...")

        self.s3_client.upload_file(local_path, self.bucket_name, s3_key)

        print(f"Uploaded to s3://{self.bucket_name}/{s3_key}")
        return f"s3://{self.bucket_name}/{s3_key}"
    
    def upload_directory(self, local_dir, s3_prefix):
        local_path = Path(local_dir)
        files_uploaded = 0

        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                s3_key = f"{s3_prefix}/{relative_path}"
            
            for file_path in local_path.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_path)
                    s3_key = f"{s3_prefix}/{relative_path}"
                    
                    self.s3_client.upload_file(
                        str(file_path),
                        self.bucket_name,
                        s3_key
                    )
                    files_uploaded += 1
        
        print(f"Uploaded {files_uploaded} files from {local_dir}")
        return f"s3://{self.bucket_name}/{s3_prefix}"

    def list_objects(self, prefix = ''):
        response = self.s3_client.list_objects_v2(
            Bucket = self.bucket_name,
            Prefix = prefix
        )

        if 'Contents' not in response:
            print(f"No objects found with prefix: {prefix}")
            return []
        
        objects = []
        for obj in response['Contents']:
            size_mb = obj['Size'] / (1024 * 1024)
            objects.append({
                'Key': obj['Key'],
                'Size_MB': f"{size_mb:.2f}",
                'LastModified': obj['LastModified']
            })
        
        df = pd.DataFrame(objects)
        print(f"\nObjects in s3://{self.bucket_name}/{prefix}:")
        print(df.to_string(index=False))
        
        return objects
    
    def download_file(self, s3_key, local_path):
        print(f"Downloading s3://{self.bucket_name}/{s3_key}....")
 
        self.s3_client.download_file(
            self.bucket_name,
            s3_key,
            local_path
        )

        print(f"Downloaded to {local_path}")
        return local_path
    

class SageMakerDataPrep:

    def __init__(self, s3_manager):
        self.s3_manager = s3_manager

    def prepare_training_data(self, csv_path, test_size = 0.2):
        print(f"Preparing data for sagemaker")

        df = pd.read_csv(csv_path)
        print(f"loaded {len(df)} records from {csv_path}")

        print(f"egineering feature...")
        feature_cols = ['daily_return', 'volatility', 'max_drawdown','portfolio_value', 'total_return']
        feature_cols = [col for col in feature_cols if col in df.columns]

        target_col = 'sharpe_ratio'

        ml_df = df[feature_cols] + [target_col].dropna()

        train_df, temp_df = train_test_split(ml_df, test_size=test_size*2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        print(f"\nData splits:")
        print(f"  Train: {len(train_df)} ({len(train_df)/len(ml_df)*100:.1f}%)")
        print(f"  Validation: {len(val_df)} ({len(val_df)/len(ml_df)*100:.1f}%)")
        print(f"  Test: {len(test_df)} ({len(test_df)/len(ml_df)*100:.1f}%)")

        s3_paths = {}
        
        for name, data in [('train', train_df), ('validation', val_df), ('test', test_df)]:
            s3_key = f"sagemaker/data/{name}/data.csv"
            s3_paths[name] = self.s3_manager.upload_dataframe(
                data, s3_key, file_format='csv'
            )
        
        print(f"\nData prepared and uploaded to S3")
        
        return s3_paths, feature_cols, target_col
    

