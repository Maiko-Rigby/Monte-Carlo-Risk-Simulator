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
    
class SageMakerTrainer:

    def __init__(self, role, bucket_name, region = "eu-north-1"):
        self.role = role
        self.bucket_name = bucket_name
        self.region = region

    def create_training_script(self, output_dir = "sagemaker_code"):
        os.makedirs(output_dir,exist_ok= True)

        training_script = '''
    import argparse
    import os
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
    import json

    def train(args):

        # Load data from SageMaker paths
        train_df = pd.read_csv(os.path.join(args.train, 'data.csv'))
        val_df = pd.read_csv(os.path.join(args.validation, 'data.csv'))
        
        print(f"Training data: {train_df.shape}")
        print(f"Validation data: {val_df.shape}")
        
        # Split features and target
        target_col = 'sharpe_ratio'
        feature_cols = [col for col in train_df.columns if col != target_col]
        
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_val = val_df[feature_cols].values
        y_val = val_df[target_col].values
        
        print(f"Features: {feature_cols}")
        
        # Train model with hyperparameters
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=42,
            n_jobs=-1
        )
        
        print("Training Random Forest model...")
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        print(f"\\nTraining Results:")
        print(f"  Train RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
        print(f"  Val RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
        
        # Save model
        model_path = os.path.join(args.model_dir, 'model.joblib')
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
        # Save metrics
        metrics = {
            'train_rmse': float(train_rmse),
            'val_rmse': float(val_rmse),
            'train_r2': float(train_r2),
            'val_r2': float(val_r2)
        }
        
        metrics_path = os.path.join(args.output_data_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        
        # Hyperparameters
        parser.add_argument('--n-estimators', type=int, default=100)
        parser.add_argument('--max-depth', type=int, default=10)
        parser.add_argument('--min-samples-split', type=int, default=5)
        
        # SageMaker environment paths
        parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
        parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
        parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
        parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
        
        args = parser.parse_args()
        train(args)
    '''
        script_path = os.path.join(output_dir, 'train.py')
        with open(script_path, 'w') as f:
            f.write(training_script)

            print(f"Training script created: {script_path}")
            return script_path
        
    def launch_training_script(self, s3_data_paths, script_path, instance_type='ml.m5.large', use_spot=True):
        print('Launching training script')

        sklearn_estimator = SKLearn(
            entry_point= script_path,
            role = self.role,
            instance_type = instance_type,
            instance_count=1,
            framework_version='1.2-1',
            py_version='py3',
            hyperparameters={
                'n-estimators': 100,
                'max-depth': 10,
                'min-samples-split': 5
            },
            output_path=f"s3://{self.bucket_name}/sagemaker/output",
            base_job_name='portfolio-risk-model',
            use_spot_instances=use_spot,
            max_run=3600,  # 1 hour max
            max_wait=7200 if use_spot else None,  # Wait up to 2 hours for spot
        )

        print(f"Instance type: {instance_type}")
        print(f"Spot instances: {use_spot}")
        print(f"Role: {self.role}")
        
        print("Starting training job...")
        print("This will take 5-10 minutes...")
        
        sklearn_estimator.fit({
            'train': s3_data_paths['train'],
            'validation': s3_data_paths['validation']
        })
        
        print(f"Training complete!")
        print(f"Model artifacts: {sklearn_estimator.model_data}")
        
        return sklearn_estimator
        
class SageMakerLSTMTrainer:

    def __init__(self, role, bucket_name):
        self.role = role
        self.bucket_name = bucket_name
        self.sagemaker_session = sagemaker.session()

    def create_pytorch_script(self, output_dir = 'sagemaker_code'):

        os.makedirs(output_dir, exist_ok= True)

        lstm_script = '''
    import argparse
    import os
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    import json

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :]).squeeze()

    def create_sequences(data, seq_length=10):
        """Create time series sequences"""
        sequences, targets = [], []
        
        for i in range(len(data) - seq_length):
            seq = data[i:i+seq_length, :-1]  # All columns except target
            target = data[i+seq_length, -1]   # Last column is target
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)

    def train(args):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load data
        train_df = pd.read_csv(os.path.join(args.train, 'data.csv'))
        val_df = pd.read_csv(os.path.join(args.validation, 'data.csv'))
        
        # Create sequences
        X_train, y_train = create_sequences(train_df.values, args.sequence_length)
        X_val, y_val = create_sequences(val_df.values, args.sequence_length)
        
        # Convert to tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        # Model
        model = LSTMModel(
            input_size=X_train.shape[2],
            hidden_size=args.hidden_size,
            num_layers=args.num_layers
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Training loop
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss/len(train_loader):.4f}")
        
        # Save model
        torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
        print("Model saved")

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--batch-size', type=int, default=64)
        parser.add_argument('--learning-rate', type=float, default=0.001)
        parser.add_argument('--hidden-size', type=int, default=64)
        parser.add_argument('--num-layers', type=int, default=2)
        parser.add_argument('--sequence-length', type=int, default=10)
        parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
        parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
        parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
        
        args = parser.parse_args()
        train(args)
    '''
        script_path = os.path.join(output_dir, 'train_lstm.py')
        with open(script_path, 'w') as f:
            f.write(lstm_script)

        print(f'PyTorch LSTM script created: {script_path}')
        return script_path
    

        
def complete_asw_workflow(csv_path = 'simulation_results.csv', bucket_name = 'my-portfolio-ml-bucket'):
    print('AWS PIPELINE - WORKFLOW')

    print('Setting up AWS')
    aws_setup = AWSSetup(region = 'eu-north-1')

    if not aws_setup.verify_credentials():
        return
    
    if aws_setup.role is None:
        aws_setup.create_sagemaker_role()

    print('Creating S3 Bucket')
    s3_manager = S3DataManager(bucket_name, region = 'eu-north-1')
    s3_manager.create_bucket()

    print('Preparing and Uploading Data')
    data_prep = SageMakerDataPrep(s3_manager= s3_manager)
    s3_paths, feature_cols, target_col = data_prep.prepare_training_data(csv_path)

    s3_manager.list_objects(prefix = 'sagemaker/data')

    print('Creating Training Script')
    trainer = SageMakerTrainer(
        role = aws_setup.role,
        bucket_name = bucket_name,
        region = 'eu-north-1'
    )
    script_path = trainer.create_training_script()

    proceed = input('Proceed with Training (YES/NO)?')

    if proceed.lower() is 'yes':
        estimator = trainer.launch_training_script(
            s3_data_paths= s3_paths,
            script_path= script_path,
            instance_type= 'ml.m5.large',
            use_spot= True
        )

        print('Training Complete')
        print(f'Model Location: {estimator.model_data}')

        model_s3_key = estimator.model_data.split(f'{bucket_name}/')[1]
        s3_manager.download_file(model_s3_key, 'model.tar.gz')

        print('WORKFLOW COMPLETE!')
