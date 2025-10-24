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