import os
import argparse
import boto3
import json
from datetime import datetime

# Import project modules
from infrastructure.setup_aws import create_s3_bucket, create_lambda_function, create_api_gateway
from infrastructure.deploy_models import package_model_for_lambda, upload_model_to_s3

def setup_aws_infrastructure(project_name):
    """Set up AWS infrastructure"""
    print("Setting up AWS infrastructure...")
    
    # Create S3 bucket
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    bucket_name = f"{project_name.lower()}-{timestamp}"
    create_s3_bucket(bucket_name)
    
    # Create IAM role for Lambda
    iam = boto3.client('iam')
    
    # Check if role already exists
    role_name = f"{project_name}-LambdaRole"
    role_arn = None
    
    try:
        response = iam.get_role(RoleName=role_name)
        role_arn = response['Role']['Arn']
        print(f"Using existing role: {role_name}")
    except iam.exceptions.NoSuchEntityException:
        # Create new role
        assume_role_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy)
        )
        role_arn = response['Role']['Arn']
        print(f"Created role: {role_name}")
        
        # Attach policies
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
        )
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
        )
    
    return bucket_name, role_arn

def deploy_model(model_path, region_name, model_type, bucket_name, role_arn):
    """Deploy a single model to AWS Lambda"""
    print(f"Deploying {model_type} model for {region_name}...")
    
    # Package model
    model_zip = package_model_for_lambda(model_path, region_name, model_type)
    
    # Upload to S3
    s3_key = f"models/{region_name}/{model_type}/model.zip"
    upload_model_to_s3(model_zip, bucket_name, s3_key)
    
    # Create Lambda function
    function_name = f"housing-price-prediction-{region_name}-{model_type}"
    lambda_arn = create_lambda_function(
        function_name,
        role_arn,
        "lambda_function.lambda_handler",
        code_s3_bucket=bucket_name,
        code_s3_key=s3_key
    )
    
    # Create API Gateway
    api_name = f"housing-price-api-{region_name}-{model_type}"
    api_url = create_api_gateway(api_name, lambda_arn)
    
    return {
        'region': region_name,
        'model_type': model_type,
        'function_name': function_name,
        'api_url': api_url
    }

def main():
    parser = argparse.ArgumentParser(description='Deploy Housing Market Models to AWS')
    parser.add_argument('--models-dir', type=str, default='models', help='Models directory')
    parser.add_argument('--project-name', type=str, default='HousingMarketAnalysis', help='Project name')
    args = parser.parse_args()
    
    # Setup AWS infrastructure
    bucket_name, role_arn = setup_aws_infrastructure(args.project_name)
    
    # Find models to deploy
    deployed_endpoints = []
    for region in ['US_PacificNW', 'EU_Madrid', 'AU_Housing']:
        for model_type in ['linear', 'xgboost', 'nn']:
            # Find latest model file
            model_files = [f for f in os.listdir(args.models_dir) if f.startswith(f"{region}_{model_type}")]
            if not model_files:
                print(f"No {model_type} model found for {region}")
                continue
            
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join(args.models_dir, latest_model)
            
            # Deploy model
            endpoint_info = deploy_model(model_path, region, model_type, bucket_name, role_arn)
            deployed_endpoints.append(endpoint_info)
    
    # Save endpoint information
    endpoints_file = os.path.join(args.models_dir, 'deployed_endpoints.json')
    with open(endpoints_file, 'w') as f:
        json.dump(deployed_endpoints, f, indent=2)
    
    print(f"Deployed {len(deployed_endpoints)} models")
    print(f"Endpoint information saved to {endpoints_file}")

if __name__ == "__main__":
    main()