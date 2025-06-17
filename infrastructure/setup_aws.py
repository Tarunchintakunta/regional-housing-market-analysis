import boto3
import json
import time

def create_s3_bucket(bucket_name, region='us-east-1'):
    """Create an S3 bucket for storing datasets and models"""
    s3 = boto3.client('s3', region_name=region)
    
    try:
        if region == 'us-east-1':
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        print(f"Created bucket: {bucket_name}")
        return True
    except Exception as e:
        print(f"Error creating bucket: {e}")
        return False

def create_lambda_function(function_name, role_arn, handler, runtime='python3.9',
                          code_s3_bucket=None, code_s3_key=None, zip_file=None):
    """Create a Lambda function for model serving"""
    lambda_client = boto3.client('lambda')
    
    # Prepare code parameter
    code = {}
    if zip_file:
        with open(zip_file, 'rb') as f:
            code['ZipFile'] = f.read()
    elif code_s3_bucket and code_s3_key:
        code['S3Bucket'] = code_s3_bucket
        code['S3Key'] = code_s3_key
    
    try:
        response = lambda_client.create_function(
            FunctionName=function_name,
            Runtime=runtime,
            Role=role_arn,
            Handler=handler,
            Code=code,
            Timeout=30,
            MemorySize=512
        )
        print(f"Created Lambda function: {function_name}")
        return response['FunctionArn']
    except Exception as e:
        print(f"Error creating Lambda function: {e}")
        return None

def create_api_gateway(api_name, lambda_arn):
    """Create API Gateway to expose Lambda function"""
    apigw = boto3.client('apigatewayv2')
    
    try:
        # Create API
        api = apigw.create_api(
            Name=api_name,
            ProtocolType='HTTP',
            Version='1.0'
        )
        api_id = api['ApiId']
        
        # Create route
        route = apigw.create_route(
            ApiId=api_id,
            RouteKey='POST /predict',
            Target=f'integrations/{integration_id}'
        )
        
        # Create stage
        stage = apigw.create_stage(
            ApiId=api_id,
            StageName='prod',
            AutoDeploy=True
        )
        
        print(f"Created API Gateway: {api_name}")
        return f"https://{api_id}.execute-api.{boto3.session.Session().region_name}.amazonaws.com/prod/predict"
    except Exception as e:
        print(f"Error creating API Gateway: {e}")
        return None