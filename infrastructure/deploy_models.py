import boto3
import os
import json
import zipfile
import io

def package_model_for_lambda(model_path, region_name, model_type):
    """Package model for Lambda deployment"""
    # Create a zip file containing the model and Lambda handler
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add model file
        zipf.write(model_path, os.path.basename(model_path))
        
        # Add Lambda handler
        handler_content = f"""
import json
import joblib
import numpy as np

# Load model at initialization time
model = joblib.load('{os.path.basename(model_path)}')

def lambda_handler(event, context):
    try:
        # Parse input
        body = json.loads(event['body'])
        features = body['features']
        
        # Make prediction
        prediction = float(model.predict([features])[0])
        
        return {{
            'statusCode': 200,
            'body': json.dumps({{
                'predicted_price': prediction,
                'region': '{region_name}',
                'model_type': '{model_type}'
            }})
        }}
    except Exception as e:
        return {{
            'statusCode': 400,
            'body': json.dumps({{
                'error': str(e)
            }})
        }}
        """
        zipf.writestr('lambda_function.py', handler_content)
    
    zip_buffer.seek(0)
    return zip_buffer.read()

def upload_model_to_s3(model_zip, bucket_name, s3_key):
    """Upload packaged model to S3"""
    s3 = boto3.client('s3')
    try:
        s3.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=model_zip
        )
        print(f"Uploaded model to s3://{bucket_name}/{s3_key}")
        return True
    except Exception as e:
        print(f"Error uploading model: {e}")
        return False