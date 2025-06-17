# Regional Housing Market Analysis: Predictive Modeling Across Global Markets

This project develops a cloud-based predictive system that analyzes housing markets across three distinct regions: the US Pacific Northwest, European urban centers (Madrid), and Australian housing markets. By implementing and comparing different machine learning techniques on consistently structured datasets, we examine how regional factors influence housing prices and which prediction approaches perform best across different global markets.

## Project Structure
- `data/`: Contains raw and processed datasets
- `models/`: Stores trained models and evaluation results
- `notebooks/`: Jupyter notebooks for exploration and visualization
- `src/`: Source code for data processing, feature engineering, and modeling
  - `data/`: Data loading and preprocessing scripts
  - `features/`: Feature engineering code
  - `models/`: Implementation of the three modeling approaches
  - `visualization/`: Code for creating visualizations
- `infrastructure/`: Scripts for AWS deployment

## Setup and Usage
1. Clone the repository
2. Install required packages: `pip install -r requirements.txt`
3. Download the datasets from Kaggle and place them in `data/raw/`
4. Run the data processing pipeline: `python src/main.py`
5. Deploy models to AWS: `python src/deploy_aws.py`

## Models
Three machine learning approaches are implemented and compared:
1. Traditional Regression: Multiple Linear Regression with Ridge/Lasso Regularization
2. Tree-Based Ensemble: Gradient Boosting (XGBoost/LightGBM) 
3. Deep Learning: Neural Networks with embedding layers for categorical features

## Datasets
- King County House Sales (US Market)
- Madrid Real Estate Market (European Market)
- Australian Housing Market Data (Australia)

## Cloud Implementation
Models are deployed to AWS using a serverless architecture:
- AWS Lambda for model serving
- API Gateway for creating REST endpoints
- S3 for storing datasets and models