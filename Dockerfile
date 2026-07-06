FROM public.ecr.aws/lambda/python:3.11

# Copy requirements
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install dependencies (only what's needed for prediction)
# pandas, scikit-learn, joblib, and their dependencies are required.
# We run pip install --no-cache-dir to keep the image size minimal.
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model files
COPY src/predict.py ${LAMBDA_TASK_ROOT}
COPY models/model.joblib ${LAMBDA_TASK_ROOT}/models/

# Set the Lambda handler
CMD ["predict.handler"]
