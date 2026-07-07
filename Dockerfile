FROM public.ecr.aws/lambda/python:3.11

# Copy requirements
COPY requirements-predict.txt ${LAMBDA_TASK_ROOT}

# Install dependencies (only what's needed for prediction)
# pandas, scikit-learn, joblib, and their dependencies are required.
# We upgrade pip first so that it correctly resolves newer wheel tags.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-predict.txt

# Copy source code and model files
COPY src/predict.py ${LAMBDA_TASK_ROOT}
COPY models/model.joblib ${LAMBDA_TASK_ROOT}/models/

# Set the Lambda handler
CMD ["predict.handler"]
