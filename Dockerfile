FROM public.ecr.aws/lambda/python:3.11

# Install dependencies
COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt

# Copy the serving module
COPY src/serving/ ${LAMBDA_TASK_ROOT}/serving/
COPY src/features/ ${LAMBDA_TASK_ROOT}/features/

# Set Python path so imports work
ENV PYTHONPATH="${LAMBDA_TASK_ROOT}"

# Use your actual production handler
CMD ["serving.handler.lambda_handler"]