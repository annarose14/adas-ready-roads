FROM public.ecr.aws/lambda/python:3.12
COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt
COPY adas_pipeline.py handler.py ${LAMBDA_TASK_ROOT}/
COPY models/fastseg_large_512x1024.onnx ${LAMBDA_TASK_ROOT}/models/
CMD ["handler.handler"]
