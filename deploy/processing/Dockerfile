# Processing worker — container Lambda (the one piece that CAN'T be a zip: torch +
# mediapipe + the model bundle exceed the 250 MB zip limit).
#
# Build from repo root; the model bundle (~470 MB, gitignored) is pulled from S3
# into ./deploy/processing/models by the buildspec BEFORE this build. See buildspec.yml.
#   docker build -f deploy/processing/Dockerfile -t motion-caddie/processing .
#
# STATUS: authored, not yet built (no Docker/CodeBuild locally). Expect to iterate
# on native libs (mediapipe/opencv) on the first real build.

FROM public.ecr.aws/lambda/python:3.12

# native libs mediapipe/opencv need at runtime
RUN dnf install -y mesa-libGL libglvnd-glx && dnf clean all

COPY deploy/processing/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV APP_ROOT=/var/task
# pipeline scripts + adapters + the modules they import (whole Scripts/ is simplest & correct)
COPY Scripts/                 ${LAMBDA_TASK_ROOT}/Scripts/
COPY deploy/processing/processing_handler.py ${LAMBDA_TASK_ROOT}/processing_handler.py
# KB for the coaching eval step
COPY Data/coaching/indicator_kb.json ${LAMBDA_TASK_ROOT}/Data/coaching/indicator_kb.json
# model bundle (staged from S3 by buildspec into deploy/processing/models/)
COPY deploy/processing/models/ ${LAMBDA_TASK_ROOT}/Models/

CMD [ "processing_handler.handler" ]
