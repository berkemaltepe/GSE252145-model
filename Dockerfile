FROM python:3.11-slim-bullseye
WORKDIR /model

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py .

EXPOSE 8050

CMD [ "python", "./model.py" ]

