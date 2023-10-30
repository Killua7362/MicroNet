FROM python:3.9-slim-buster
WORKDIR /app
COPY . .
RUN apt-get -y update && apt-get install -y gcc
RUN pip install -r requirements.txt
EXPOSE 8000
ENV FLASK_APP=models/gpt2/server.py
#CMD [ "python3", "-m" , "flask", "run","--host=0.0.0.0","--port=8000"]
#CMD ["python","models/gpt2/inference.py"]
CMD ["python","models/gpt2/server.py"]