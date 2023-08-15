FROM python:3.7-buster
COPY . /app             
WORKDIR /app
RUN pip install -r requirements.txt &&\
    ls -la /app/uploads &&\
    chmod a+x /app/uploads
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

EXPOSE 7860
CMD ["python3","-m", "flask", "run","--host","0.0.0.0","--port","7860"]
