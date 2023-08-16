FROM python:3.7-buster
COPY . /app             
WORKDIR /app
RUN pip install -r requirements.txt &&\
    ls -la /app/uploads &&\
    chmod a+x /app/uploads
RUN wget https://download1479.mediafire.com/f6y8zk6irg7giPemwX-Mgj5MRqeSM57Z2VKAdOHvxNOogsqF8nWrlObXzBIJrzCDsPJwbtT10axnklTLiujC1oUbQKzG8wfFRlYLJpEKWEO_nUA4pPNN2c7qpnF4ClE7eWZzxXkeGFnc4srR6k42xXQUCSR9RY5g0jJeHphSst8/ftjzkfi4ln32iza/features2.pkl -P /app
RUN wget https://download1652.mediafire.com/9o4tbixit20gGRGrHcl9WgsOvhljO3drMZ-2jS2IFTft0Suabqdt5TXJMkyTjjgPpBMIwmwnePs64Gb4_7-WJg6L1bmXO69Kz6JhCiHEvQK6xDUl3qLP88J9ssmcH5T-lIpbUvvc7in-lLdLh2l23uxtYTdzKDS26XcCo8iBpHg/g7aph18cd07zye4/model_mobilenet.h5 -P /app
RUN wget https://download852.mediafire.com/0jmok3sb3mmgjCA9z8qrhabUvHbtRqHPi01IPdiXIoSkhtyt-wts1MUxM5BUPAyCzv5ln6y5LjGy1j21BJC93y9mO-_bdxOZvpz4Bb3FTtn_JmWXdRRH_kAx6l8fy4w9HNZuGukEQ88UbdUOBbLr5xLGIQ87wLIaEVyNFDl7WIc/myzzh47zfs2jjoc/tokenizer2.pkl -P /app


RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

EXPOSE 7860
CMD ["python3","-m", "flask", "run","--host","0.0.0.0","--port","7860"]
