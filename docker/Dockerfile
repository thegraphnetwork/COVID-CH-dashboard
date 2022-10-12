FROM condaforge/mambaforge

USER root 

ENV DEVIAN_FRONTEND=noninteractive

ENV HOME "/home/covidch"
ENV PATH "$PATH:/home/covidch/.local/bin"
ENV ENV_NAME base
ENV PATH "/opt/conda/envs/$ENV_NAME/bin:$PATH"
ENV PATH "/opt/poetry/bin:$PATH"

RUN apt-get -qq update --yes \
  && apt-get -qq install --yes --no-install-recommends \
  ca-certificates sudo curl \
  && rm -rf /var/lib/apt/lists/*


RUN useradd -ms /bin/bash covidch \
  && echo "covidch ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/covidch \
  && chmod 0440 /etc/sudoers.d/ \
  && echo 'source /opt/conda/bin/activate "$ENV_NAME" && exec "$@"' > /activate.sh \
  && echo 'source activate "$ENV_NAME"' >  /home/covidch/.bashrc \
  && chmod +x /activate.sh \
  && chmod -R a+rwx /opt/conda /tmp \
  && sudo chown -R covidch:covidch /usr/src


COPY --chown=covidch:covidch conda/dev.yaml /tmp/dev.yaml
COPY --chown=covidch:covidch pyproject.toml poetry.lock README.md /usr/src/
COPY --chown=covidch:covidch app /home/covidch/app/

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

RUN chmod -R a+rwx /home/covidch/.config/pypoetry/

USER covidch

RUN mamba env update -n $ENV_NAME \
    --file /tmp/dev.yaml \
  && cd /usr/src && poetry install \
  && mamba clean -afy

WORKDIR /home/covidch/app/

ENTRYPOINT ["streamlit", "run", "dashboard.py"]

#ENTRYPOINT ["bash", "/activate.sh"]
