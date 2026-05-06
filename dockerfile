FROM nvcr.io/nvidia/pytorch:25.04-py3

ARG PYPI_MIRROR=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL=
ARG DEBIAN_FRONTEND=noninteractive

ARG CLEAR_PIP_CONSTRAINTS=1
RUN set -eux; \
    if [ "${CLEAR_PIP_CONSTRAINTS}" = "1" ] && [ -f /etc/pip/constraint.txt ]; then \
        cp /etc/pip/constraint.txt /etc/pip/constraint.txt.bak; \
        : > /etc/pip/constraint.txt; \
    fi

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git jq vim less rsync wget curl ca-certificates openssh-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -i ${PYPI_MIRROR} \
    ${PIP_EXTRA_INDEX_URL:+--extra-index-url ${PIP_EXTRA_INDEX_URL}} \
    -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

COPY . /workspace
ENV PYTHONPATH=/workspace/ReMoScene:/workspace/ReMoScene/src


CMD ["/bin/bash"]
