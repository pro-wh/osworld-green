FROM ghcr.io/astral-sh/uv:python3.13-bookworm

RUN \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y \
        qemu-system-x86

RUN adduser agent
USER agent
WORKDIR /home/agent

# Download VM image.
# See osworld/desktop_env/providers/docker/provider.py
RUN \
    --mount=type=cache,target=/tmp/osworld-cache,uid=1000 \
    --mount=type=bind,source=Makefile,target=/home/agent/Makefile \
    make docker_vm_data/Ubuntu.qcow2

COPY pyproject.toml uv.lock README.md ./

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked --extra osworld

COPY osworld osworld

COPY src src

ENV PROXY_CONFIG_FILE="/home/agent/osworld/evaluation_examples/settings/proxy/dataimpulse.json"

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009