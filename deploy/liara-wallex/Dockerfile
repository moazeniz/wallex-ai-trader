FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 TZ=UTC PORT=8080
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
# Unprivileged runtime user. State, kill file, job queue and MCP call log live on the /data disk (Liara mounts it
# root-owned, so the entrypoint chowns it once as root, then drops to uid 10001 with setpriv; python is PID 1).
ENV WALLEX_STATE_DIR=/data/wallex-arena WALLEX_RUN_DIR=/tmp/wallex
RUN useradd --uid 10001 --create-home app && mkdir -p /data/wallex-arena && chown -R app:app /data && command -v setpriv
COPY . /app
# Code and the bundled results snapshot are world-readable (snapshot files can arrive 0600 from the source host).
RUN chmod -R a+rX /app
# The test suite runs at build time (fake transports, no network): first as root, then the dashboard nav pages as
# the runtime user, so a file the app cannot read fails the build instead of a page in production.
RUN python -m pytest -q -p no:cacheprovider checks \
    && setpriv --reuid=10001 --regid=10001 --init-groups python -m pytest -q -p no:cacheprovider checks/test_liara_image.py -k nav \
    && python -m compileall -q experiments liara && chmod -R a+rX /app && rm -rf /app/.pytest_cache
EXPOSE 8080
CMD ["sh", "-c", "chown -R app:app /data 2>/dev/null; exec setpriv --reuid=10001 --regid=10001 --init-groups python /app/liara/start.py"]
