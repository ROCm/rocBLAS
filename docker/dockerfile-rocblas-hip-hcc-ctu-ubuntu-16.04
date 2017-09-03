# Parameters related to building rocblas
ARG base_image

FROM ${base_image}
LABEL maintainer="kent.knox@amd"

# Copy the debian package of rocblas into the container from host
COPY *.deb /tmp/

# Install the debian package
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y curl \
  && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --allow-unauthenticated -y \
    /tmp/rocblas-*.deb \
  && rm -f /tmp/*.deb \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
