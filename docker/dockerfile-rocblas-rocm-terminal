# Parameters related to building rocblas
ARG base_image

FROM ${base_image}
LABEL maintainer="kent.knox@amd"

# Copy the debian package of rocblas into the container from host
COPY *.deb /tmp/

# Install the debian package
RUN sudo apt-get update && DEBIAN_FRONTEND=noninteractive sudo apt-get install --no-install-recommends -y curl \
  && sudo apt-get update && DEBIAN_FRONTEND=noninteractive sudo apt-get install --no-install-recommends --allow-unauthenticated -y \
    /tmp/rocblas-*.deb \
  && sudo rm -f /tmp/*.deb \
  && sudo apt-get clean \
  && sudo rm -rf /var/lib/apt/lists/*
