FROM opencfd/openfoam-dev:2206

SHELL ["/bin/bash", "-lc"]

ARG DEBIAN_FRONTEND=noninteractive
ARG FREEMHD_REPO=https://github.com/PlasmaControl/FreeMHD.git
ARG FREEMHD_REF=14b54a3e8e1a05b6ee4c98331995abaaae96e7a5

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git rsync \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN git clone --depth 1 "${FREEMHD_REPO}" FreeMHD-src \
    && cd /opt/FreeMHD-src \
    && git fetch --depth 1 origin "${FREEMHD_REF}" \
    && git checkout "${FREEMHD_REF}"

RUN source /usr/lib/openfoam/openfoam2206/etc/bashrc \
    && cd /opt/FreeMHD-src/MHD_Solvers/libs/smoothedInterfaceProperties \
    && ./Allwmake \
    && cd /opt/FreeMHD-src/MHD_Solvers/solvers/epotMultiRegionInterFoam \
    && wmake

COPY docker/freemhd-openfoam/run-freemhd-case.sh /usr/local/bin/run-freemhd-case

RUN chmod +x /usr/local/bin/run-freemhd-case \
    && mkdir -p /workspace/case /workspace/output /work \
    && chmod 1777 /workspace /workspace/case /workspace/output /work

WORKDIR /workspace

ENTRYPOINT ["/openfoam/run"]
CMD ["bash"]
