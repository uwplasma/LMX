FROM microfluidica/openfoam:2206

ENV DEBIAN_FRONTEND=noninteractive
ENV WM_PROJECT_DIR=/opt/OpenFOAM/OpenFOAM-v2206
ENV WM_THIRD_PARTY_DIR=/opt/OpenFOAM/ThirdParty-v2206
ENV FOAM_INST_DIR=/opt/OpenFOAM

RUN apt-get update && apt-get install -y \
    bash \
    ca-certificates \
    curl \
    git \
    python3 \
    rsync \
    wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN git clone --depth 1 https://github.com/PlasmaControl/FreeMHD.git /opt/FreeMHD

RUN /bin/bash -lc "source /opt/OpenFOAM/OpenFOAM-v2206/etc/bashrc && cd /opt/FreeMHD && \
    if [ -x ./Allwmake ]; then ./Allwmake; fi && \
    wmake MHD_Solvers/solvers/epotMultiRegionFoam && \
    wmake MHD_Solvers/solvers/epotMultiRegionInterFoam"

RUN mkdir -p /opt/lmx
COPY run_freemhd_case.sh /opt/lmx/run_freemhd_case.sh
RUN chmod +x /opt/lmx/run_freemhd_case.sh

WORKDIR /workspace
CMD ["/bin/bash"]
