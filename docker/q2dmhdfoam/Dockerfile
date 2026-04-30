FROM foamscience/foam-extend-4.1-openmpi:latest

SHELL ["/bin/bash", "-lc"]

ARG Q2DMHDFOAM_REPO=https://github.com/iraola/Q2DmhdFoam.git
ARG Q2DMHDFOAM_REF=e13c6659e403d96a7b52b6cc65fc0260260b0b79

USER openfoam
WORKDIR /home/openfoam

RUN git init Q2DmhdFoam \
    && cd Q2DmhdFoam \
    && git remote add origin "${Q2DMHDFOAM_REPO}" \
    && git fetch --depth 1 origin "${Q2DMHDFOAM_REF}" \
    && git checkout FETCH_HEAD \
    && perl -0pi -e 's/mesh\.setFluxRequired\(p\.name\(\)\);/mesh.schemesDict().setFluxRequired(p.name());/g; s/fvc::interpolate\(rUA\)\*fvc::ddtCorr\(U, phi\)/fvc::ddtPhiCorr(rUA, U, phi)/g' Q2DmhdFoam/createFields.H Q2DmhdFoam/Q2DmhdFoam.C \
    && source /home/openfoam/foam/foam-extend-4.1/etc/bashrc \
    && cd Q2DmhdFoam \
    && wmake

COPY --chown=openfoam:openfoam run_reference_case.sh extract_reference_profile.py /home/openfoam/
RUN chmod +x /home/openfoam/run_reference_case.sh

ENTRYPOINT ["/home/openfoam/run_reference_case.sh"]
