from pathlib import Path

import pytest

from scripts.extract_freemhd_coupled_log import extract_records, parse_lmx_diag_line
from scripts.patch_freemhd_coupled_logging import (
    patch_epot_file,
    patch_freemhd_tree,
    patch_main_file,
    patch_peqn_file,
    patch_ueqn_file,
)


pytestmark = pytest.mark.unit


def test_patch_main_file_injects_control_flag_and_outer_log():
    source = '   #include "createMhdFields.H"\n            const bool finalIter = (oCorr == nOuterCorr-1);\n'
    patched = patch_main_file(source)

    assert "logCoupledMhdIterations" in patched
    assert 'Info<< "LMX_DIAG outer"' in patched


def test_patch_epot_file_wraps_solve_and_adds_diag_log():
    source = "\tPotEEqn.solve();\n\t\tJxB = (J ^ B);\n\t}\n"
    patched = patch_epot_file(source)

    assert "auto potEPerf = PotEEqn.solve();" in patched
    assert 'Info<< "LMX_DIAG epot"' in patched


def test_patch_ueqn_file_wraps_momentum_solve_and_adds_diag_log():
    source = """        solve
        (
            UEqn
         ==
            fvc::reconstruct
            (
                (
                    mixture.surfaceTensionForce()
                  - ghf*fvc::snGrad(rho)
                  - fvc::snGrad(p_rgh)
                ) * mesh.magSf()
            )
        );
        K = 0.5*magSqr(U);
"""
    patched = patch_ueqn_file(source)

    assert "auto UPerf = solve" in patched
    assert 'Info<< "LMX_DIAG momentum"' in patched


def test_patch_peqn_file_wraps_pressure_solve_and_adds_diag_log():
    source = """        solve
        (
            p_rghEqnComp1() + p_rghEqnComp2() + p_rghEqnIncomp,
            mesh.solver
            ( 
              p_rgh.select
              (
                (                
                  oCorr == nOuterCorr-1
                  && corr == nCorr-1
                  && nonOrth == nNonOrthCorr
                )
              )
            )
        );
            fvOptions.correct(U);
"""
    patched = patch_peqn_file(source)

    assert "auto pPerf = solve" in patched
    assert 'Info<< "LMX_DIAG pressure"' in patched


def test_patch_freemhd_tree_updates_expected_files(tmp_path: Path):
    root = tmp_path
    main_path = root / "MHD_Solvers/solvers/epotMultiRegionInterFoam/epotMultiRegionInterFoam.C"
    epot_path = root / "MHD_Solvers/solvers/epotMultiRegionInterFoam/fluid/ePotEqn.H"
    ueqn_path = root / "MHD_Solvers/solvers/epotMultiRegionInterFoam/fluid/mhdUEqn.H"
    peqn_path = root / "MHD_Solvers/solvers/common/interFoam/fluid/pEqn.H"
    for path, content in (
        (main_path, '   #include "createMhdFields.H"\n            const bool finalIter = (oCorr == nOuterCorr-1);\n'),
        (epot_path, "\tPotEEqn.solve();\n\t\tJxB = (J ^ B);\n\t}\n"),
        (
            ueqn_path,
            """        solve
        (
            UEqn
         ==
            fvc::reconstruct
            (
                (
                    mixture.surfaceTensionForce()
                  - ghf*fvc::snGrad(rho)
                  - fvc::snGrad(p_rgh)
                ) * mesh.magSf()
            )
        );
        K = 0.5*magSqr(U);
""",
        ),
        (
            peqn_path,
            """        solve
        (
            p_rghEqnComp1() + p_rghEqnComp2() + p_rghEqnIncomp,
            mesh.solver
            ( 
              p_rgh.select
              (
                (                
                  oCorr == nOuterCorr-1
                  && corr == nCorr-1
                  && nonOrth == nNonOrthCorr
                )
              )
            )
        );
            fvOptions.correct(U);
""",
        ),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    updated = patch_freemhd_tree(root)

    assert main_path in updated
    assert epot_path in updated
    assert ueqn_path in updated
    assert peqn_path in updated


def test_parse_lmx_diag_line_extracts_numeric_fields():
    line = "LMX_DIAG epot time=0.0001 region=liquid oCorr=0 potEInitialResidual=0.3 potEFinalResidual=1e-4 potEIterations=7"
    parsed = parse_lmx_diag_line(line)

    assert parsed == {
        "kind": "epot",
        "time": 0.0001,
        "region": "liquid",
        "oCorr": 0,
        "potEInitialResidual": 0.3,
        "potEFinalResidual": 1e-4,
        "potEIterations": 7,
    }


def test_extract_records_reads_only_lmx_diag_lines(tmp_path: Path):
    log_path = tmp_path / "solver.log"
    log_path.write_text(
        "noise\n"
        "LMX_DIAG outer time=0.0001 oCorr=0 nOuterCorr=2 finalIter=false\n"
        "LMX_DIAG momentum time=0.0001 region=liquid oCorr=0 UInitialResidual=0.1 UFinalResidual=0.01 UIterations=3\n"
        "LMX_DIAG pressure time=0.0001 region=liquid oCorr=0 corr=0 nonOrth=1 pInitialResidual=0.2 pFinalResidual=0.02 pIterations=5\n"
    )

    records = extract_records(log_path)

    assert len(records) == 3
    assert records[0]["kind"] == "outer"
    assert records[1]["kind"] == "momentum"
    assert records[2]["kind"] == "pressure"
