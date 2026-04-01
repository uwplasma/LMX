from __future__ import annotations

import argparse
from pathlib import Path


MAIN_FILE = Path("MHD_Solvers/solvers/epotMultiRegionInterFoam/epotMultiRegionInterFoam.C")
EPOT_FILE = Path("MHD_Solvers/solvers/epotMultiRegionInterFoam/fluid/ePotEqn.H")
UEQN_FILE = Path("MHD_Solvers/solvers/epotMultiRegionInterFoam/fluid/mhdUEqn.H")
PEQN_FILE = Path("MHD_Solvers/solvers/common/interFoam/fluid/pEqn.H")


def _inject_once(source: str, needle: str, insertion: str) -> str:
    if insertion.strip() in source:
        return source
    if needle not in source:
        raise ValueError(f"Needle not found for patch:\n{needle}")
    return source.replace(needle, needle + insertion, 1)


def patch_main_file(source: str) -> str:
    flag_block = """
    const bool logCoupledMhdIterations
    (
        runTime.controlDict().lookupOrDefault<bool>("logCoupledMhdIterations", false)
    );
"""
    source = _inject_once(
        source,
        '   #include "createMhdFields.H"\n',
        flag_block,
    )
    outer_log = """
            if (logCoupledMhdIterations)
            {
                Info<< "LMX_DIAG outer"
                    << " time=" << runTime.timeName()
                    << " oCorr=" << oCorr
                    << " nOuterCorr=" << nOuterCorr
                    << " finalIter=" << finalIter
                    << endl;
            }
"""
    source = _inject_once(
        source,
        "            const bool finalIter = (oCorr == nOuterCorr-1);\n",
        outer_log,
    )
    return source


def patch_epot_file(source: str) -> str:
    source = source.replace('<< " maxPotE=" << gMax(mag(potE))', '<< " maxPotE=" << max(mag(potE)).value()')
    source = source.replace('<< " maxJ=" << gMax(mag(J))', '<< " maxJ=" << max(mag(J)).value()')
    source = source.replace('<< " maxJxB=" << gMax(mag(JxB))', '<< " maxJxB=" << max(mag(JxB)).value()')
    source = source.replace(
        '<< " maxJnDensity=" << max(mag(jn/(mesh.magSf() + SMALL))).value()',
        '<< " maxJnDensity=" << max(mag(jn/mesh.magSf())).value()',
    )
    source = source.replace(
        '<< " maxPsiubDensity=" << max(mag(psiub/(mesh.magSf() + SMALL))).value()',
        '<< " maxPsiubDensity=" << max(mag(psiub/mesh.magSf())).value()',
    )
    if '<< " maxJnDensity="' not in source:
        source = source.replace(
            '<< " maxJn=" << max(mag(jn)).value()',
            '<< " maxJn=" << max(mag(jn)).value()\n'
            '\t\t\t<< " maxJnDensity=" << max(mag(jn/mesh.magSf())).value()',
        )
    if '<< " maxPsiubDensity="' not in source:
        source = source.replace(
            '<< " maxPsiub=" << max(mag(psiub)).value()',
            '<< " maxPsiub=" << max(mag(psiub)).value()\n'
            '\t\t\t<< " maxPsiubDensity=" << max(mag(psiub/mesh.magSf())).value()',
        )
    conservative_marker = "\tvolVectorField centeredJxB(J ^ B);\n"
    if conservative_marker.strip() not in source:
        if "\t//Update current density distribution and boundary condition\n\tJ.correctBoundaryConditions();\n" in source:
            source = _inject_once(
                source,
                "\t//Update current density distribution and boundary condition\n\tJ.correctBoundaryConditions();\n",
                conservative_marker,
            )
        else:
            source = _inject_once(
                source,
                "\t\tJxB = (J ^ B);\n",
                conservative_marker,
            )
    solve_old = "\tPotEEqn.solve();\n"
    solve_new = """\tauto potEPerf = PotEEqn.solve();\n"""
    if "auto potEPerf = PotEEqn.solve();" not in source:
        if solve_old not in source:
            raise ValueError("Could not find PotEEqn.solve() call")
        source = source.replace(solve_old, solve_new, 1)

    log_block = """
\tif (logCoupledMhdIterations)
\t{
\t\tInfo<< "LMX_DIAG epot"
\t\t\t<< " time=" << runTime.timeName()
\t\t\t<< " region=" << mesh.name()
\t\t\t<< " oCorr=" << oCorr
\t\t\t<< " potEInitialResidual=" << potEPerf.initialResidual()
\t\t\t<< " potEFinalResidual=" << potEPerf.finalResidual()
\t\t\t<< " potEIterations=" << potEPerf.nIterations()
\t\t\t<< " maxPotE=" << max(mag(potE)).value()
\t\t\t<< " maxJ=" << max(mag(J)).value()
\t\t\t<< " maxJn=" << max(mag(jn)).value()
\t\t\t<< " maxJnDensity=" << max(mag(jn/mesh.magSf())).value()
\t\t\t<< " maxPsiub=" << max(mag(psiub)).value()
\t\t\t<< " maxPsiubDensity=" << max(mag(psiub/mesh.magSf())).value()
\t\t\t<< " maxCenteredJxB=" << max(mag(centeredJxB)).value()
\t\t\t<< " maxJxB=" << max(mag(JxB)).value()
\t\t\t<< endl;
\t}
"""
    if 'Info<< "LMX_DIAG epot"' not in source:
        if "\t\tJxB = (J ^ B);\n\t}\n" in source:
            source = _inject_once(
                source,
                "\t\tJxB = (J ^ B);\n\t}\n",
                log_block,
            )
        else:
            source = _inject_once(
                source,
                "\t}\n",
                log_block,
            )
    return source


def patch_ueqn_file(source: str) -> str:
    source = source.replace('<< " maxU=" << gMax(mag(U))', '<< " maxU=" << max(mag(U)).value()')
    source = source.replace('<< " maxJxB=" << gMax(mag(JxB))', '<< " maxJxB=" << max(mag(JxB)).value()')
    solve_old = """        solve
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
"""
    solve_new = """        auto UPerf = solve
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
"""
    if "auto UPerf = solve" not in source:
        if solve_old not in source:
            raise ValueError("Could not find momentum solve block")
        source = source.replace(solve_old, solve_new, 1)

    log_block = """
        if (logCoupledMhdIterations)
        {
            Info<< "LMX_DIAG momentum"
                << " time=" << runTime.timeName()
                << " region=" << mesh.name()
                << " oCorr=" << oCorr
                << " UInitialResidual=" << UPerf.initialResidual()
                << " UFinalResidual=" << UPerf.finalResidual()
                << " UIterations=" << UPerf.nIterations()
                << " maxU=" << max(mag(U)).value()
                << " maxJxB=" << max(mag(JxB)).value()
                << endl;
        }
"""
    source = _inject_once(
        source,
        "        K = 0.5*magSqr(U);\n",
        log_block,
    )
    return source


def patch_peqn_file(source: str) -> str:
    source = source.replace(
        '<< " maxU=" << max(mag(U)).value()\n                    << " maxJxB=" << max(mag(JxB)).value()',
        '<< " maxU=" << max(mag(U)).value()\n'
        '                    << " maxP=" << max(mag(p)).value()\n'
        '                    << " maxPRgh=" << max(mag(p_rgh)).value()\n'
        '                    << " maxJxB=" << max(mag(JxB)).value()',
    )
    solve_old = """        solve
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
"""
    solve_new = """        auto pPerf = solve
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
"""
    if "auto pPerf = solve" not in source:
        if solve_old not in source:
            raise ValueError("Could not find pressure solve block")
        source = source.replace(solve_old, solve_new, 1)

    log_block = """
            if (logCoupledMhdIterations)
            {
                Info<< "LMX_DIAG pressure"
                    << " time=" << runTime.timeName()
                    << " region=" << mesh.name()
                    << " oCorr=" << oCorr
                    << " corr=" << corr
                    << " nonOrth=" << nonOrth
                    << " pInitialResidual=" << pPerf.initialResidual()
                    << " pFinalResidual=" << pPerf.finalResidual()
                    << " pIterations=" << pPerf.nIterations()
                    << " maxU=" << max(mag(U)).value()
                    << " maxP=" << max(mag(p)).value()
                    << " maxPRgh=" << max(mag(p_rgh)).value()
                    << " maxJxB=" << max(mag(JxB)).value()
                    << endl;
            }
"""
    source = _inject_once(
        source,
        "            fvOptions.correct(U);\n",
        log_block,
    )
    return source


def patch_freemhd_tree(root: Path) -> list[Path]:
    targets = {
        MAIN_FILE: patch_main_file,
        EPOT_FILE: patch_epot_file,
        UEQN_FILE: patch_ueqn_file,
        PEQN_FILE: patch_peqn_file,
    }
    updated: list[Path] = []
    for relative_path, patcher in targets.items():
        path = root / relative_path
        source = path.read_text()
        patched = patcher(source)
        if patched != source:
            path.write_text(patched)
            updated.append(path)
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Patch FreeMHD epotMultiRegionInterFoam with coupled-iteration diagnostics.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/Users/rogerio/local/tests/LMX/external/FreeMHD"),
        help="FreeMHD checkout root",
    )
    args = parser.parse_args(argv)

    updated = patch_freemhd_tree(args.root)
    for path in updated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
