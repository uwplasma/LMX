from __future__ import annotations

import argparse
from pathlib import Path


MAIN_FILE = Path("MHD_Solvers/solvers/epotMultiRegionInterFoam/epotMultiRegionInterFoam.C")
EPOT_FILE = Path("MHD_Solvers/solvers/epotMultiRegionInterFoam/fluid/ePotEqn.H")
UEQN_FILE = Path("MHD_Solvers/solvers/epotMultiRegionInterFoam/fluid/mhdUEqn.H")


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
\t\t\t<< " maxPotE=" << gMax(mag(potE))
\t\t\t<< " maxJ=" << gMax(mag(J))
\t\t\t<< " maxJxB=" << gMax(mag(JxB))
\t\t\t<< endl;
\t}
"""
    source = _inject_once(
        source,
        "\t\tJxB = (J ^ B);\n\t}\n",
        log_block,
    )
    return source


def patch_ueqn_file(source: str) -> str:
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
                << " maxU=" << gMax(mag(U))
                << " maxJxB=" << gMax(mag(JxB))
                << endl;
        }
"""
    source = _inject_once(
        source,
        "        K = 0.5*magSqr(U);\n",
        log_block,
    )
    return source


def patch_freemhd_tree(root: Path) -> list[Path]:
    targets = {
        MAIN_FILE: patch_main_file,
        EPOT_FILE: patch_epot_file,
        UEQN_FILE: patch_ueqn_file,
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
