from __future__ import print_function

import csv
import json
import math
import os
import re
import sys


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def main():
    case_dir = sys.argv[1]
    out_dir = sys.argv[2]
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    latest = latest_time_dir(case_dir)
    variables = parse_block_mesh_variables(os.path.join(case_dir, "constant/polyMesh/blockMeshDict"))
    transport = parse_transport_properties(os.path.join(case_dir, "constant/transportProperties"))
    vectors = parse_openfoam_vector_field(os.path.join(latest, "U"))
    b_field = parse_openfoam_scalar_or_uniform(os.path.join(latest, "B"))

    nx = int(round(variables.get("Nx", 1.0)))
    y_centers, y_widths = q2d_fully_developed_y_cells(variables)
    expected = nx * len(y_centers)
    if len(vectors) != expected:
        raise ValueError("Expected {0} velocity cells, found {1}".format(expected, len(vectors)))

    ux_profile = []
    for row_index in range(len(y_centers)):
        row = vectors[row_index * nx : (row_index + 1) * nx]
        ux_profile.append(sum(vector[0] for vector in row) / nx)

    half_width = abs(float(transport.get("b", max(abs(y) for y in y_centers))))
    weighted_mean = sum(u * w for u, w in zip(ux_profile, y_widths)) / sum(y_widths)
    peak = max(ux_profile)
    target = float(transport.get("Ubar_x", weighted_mean))
    density = float(transport.get("rho0", 1.0))
    viscosity = float(transport.get("nu", 1.0))
    conductivity = float(transport.get("sigma", 1.0))
    hartmann = abs(b_field) * half_width * math.sqrt(conductivity / max(density * viscosity, 1.0e-300))
    symmetry_l2 = profile_symmetry_l2(y_centers, ux_profile, weighted_mean)
    final_time = float(os.path.basename(latest))
    steady = "Steady-state criteria reached" in read_text(os.path.join(case_dir, "log.Q2DmhdFoam"))

    with open(os.path.join(out_dir, "profile.csv"), "wb") as handle:
        writer = csv.writer(handle)
        writer.writerow(("y", "y_over_b", "ux", "ux_over_mean", "ux_over_peak"))
        for y, ux in zip(y_centers, ux_profile):
            writer.writerow(
                (
                    "{0:.16e}".format(y),
                    "{0:.16e}".format(y / half_width),
                    "{0:.16e}".format(ux),
                    "{0:.16e}".format(ux / weighted_mean),
                    "{0:.16e}".format(ux / peak),
                )
            )

    rank_count = 2 if os.path.exists(os.path.join(case_dir, "log.decomposePar")) else 1
    summary = {
        "case": "Q2DmhdFoam/Q2DfullyDeveloped",
        "status": "external_reference_case_complete" if steady else "external_reference_case_finished_without_steady_marker",
        "final_time": final_time,
        "rank_count": rank_count,
        "cell_count": len(vectors),
        "profile_sample_count": len(y_centers),
        "x_cell_count": nx,
        "hartmann": hartmann,
        "magnetic_field": b_field,
        "mean_velocity": weighted_mean,
        "target_mean_velocity": target,
        "flow_rate_relative_error": abs(weighted_mean - target) / max(abs(target), 1.0e-300),
        "peak_to_mean_velocity": peak / weighted_mean,
        "center_normalized_velocity": interpolate_center(y_centers, ux_profile) / weighted_mean,
        "edge_normalized_velocity": 0.5 * (ux_profile[0] + ux_profile[-1]) / weighted_mean,
        "symmetry_l2": symmetry_l2,
        "source": "Q2DmhdFoam foam-extend 4.1 tutorial Q2DfullyDeveloped",
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_text(path):
    with open(path, "r") as handle:
        return handle.read()


def latest_time_dir(case_dir):
    candidates = []
    for name in os.listdir(case_dir):
        path = os.path.join(case_dir, name)
        if not os.path.isdir(path):
            continue
        try:
            value = float(name)
        except ValueError:
            continue
        candidates.append((value, path))
    if not candidates:
        raise IOError("No OpenFOAM time directories found under {0}".format(case_dir))
    return max(candidates, key=lambda item: item[0])[1]


def parse_block_mesh_variables(path):
    variables = {}
    for line in read_text(path).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z][A-Za-z0-9_]*)\s+(" + FLOAT + r")\s*;", stripped)
        if match:
            variables[match.group(1)] = float(match.group(2))
    return variables


def parse_transport_properties(path):
    text = read_text(path)
    result = {}
    for key in ("rho0", "nu", "sigma", "a", "b"):
        match = re.search(r"\b" + re.escape(key) + r"\s+(?:(?:\[[^\]]+\])\s*)?(" + FLOAT + r")\s*;", text)
        if match:
            result[key] = float(match.group(1))
    match = re.search(r"\bUbar\s+(?:\[[^\]]+\]\s*)?\((" + FLOAT + r")\s+(" + FLOAT + r")\s+(" + FLOAT + r")\)", text)
    if match:
        result["Ubar_x"] = float(match.group(1))
    return result


def parse_openfoam_vector_field(path):
    text = read_text(path)
    match = re.search(r"internalField\s+nonuniform\s+List<vector>\s+\d+\s*\((.*?)\)\s*;", text, re.S)
    if not match:
        raise ValueError("{0} does not contain a nonuniform vector internalField".format(path))
    vectors = []
    for vector_match in re.finditer(r"\((" + FLOAT + r")\s+(" + FLOAT + r")\s+(" + FLOAT + r")\)", match.group(1)):
        vectors.append(tuple(float(vector_match.group(index)) for index in (1, 2, 3)))
    return vectors


def parse_openfoam_scalar_or_uniform(path):
    text = read_text(path)
    uniform = re.search(r"internalField\s+uniform\s+(" + FLOAT + r")\s*;", text)
    if uniform:
        return float(uniform.group(1))
    nonuniform = re.search(r"internalField\s+nonuniform\s+List<scalar>\s+\d+\s*\((.*?)\)\s*;", text, re.S)
    if nonuniform:
        values = [float(item) for item in re.findall(FLOAT, nonuniform.group(1))]
        if values:
            return sum(values) / len(values)
    raise ValueError("{0} does not contain a scalar internalField".format(path))


def q2d_fully_developed_y_cells(variables):
    y_wall = float(variables["y"])
    y_wall_neg = float(variables["yNeg"])
    y_bl = float(variables["yBL"])
    y_bl_neg = float(variables["yNegBL"])
    ny = int(round(variables["Ny"]))
    ny_bl = int(round(variables["NyBL"]))
    segments = [
        (y_wall_neg, y_bl_neg, ny_bl, float(variables["GyBL"])),
        (y_bl_neg, 0.0, ny, float(variables["Gy"])),
        (0.0, y_bl, ny, float(variables["GyInv"])),
        (y_bl, y_wall, ny_bl, float(variables["GyBLinv"])),
    ]
    centers = []
    widths = []
    for start, end, count, grading in segments:
        for center, width in graded_cell_centers(start, end, count, grading):
            centers.append(center)
            widths.append(width)
    return centers, widths


def graded_cell_centers(start, end, count, ratio):
    length = end - start
    sign = 1.0 if length >= 0.0 else -1.0
    length_abs = abs(length)
    if count <= 0:
        return []
    if count == 1 or abs(ratio - 1.0) < 1.0e-12:
        widths = [length_abs / count] * count
    else:
        q = ratio ** (1.0 / (count - 1))
        first = length_abs * (1.0 - q) / (1.0 - q**count)
        widths = [first * q**index for index in range(count)]
    cells = []
    cursor = start
    for width_abs in widths:
        width = sign * width_abs
        cells.append((cursor + 0.5 * width, abs(width)))
        cursor += width
    return cells


def interpolate_center(y, u):
    for index in range(len(y) - 1):
        left = y[index]
        right = y[index + 1]
        if left <= 0.0 <= right:
            weight = (0.0 - left) / max(right - left, 1.0e-300)
            return u[index] * (1.0 - weight) + u[index + 1] * weight
    return u[min(range(len(y)), key=lambda index: abs(y[index]))]


def profile_symmetry_l2(y, u, mean):
    pairs = []
    for yi, ui in zip(y, u):
        mirrored = interp_sorted(y, u, -yi)
        pairs.append((ui / mean - mirrored / mean) ** 2)
    return math.sqrt(sum(pairs) / len(pairs))


def interp_sorted(x, y, target):
    if target <= x[0]:
        return y[0]
    if target >= x[-1]:
        return y[-1]
    for index in range(len(x) - 1):
        if x[index] <= target <= x[index + 1]:
            weight = (target - x[index]) / (x[index + 1] - x[index])
            return y[index] * (1.0 - weight) + y[index + 1] * weight
    return y[-1]


if __name__ == "__main__":
    main()
