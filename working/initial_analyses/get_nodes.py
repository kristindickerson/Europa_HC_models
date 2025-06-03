from paraview.simple import *
import numpy as np

# Load your mesh
folder = '/home/krdicker/eu-kd/Europa_HC_models/working/fehm/europa/100_100_50_km_domain/200msed/3_large_fractures/mesh'
mesh_file = folder + '/zoned_mesh.inp'
output_file = folder + '/nodes_report_.txt'
mesh = OpenDataFile(mesh_file)
mesh.UpdatePipeline()

# Get point and cell data
data = servermanager.Fetch(mesh)
points_vtk = data.GetPoints()

if not points_vtk:
    raise RuntimeError("Failed to load point coordinates from mesh.")

point_data = data.GetPointData()
imt_array = point_data.GetArray("imt1")

if not imt_array:
    raise RuntimeError("Could not find 'imt1' array in point data.")

# Define coordinates to check
target_coords = [
    (0,	0,	25000),
    (0,	0,	24500),
    (0,	0,	24500),
    (0,	0,	24500),
    (0,	0,	24500),
    (0,	0,	24000),
    (0,	0,	20000),
    (0,	0,	15000),
    (0,	0,	14500),
    (0,	0,	14000),
    (0,	0,	13000),
    (0,	0,	-5000),
    (0,	0,	-25000),
    (0,	-15000,	25000),
    (0,	-15000,	24500),
    (0,	-15000,	24500),
    (0,	-15000,	24500),
    (0,	-15000,	24500),
    (0,	-15000,	24000),
    (0,	-15000,	20000),
    (0,	-15000,	15000),
    (0,	-15000,	14500),
    (0,	-15000,	14000),
    (0,	-15000,	13000),
    (0,	-16000,	-5000),
    (0,	-16000,	-25000),
    (0,	15000,	25000),
    (0,	15000,	24500),
    (0,	15000,	24500),
    (0,	15000,	24500),
    (0,	15000,	24500),
    (0,	15000,	24000),
    (0,	15000,	20000),
    (0,	15000,	15000),
    (0,	15000,	14500),
    (0,	15000,	14000),
    (0,	15000,	13000),
    (0,	14000,	-5000),
    (0,	14000,	-25000),
    (0,	-8000,	25000),
    (0,	-8000,	24500),
    (0,	-8000,	24500),
    (0,	-8000,	19000),
    (0,	-8000,	14500),
    (0,	-8000,	-5000),
    (0,	-8000,	-25000),
    (0,	8000,	25000),
    (0,	8000,	24500),
    (0,	8000,	24500),
    (0,	8000,	19000),
    (0,	8000,	14500),
    (0,	8000,	-5000),
    (0,	8000,	-25000),
    (0,	8000,	25000),
    (0,	30000,	24500),
    (0,	30000,	24500),
    (0,	30000,	19000),
    (0,	30000,	14500),
    (0,	30000,	-5000),
    (0,	30000,	-25000)
]
tolerance = 1e-3

# Open output file and write header
with open(output_file, "w") as f:
    f.write(f"{'X':>10} {'Y':>10} {'Z':>10} {'PointID':>10} {'imt':>5}\n")
    f.write("=" * 50 + "\n")

    for target in target_coords:
        found = False
        for i in range(points_vtk.GetNumberOfPoints()):
            pt = points_vtk.GetPoint(i)
            if np.allclose(pt, target, atol=tolerance):
                x, y, z = map(int, pt)
                imt_val = int(imt_array.GetValue(i))
                f.write(f"{x:10d} {y:10d} {z:10d} {i:10d} {imt_val:5d}\n")
                found = True
                break
        if not found:
            f.write(f"{int(target[0]):10d} {int(target[1]):10d} {int(target[2]):10d} {'N/A':>10} {'N/A':>5}\n")

print(f"✔ Done. Report written to {output_file}")
