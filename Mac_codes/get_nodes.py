from paraview.simple import *
import numpy as np

# Load your mesh
mesh_file = "zoned_mesh.inp"
output_file = "nodes_report.txt"
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
    (0,	0,	22000),
    (0,	0,	20500),
    (0,	0,	20000),
    (0,	0,	19500),
    (0,	0,	19000),
    (0,	0,	15000),
    (0,	0,	10500),
    (0,	0,	10000),
    (0,	0,	9500),
    (0,	0,	7000),
    (0,	0,	-9000),
    (0,	0,	-25000)
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
