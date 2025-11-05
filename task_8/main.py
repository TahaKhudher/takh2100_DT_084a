import open3d as o3d

# Load PLY file
pcd = o3d.io.read_point_cloud("structure_3d.ply")
o3d.visualization.draw_geometries([pcd])
