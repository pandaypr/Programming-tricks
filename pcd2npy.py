#!/usr/bin/env python
from data import *
import open3d as o3d
#from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian
# range image size, depends on your sensor, i.e., VLP-16: 16x1800, OS1-64: 64x1024
image_rows_full = 360
image_cols = 374

# Ouster OS1-64 (gen1)
ang_res_x = 120.0/float(image_cols) # horizontal resolution
ang_res_y = 30/float(image_rows_full-1) # vertical resolution
ang_start_y = 15 # bottom beam angle
max_range = 200.0
min_range = 2.0

# Convert ros bags to npy
def bag2np(data_set_name, npy_file_name):
    """
    convert all .bag files in a specific folder to a single .npy range image file.
    """
    print('#'*50)
    print('Dataset name: {}'.format(data_set_name))
    range_image_array = np.empty([0, image_rows_full, image_cols, 1], dtype=np.float32)
    # find all bag files in the given dir
    bag_file_path = os.path.join(data_set_name)
    bag_files = os.listdir(bag_file_path)
    print (bag_files)
    # loop through all bags in the given directory
    for file_name in bag_files:
        file_path = os.path.join(bag_file_path, file_name)
        pcd = o3d.io.read_point_cloud(file_path)  
        pcd_np = np.asarray(pcd.points)
        range_image = np.zeros((1, image_rows_full, image_cols, 1), dtype=np.float32)
        x = pcd_np[:,0]
        y = pcd_np[:,1]
        z = pcd_np[:,2]
        # find row id
        vertical_angle = np.arctan2(z, np.sqrt(x * x + y * y)) * 180.0 / np.pi
        relative_vertical_angle = vertical_angle + ang_start_y
        rowId = np.int_(np.round_(relative_vertical_angle / ang_res_y))
        # find column id
        horitontal_angle = np.arctan2(x, y) * 180.0 / np.pi
        colId = -np.int_((horitontal_angle-90.0)/ang_res_x) + image_cols/2
        shift_ids = np.where(colId>=image_cols)
        colId[shift_ids] = colId[shift_ids] - image_cols
        # filter range
        thisRange = np.sqrt(x * x + y * y + z * z)
        thisRange[thisRange > max_range] = 0
        thisRange[thisRange < min_range] = 0
        # save range info to range image
        for i in range(len(thisRange)):
            if rowId[i] < 0 or rowId[i] >= image_rows_full or colId[i] < 0 or colId[i] >= image_cols:
                continue
            range_image[0, int(rowId[i]), int(colId[i]), 0] = thisRange[i]
        # append range image to array
        range_image_array = np.append(range_image_array, range_image, axis=0)

    # save full resolution image array
    np.save(npy_file_name, range_image_array)
    print('Dataset saved: {}'.format(npy_file_name))


if __name__=='__main__':
    
    # put all the bags that can be trained in this folder, the package will combine them as a single npy file
    data_set_name = os.path.join(home_dir, 'Documents', project_name, 'test')
    # your point cloud ros topic in these bag files, i.e., velodyne_points
    #pointcloud_topic = '/os1_node/points'
    # the location where your npy file will be saved
    npy_file_name = os.path.join(home_dir, 'Documents', project_name, 'test', "data.npy")

    bag2np(data_set_name, npy_file_name)