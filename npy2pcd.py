#!/usr/bin/env python
from data import *
import open3d as o3d


image_rows_low = 360 # 8, 16, 32
image_rows_high = 360 # 16, 32, 64
image_rows_full = 360

# ouster
ang_res_x = 120.0/float(image_cols) # horizontal resolution
ang_res_y = 30.0/float(image_rows_high-1) # vertical resolution
ang_start_y = 15.0 # bottom beam angle
sensor_noise = 0.03
max_range = 200.0
min_range = 2.0

upscaling_factor = int(image_rows_high / image_rows_low)

class PointCloudProcessor:
    """Process Point Cloud"""
    def __init__(self):
        self.rowList = []
        self.colList = []
        for i in range(image_rows_high):
            self.rowList = np.append(self.rowList, np.ones(image_cols)*i)
            self.colList = np.append(self.colList, np.arange(image_cols))

        self.verticalAngle = np.float32(self.rowList * ang_res_y) - ang_start_y
        self.horizonAngle = - np.float32(self.colList + 1 - (image_cols/2)) * ang_res_x + 90.0
        
        self.verticalAngle = self.verticalAngle / 180.0 * np.pi
        self.horizonAngle = self.horizonAngle / 180.0 * np.pi

        self.intensity = self.rowList + self.colList / image_cols
        
    def publishPointCloud(self, thisImage, height):
        # multi-channel range image, the first channel is range
        if len(thisImage.shape) == 3:
            thisImage = thisImage[:,:,0]

        lengthList = thisImage.reshape(image_rows_high*image_cols)
        lengthList[lengthList > max_range] = 0.0
        lengthList[lengthList < min_range] = 0.0

        x = np.sin(self.horizonAngle) * np.cos(self.verticalAngle) * lengthList
        y = np.cos(self.horizonAngle) * np.cos(self.verticalAngle) * lengthList
        z = np.sin(self.verticalAngle) * lengthList + height
        
        points = np.column_stack((x,y,z))
        # delete points that has range value 0
        points = np.delete(points, np.where(lengthList==0), axis=0) # comment this line for visualize at the same speed (for video generation)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud("./sync.pcd", pcd)
        return True

    def test_comparison(self):

        # load images
        # ground truth image: n x rows x cols x 1
        origImages = np.load(os.path.join(home_dir, 'Documents', project_name, 'test', 'data.npy')) # 
        # predicted image: n x rows x cols x 2
        #predImages = np.load(os.path.join(home_dir, 'Documents', project_name, 'ouster_range_image-from-16-to-64_prediction.npy')) * normalize_ratio # 

        print ('Input range image shape:')
        print (origImages.shape)
        #print (predImages.shape)

        #low_res_index = range(0, image_rows_high, upscaling_factor)
        #predImages[:,low_res_index,:,0:1] = origImages[:,low_res_index,:,0:1] # copy some of beams from origImages to predImages
        origImagesSmall = np.zeros(origImages.shape, dtype=np.float32) # for visualizing NN input images
        #origImagesSmall[:,low_res_index] = origImages[:,low_res_index]

        #predImagesNoiseReduced = np.copy(predImages[:,:,:,0:1])

        # remove noise
        '''
        if len(predImages.shape) == 4 and predImages.shape[-1] == 2:
            noiseLabels = predImages[:,:,:,1:2]
            predImagesNoiseReduced[noiseLabels > predImagesNoiseReduced * 0.03] = 0 # after noise removal
            predImagesNoiseReduced[:,low_res_index] = origImages[:,low_res_index]

        predImages[:,:,:,1:2] = None
        '''
        for i in range(0, len(origImages), 1):
            print(len(origImages))
            #timeStamp = rospy.Time.now()

            self.publishPointCloud(origImages[i], 0)

            print ('Displaying {} th of {} images'.format(i, len(origImages)) )
            # raw_input('Displaying {} th of {} images. Press ENTER to continue ... '.format(i, len(origImages)) )
  

if __name__ == '__main__':

    #rospy.init_node("cloud_reader")
    
    reader = PointCloudProcessor()

    reader.test_comparison()