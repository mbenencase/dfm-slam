#!/home/mbenencase/anaconda3/envs/torch1.11/bin/python

import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import time

import geometry
from DFM import DeepFeatureMatcher


class FeatureTrackerNode(object):
    def __init__(self):
        rospy.init_node("feature_tracker_node", anonymous=True)

        self.cv_bridge = CvBridge()

        # --- Feature Tracker System Variables ---- #
        self.last_img = None

        self.MAX_FEATURES = 200
        self.K = np.array([
            525.0, 0.0, 319.5,
            0.0, 525.0, 239.5,
            0.0,   0.0,   1.0
        ]).reshape(3, 3)
        self.K_inv = np.linalg.inv(self.K)
        self.matcher = DeepFeatureMatcher(
            enable_two_stage = True,
            model = 'VGG19',
            ratio_th = [0.9, 0.9, 0.9, 0.9, 0.95, 1.0],
            bidirectional = True
        )

        rgb_sub = message_filters.Subscriber('/camera/rgb/image_color', Image)
        depth_sub = message_filters.Subscriber('/camera/depth/image', Image)

        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=1, slop=0.05)
        ts.registerCallback(self.camera_callback)

        self.T = np.eye(4)
        self.p3d = list()

        rospy.spin()


    def camera_callback(self, rgb_msg : Image, depth_msg : Image):
        rgb_img = self.cv_bridge.imgmsg_to_cv2(rgb_msg)
        rgb_img = cv2.resize(rgb_img, dsize=(0, 0), fx=0.5, fy=0.5)

        depth_img = self.cv_bridge.imgmsg_to_cv2(depth_msg)
        depth_img = np.nan_to_num(depth_img)

        if self.last_img is None:
            self.last_img = np.array(rgb_img)
            return
        else:
            # --- Feature Matching --- #
            start = time.perf_counter()
            _, _, kpts0, kpts1 = self.matcher.match(
                rgb_img.copy(), self.last_img.copy()
            )
            end = time.perf_counter()
            spent = (end - start) * 1000.0
            
            rospy.loginfo("Inference time = {:.2f} [ms]".format(spent))

            kpts0 = kpts0.T * 2.0 + 0.5
            kpts1 = kpts1.T * 2.0 + 0.5
            kpts1_int32 : np.ndarray = np.int0(kpts1)

            x_int32, y_int32 = kpts1_int32[:, 0], kpts1_int32[:, 1]

            depths = depth_img[y_int32, x_int32]

            # -- Removing invalid depths --- #
            valid_depths = np.where(depths != 0)[0]
            depths = depths[valid_depths]
            kpts1 = kpts1[valid_depths]
            kpts0 = kpts0[valid_depths]

            p3d = geometry.get_3D_points(kpts1, self.K, depths)

            # -- Pose Estimation --- #
            start = time.perf_counter()
            ret, rvec, tvec, _ = cv2.solvePnPRansac(
                p3d.reshape(-1, 1, 3).astype(np.float32),
                kpts0.reshape(-1, 1, 2).astype(np.float32),
                self.K.astype(np.float32),
                np.zeros((4, ), dtype=np.float32),
                iterationsCount=500,
                confidence=0.99,
                reprojectionError=1.5
            )
            end = time.perf_counter()
            spent = (end - start) * 1000.0
            rospy.loginfo("[INFO] Pose estimation time = {:.2f} [ms]".format(spent))

            self.last_img = np.array(rgb_img)

            R = cv2.Rodrigues(rvec)[0]
            tvec = tvec.reshape(-1)

            T = np.eye(4)
            T[0:3, 0:3] = R
            T[0:3, 3] = tvec

            # --- Applying Update Transform --- #
            self.T = T @ self.T

            p3d = (np.linalg.inv(self.T) @ np.hstack([p3d, np.ones((len(p3d), 1))]).T).T
            p3d = p3d[:, :3].tolist()

            # Including the new points to verify the reconstruction
            self.p3d += p3d

            # NOTE: Not necessary, just for visualization
            np.savetxt("p3d.xyz", self.p3d)

            return


def main():
    FeatureTrackerNode()


if __name__ == "__main__":
    main()