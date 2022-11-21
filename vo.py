import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp
import time

class SimpleVO:
    def __init__(self, args):
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params['K']
        self.dist = camera_params['dist']
        
        self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))
        
    def draw_camera(self, R, t):    
        points = np.array([[0, 0, 1], [1080/2, 0, 1], [1080/2, 1920/2, 1], [0, 1920/2, 1]])
        points = np.linalg.pinv(self.K)@points.T
        points =  t.reshape(3,1) + R@points
        points = points.T 

        # get center and concate all
        all_points = np.ones((points.shape[0]+1, points.shape[1]))
        all_points[:-1, :] = points
        # print(all_points)

        all_points[-1, :] = t
        # print(all_points)

        # set up line
        camera = o3d.geometry.LineSet(
            points = o3d.utility.Vector3dVector(all_points),
            lines = o3d.utility.Vector2iVector([[0,1], [1,2], [2, 3], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4]])
        )
        # print('git caemra', camera)

        # generate color
        color = [0, 0, 0]
        colors = np.tile(color, (8,1))

        camera.colors = o3d.utility.Vector3dVector(colors)

        return camera
    
    def get_scale_factor(self, points_past, points_prev, X_prev, X_curr):
        X_prev, X_curr = X_prev.T, X_curr.T
        scale_factor = []
        same_idx = []
        for i in range(points_past.shape[0]):
            for j in range(points_prev.shape[0]):
                if np.all(points_past[i] == points_prev[j]):
                    same_idx.append([X_prev[i] ,X_curr[j]])
        if len(same_idx) <= 1:
            return 1
        
        for i in range(len(same_idx)):
            for j in range(len(same_idx)):
                if i != j:
                    scale_factor.append(np.linalg.norm(same_idx[i][0] - same_idx[i][1])/np.linalg.norm(same_idx[j][0] - same_idx[j][1]))

        return np.mean(scale_factor)

    def process_frames(self, queue):
        print(f'\n\nMethod Using : {args.method}\n\n')
        R, t = np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
        # print(f'Intrinsic Matrix : {self.K}')
        # print(f'Distortion Coefficient : {self.dist}')
        curr_pose = np.eye(4, dtype=np.float64)
        prev_img = cv.imread(self.frame_paths[0])
        first_img = True
        
        start_time = time.time()

        for frame_path in self.frame_paths[1:]:
            img = cv.imread(frame_path)
            #TODO: compute camera pose here
            if args.method == 'orb':
                orb = cv.ORB_create()

                kp_prev, des_prev = orb.detectAndCompute(prev_img, None)
                kp_curr, des_curr = orb.detectAndCompute(img, None)

                bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
                
                matches = bf.match(des_curr, des_prev)
                
                matches = sorted(matches, key = lambda x:x.distance)
            
            else:
                sift = cv.SIFT_create()

                kp_prev, des_prev = sift.detectAndCompute(prev_img, None)
                kp_curr, des_curr = sift.detectAndCompute(img, None)

                bf = cv.BFMatcher()

                matches = bf.match(des_curr, des_prev)
                
                matches = sorted(matches, key = lambda x:x.distance)
                
            points_prev = np.array([kp_prev[m.trainIdx].pt for m in matches])
            points_curr = np.array([kp_curr[m.queryIdx].pt for m in matches])
            past_idx = [m.trainIdx for m in matches]

            norm_prev = cv.undistortPoints(points_prev, self.K, self.dist, None, self.K)
            norm_curr = cv.undistortPoints(points_curr, self.K, self.dist, None, self.K)
            norm_prev = points_prev
            norm_curr = points_curr
            # print(points_prev[:3])
            # print(norm_prev[:3], '\n')
            # print(points_curr[:3])
            # print(norm_curr[:3])
            
            E, mask = cv.findEssentialMat(norm_prev, norm_curr, self.K, cv.RANSAC, 0.999, 1.0)
            # print("E",E)
            # print('mask',mask)
            
            retval, R, t, mask, X_curr = cv.recoverPose(E, norm_prev, norm_curr, \
                                                            self.K, distanceThresh=1000, mask = mask)
            
            X_curr = X_curr[:3,:] / X_curr[3,:].reshape(1,-1) # 3*N
            
            scale_factor = 1
            if first_img:
                scale_factor = 1
                # first_img = False
            else:
                scale_factor = self.get_scale_factor(self.norm_past, norm_prev, self.X_prev, X_curr)
                if scale_factor > 2:
                    scale_factor = 2
                t = scale_factor * t
            
            #step5 
            #print("scale_factor",scale_factor)
            frame_pose = np.concatenate([R, t], -1)
            frame_pose = np.concatenate([frame_pose, np.zeros((1, 4))], 0)
            frame_pose[-1, -1] = 1.0

            curr_pose = curr_pose @ frame_pose
            # print('\n\ncurr pose', curr_pose)

            R = curr_pose[:3, :3]
            t = -(curr_pose[:3, 3])
            
            prev_img = img
            self.norm_past = norm_curr
            self.X_prev = X_curr

            queue.put((R, t))

            img_show = cv.drawKeypoints(img, kp_curr, None, color=(0, 255, 0))
            cv.imshow('frame', img_show)
            if first_img:
                cv.imwrite('orb.jpg', img_show)
                first_img = False
            
            if cv.waitKey(30) == 27: break
        
        end_time = time.time()
        
        print(f'\n\nTotal run time : {end_time - start_time}\n\n')
            
    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window(width = 1000, height=1000)
        
        queue = mp.Queue()
        p = mp.Process(target=self.process_frames, args=(queue, ))
        p.start()
        
        keep_running = True
        while keep_running:
            try:
                R, t = queue.get(block=False)
                if R is not None:
                    #TODO:
                    # insert new camera pose here using vis.add_geometry()
                    camera = self.draw_camera(R, t)
                    vis.add_geometry(camera)
                    
            except: 
                pass
            
            keep_running = keep_running and vis.poll_events()
        vis.destroy_window()
        p.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
    parser.add_argument('--method', default='orb', help='method of detecting feature matches')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
