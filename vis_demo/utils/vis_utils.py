import open3d as o3d
import os


class vis_pointcloud:
    def __init__(self, use_vis, online_vis=False):
        self.use_vis=use_vis
        if self.use_vis==0:
            return
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="scene",width=1269,height=778,left=50)
        render_option=self.vis.get_render_option()
        render_option.point_size=2.0
        #self.ctr = self.vis.get_view_control()
        if os.path.exists('temp.json'):
            self.param = o3d.io.read_pinhole_camera_parameters('temp.json')
            self.view = True
        else:
            self.view = False
        #self.ctr.convert_from_pinhole_camera_parameters(self.param) 

    def update(self,points,points_color):
        if self.use_vis==0:
            return
        if self.online_vis:
            self.vis.clear_geometries()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors =  o3d.utility.Vector3dVector(points_color/255)
        self.vis.add_geometry(pcd)
        if self.view:
            ctr = self.vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(self.param)
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def run(self):
        if self.use_vis==0:
            return
        self.vis.run()


class Vis_color:
    def __init__(self,use_vis):
        self.use_vis=use_vis
        if use_vis==0:
            return
        self.vis_image = o3d.visualization.Visualizer()
        self.vis_image.create_window(window_name="scene",width=320,height=240,left=50)

    def update(self,color_image):
        if self.use_vis==0:
            return
        geometry_image=o3d.geometry.Image(color_image)
        self.vis_image.add_geometry(geometry_image)
        self.vis_image.poll_events()
        self.vis_image.update_renderer()
        geometry_image.clear()
        
