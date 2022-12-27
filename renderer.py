import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
from objLoader import ObjLoader
from camera import Camera


class Renderer:
    def __init__(self):
        self.camera = Camera(pos=[0,0,-100], fov=90, far = 1000)
        self.obj = None
        self.projected = None
    
    def project_poly(self, p):
        p2 = [self.camera.project(v) for v in p]
        if p2.count(None):
            return None
        return p2
    
    def project_model(self, filename = "Low-Poly-Racing-Car.obj", resize = False):
        self.obj = ObjLoader(filename, resize=resize)
        self.projected = [self.project_poly(p) for p in self.obj.polygons]
        return self.projected
        
r = Renderer()
print(r.project_model("Low-Poly-Racing-Car.obj", 100)[:10])
