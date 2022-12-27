import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class ObjLoader:
    def __init__(self, file, resize = False):
        logging.info("Loading .Obj - Start")
        #self.vertex_indecies = {}
        self.polygons = []
        self.verticies = []
        
        with open(file, "r") as f:
            for line in f.readlines():
                if line.startswith("v "):
                    vertex = [float(x) for x in line[2:].strip().split(' ')]
                   # self.vertex_indecies[i] = vertex
                    self.verticies.append(vertex)
                
                if line.startswith("f "):
                    face = [int(x.split('/')[0])-1 for x in line[2:].strip().split(' ')]
                    #face = [[float(x) for x in f.readline(i)[2:].strip().split(' ')] for i in face]
                    face = [self.verticies[n] for n in face]
                    self.polygons.append(face)
        
        
        if resize:
            farthest = 0
            for v in self.verticies:
                if max(v) > farthest:
                    farthest = max(v)
        
            for a, f in enumerate(self.polygons):
                for b, v in enumerate(f):
                    for c, _ in enumerate(v):
                        self.polygons[a][b][c] = self.polygons[a][b][c] / farthest * resize
        
        logging.info("Loading .Obj - Done")
        
        return
        # Remove duplicates
        unique_polygons = set()

        for polygon in self.polygons:
            polygon_tuple = tuple(polygon)
            
            if polygon_tuple not in unique_polygons:
                unique_polygons.add(polygon_tuple)
        self.polygons = list(unique_polygons)

                    
        
        
    
           
           
                   
                     
#obj = ObjLoader("benz.obj")
#print(obj.polygons[:10])
#print(obj.polygons[:30])
#print(obj.vertex_indecies.keys())
