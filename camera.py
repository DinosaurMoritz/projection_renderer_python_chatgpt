#pylint:disable=W0611

import math
import time
import timeit

class Camera:
    def __init__(self, pos=(0,0,0), rot=(0,0,0),res=(1,1), fov=70, near=1, far=100):
        self.pos = pos
        self.dir, self.up = self.get_camera_vectors(*rot)
        #print("dir and up", self.dir, self.up)
        self.world_cam_matrix = self.world_to_camera_matrix(self.pos, self.dir, self.up)
        
        self.res = res
        self.fov = fov
        self.near = near
        self.far = far
        
        self.cot = lambda x: 1/math.tan(x)
    
        self.proj_matrix = [
            [self.cot(self.fov / 2), 0, 0, 0],
            [0, self.cot(self.fov / 2), 0, 0],
            [0, 0, -(self.far + self.near) / (self.far - self.near), -2 * self.far * self.near / (self.far - self.near)],
            [0, 0, 1, 0]
        ]
        
    @staticmethod
    def cross_v3d(a, b):
        return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    
    @staticmethod
    def normalize(v):
        length = math.sqrt(sum([x**2 for x in v]))
        if not length:
            raise Exception("Cant normalise, vector lenth 0.")
        return [x/length for x in v]
    
    @staticmethod
    def mul_matrix(A, B):
        if len(A[0]) != len(B):
            raise ValueError("Cannot multiply matrices with these dimensions")
        
        # Create an empty result matrix
        result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
        
        # Multiply the matrices and sum the products
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    result[i][j] += A[i][k] * B[k][j]
            
        return result
    
    def world_to_camera_matrix(self, pos, dir, up):
        # First, we need to compute the right vector, which is the cross product of the up and dir vectors
        self.right = self.cross_v3d(up, dir)
        
        # Next, we need to compute the camera's local x, y, and z axes by normalizing the right, up, and dir vectors
        x_axis = self.normalize(self.right)
        y_axis = self.normalize(self.up)
        z_axis = self.normalize(self.dir)
    
        # Now we can construct the translation matrix by taking the negation of the pos vector and constructing
        # a 4x4 matrix with it as the translation component
        t = [
            [1, 0, 0, -pos[0]],
            [0, 1, 0, -pos[1]],
            [0, 0, 1, -pos[2]],
            [0, 0, 0, 1]
        ]
    
        # Finally, we can construct the rotation matrix by creating a 4x4 matrix with the x_axis, y_axis, and z_axis
        # vectors as the first, second, and third columns, respectively
        r = [
            [x_axis[0], y_axis[0], z_axis[0], 0],
            [x_axis[1], y_axis[1], z_axis[1], 0],
            [x_axis[2], y_axis[2], z_axis[2], 0],
            [0, 0, 0, 1]
        ]
    
        # The world-to-camera transformation matrix is the product of the translation and rotation matrices
        #print("t       ", t.to_list())
        #print("r       ", r.to_list())
        
        return self.mul_matrix(t, r)
    
    def world_to_cam(self, p):
        return self.multiply_matrix4_vector3(self.world_cam_matrix, p)[:3]
    
    def get_camera_vectors(self, xRotation, yRotation, zRotation):
        def matrix_multiply_extra(matrix, vector):
            result = [0, 0, 0]
            for i in range(3):
                result[i] = matrix[i][0]*vector[0] + matrix[i][1]*vector[1] + matrix[i][2]*vector[2]
            return result
    
        # Convert the rotation values to radians
        xRadians = math.radians(xRotation)
        yRadians = math.radians(yRotation)
        zRadians = math.radians(zRotation)
    
        # Create the rotation matrices
        xRotationMatrix = [
            [1, 0, 0],
            [0, math.cos(xRadians), -math.sin(xRadians)],
            [0, math.sin(xRadians), math.cos(xRadians)]
        ]
          
        yRotationMatrix = [
            [math.cos(yRadians), 0, math.sin(yRadians)],
            [0, 1, 0],
            [-math.sin(yRadians), 0, math.cos(yRadians)]
        ]
            
        zRotationMatrix = [
            [math.cos(zRadians), -math.sin(zRadians), 0],
            [math.sin(zRadians), math.cos(zRadians), 0],
            [0, 0, 1]
        ]
        
        # Calculate the direction vector
        directionVector = [0, 0, 1]
        directionVector = matrix_multiply_extra(zRotationMatrix, directionVector)
        directionVector = matrix_multiply_extra(yRotationMatrix, directionVector)
        directionVector = matrix_multiply_extra(xRotationMatrix, directionVector)
        
        # Calculate the up vector
        upVector = [0, 1, 0]
        upVector = matrix_multiply_extra(zRotationMatrix, upVector)
        upVector = matrix_multiply_extra(yRotationMatrix, upVector)
        upVector = matrix_multiply_extra(xRotationMatrix, upVector)
        
            # Normalize the direction and up vectors
        directionVectorMagnitude = math.sqrt(directionVector[0]**2 + directionVector[1]**2 + directionVector[2]**2)
        upVectorMagnitude = math.sqrt(upVector[0]**2 + upVector[1]**2 + upVector[2]**2)
        directionVector = (directionVector[0]/directionVectorMagnitude, directionVector[1]/directionVectorMagnitude, directionVector[2]/directionVectorMagnitude)
        upVector = (upVector[0]/upVectorMagnitude, upVector[1]/upVectorMagnitude, upVector[2]/upVectorMagnitude)
            
        #directionVector = [(x if x > 0.001 else 0) for x in directionVector]
        #upVector = [(y if y > 0.001 else 0) for y in upVector]
    
        return directionVector, upVector

    @staticmethod
    def multiply_matrix4_vector3(matrix, vector):
        vector += [1]
        result = [0, 0, 0, 0]
        for i in range(4):
            for j in range(4):
                result[i] += matrix[i][j] * vector[j]
        return result
        
    def project(self, p):
        wp = self.world_to_cam(p)
        p25 = self.multiply_matrix4_vector3(self.proj_matrix, wp) #multiplyMatrixVector(self.proj_matrix, wp)
        #print(p25)
        if p25[3] == 0 or p25[3] > self.far or p25[3] < self.near:
            return None
        x, y= p25[0]/p25[2], p25[1]/p25[2]
        
        #print("p2", x, y)
        
        if -1<x<1 and -1<y<1 :
            return (x+1)/2*self.res[0], (y+1)/2*self.res[1]
        return None
     
   
c = Camera([0,0,-1],[0,-10,0], fov= 90)
#print(c.world_to_cam([0,0,10]))

#print (timeit.timeit(setup = mysetup, stmt = code, number = 100000) 



#print(c.multiplyMatrixVector(c.world_cam_matrix, [1,2,3, 1]))
