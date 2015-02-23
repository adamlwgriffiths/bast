from __future__ import absolute_import
import os
import re
import numpy as np
from pyrr import Vector3, Quaternion

# TODO: make in_position, etc configurable by setting them on the MD5_Mesh object
# or as properties of the functions

# TODO: move prepare methods into a seperate class that processes the originals

# TODO: normal calculation is going to be a problem if we calculate the mesh on the fly

def _lines(f):
    for line in f.xreadlines():
        line = line.strip()
        if line:
            yield line


class MD5_Mesh(object):
    @classmethod
    def open(cls, filename):
        with open(filename) as f:
            lines = _lines(f)
            mesh = cls()
            mesh.from_buffer(lines)

            # TODO: convert to submeshes, etc
            return mesh

    def from_buffer(self, buffer):
        self.version = None
        self.commandline = None
        self.num_joints = None
        self.num_meshes = None
        self.joints = None
        self.meshes = None

        for line in buffer:
            tokens = line.split()
            command = tokens[0]
            if command == 'MD5Version':
                self.version = int(tokens[1])
                assert self.version == 10
            elif command == 'commandline':
                self.commandline = tokens[1]
            elif command == 'numJoints':
                self.num_joints = int(tokens[1])
            elif command == 'numMeshes':
                self.num_meshes = int(tokens[1])
                self.meshes = []
            elif command == 'joints':
                assert self.num_joints is not None
                self.joints = MD5_Joints()
                self.joints.from_buffer(buffer, self.num_joints)
            elif command == 'mesh':
                assert self.num_meshes is not None
                mesh = MD5_SubMesh()
                mesh.from_buffer(buffer)
                self.meshes.append(mesh)

    def to_mesh(self):
        # convert to our mesh format
        pass


class MD5_Data(object):
    @classmethod
    def compute_quaternion_w(cls, x, y, z):
        """Computes the Quaternion W component from the
        Quaternion X, Y and Z components.
        """
        # extract our W quaternion component
        w = 1.0 - x**2 - y**2 - z**2
        if w < 0.0:
            w = 0.0
        else:
            w = np.sqrt(w)
        return w



class MD5_Joints(MD5_Data):
    def from_buffer(self, buffer, num_joints):
        self.num_joints = num_joints
        self.joints = np.empty((self.num_joints,), dtype=[('name', np.object, 1),('parent', np.uint16, 1),('position', np.float32, 3),('orientation', np.float32, 4)])

        for index, line in enumerate(buffer):
            if line == '}':
                break

            # remove any comments
            # strip any parenthesis
            line = line.split('//')[0]
            line = re.sub('[()]', '', line)

            name, parent, px,py,pz, qx,qy,qz = line.split()

            # remove quotes
            name = name[1:-1]
            parent = int(parent)
            px,py,pz,qx,qy,qz = map(float, (px,py,pz,qx,qy,qz))
            qw = self.compute_quaternion_w(qx, qy, qz)

            self.joints[index]['name'] = name
            self.joints[index]['parent'] = parent
            self.joints[index]['position'][:] = (px,py,pz)
            self.joints[index]['orientation'][:] = (qx,qy,qz,qw)

    def prepare_inverse_bind_pose(self):
        bind_pose = np.empty((self.num_joints,), dtype=[('position', np.float32, 4),('orientation', np.float32, 4)])

        for index in range(self.num_joints):
            joint = self.joints[index]

            bind_pose[index]['position'][:3] = -Vector3(joint['position'])
            bind_pose[index]['position'][3] = 0.
            bind_pose[index]['orientation'][:] = ~Quaternion(joint['orientation'])

        return bind_pose

    def __getitem__(self, index):
        return self.joints[index]


class MD5_SubMesh(MD5_Data):
    def from_buffer(self, buffer):
        self.shader = None

        self.num_verts = None
        self.verts = None
        self.weight_indices = None

        self.num_tris = None
        self.tris = None

        self.num_weights = None
        self.weights = None

        for line in buffer:
            if line == '}':
                break

            tokens = line.split()
            command = tokens[0]

            if command == 'shader':
                value = tokens[1]
                # remove quotes
                self.shader = value[1:-1]

            elif command == 'numverts':
                self.num_verts = int(tokens[1])
                self.verts = np.empty((self.num_verts, 2), dtype=np.float32)
                self.weight_indices = np.empty((self.num_verts, 2), dtype=np.uint16)

            elif command == 'vert':
                # strip any parenthesis
                line = re.sub('[()]', '', line)

                _, index, s, t, weight_index, weight_count = line.split()

                index, weight_index, weight_count = map(int, (index, weight_index, weight_count))
                s, t = map(float, (s, t))

                self.verts[index] = (s, 1.0 - t)
                self.weight_indices[index] = (weight_index, weight_count)

            elif command == 'numtris':
                self.num_tris = int(tokens[1])
                self.tris = np.empty((self.num_tris, 3), dtype=np.uint32)

            elif command == 'tri':
                # strip any parenthesis
                line = re.sub('[()]', '', line)

                _, index, a, b, c = line.split()
                index = int(index)
                a, b, c = map(float, (a, b, c))

                self.tris[index] = (a, b, c)

            elif command == 'numweights':
                self.num_weights = int(tokens[1])
                self.weights = np.empty((self.num_weights,), dtype=[('joint', np.uint16, 1),('bias', np.float32, 1),('position', np.float32, 3)])

            elif command == 'weight':
                # strip any parenthesis
                line = re.sub('[()]', '', line)

                _, index, joint, bias, x, y, z = line.split()
                index, joint = map(int, (index, joint))
                bias, x, y, z = map(float, (bias, x, y, z))

                self.weights[index]['joint'] = joint
                self.weights[index]['bias'] = bias
                self.weights[index]['position'][:] = (x,y,z)

    def prepare_mesh(self, joints):
        vertices = np.empty((self.num_verts,), dtype=[('in_position', np.float32, 3),('in_uv', np.float32, 2),('in_normal', np.float32, 3),('in_weight_indices', np.uint16, 2)])

        # copy over the texture co-ordinates verbatim
        vertices[:]['in_uv'] = self.verts
        vertices[:]['in_weight_indices'] = self.weight_indices

        # convert the vertices to bind pose local space
        # for each weight, convert the weight to be local to the joint
        # then add together based on the bias value
        for index in range(self.num_verts):
            position = Vector3(vertices[index]['in_position'])
            normal = Vector3(vertices[index]['in_normal'])
            position[:] = normal[:] = (0.,0.,0.)

            start_weight, weight_count = self.weight_indices[index]
            end_weight = start_weight + weight_count

            for weight_index in range(start_weight, end_weight):
                weight = self.weights[weight_index]
                joint = joints[weight['joint']]

                rotated_weight = Quaternion(joint['orientation']) * Vector3(weight['position'])

                position += (Vector3(joint['position']) + rotated_weight) * float(weight['bias'])


        # prepare normals
        # accumulate the normals for each face
        for a, b, c in self.tris:
            va = Vector3(vertices[a]['in_position'])
            vb = Vector3(vertices[b]['in_position'])
            vc = Vector3(vertices[c]['in_position'])

            na = Vector3(vertices[a]['in_normal'])
            nb = Vector3(vertices[b]['in_normal'])
            nc = Vector3(vertices[c]['in_normal'])

            normal = (vc - va) ^ (vb - va)

            na += normal
            nb += normal
            nc += normal

        # normalise the normals
        # then convert to bind pose local space
        for index in range(self.num_verts):
            normal = Vector3(vertices[index]['in_normal'])
            normal.normalise()

            start_weight, weight_count = self.weight_indices[index]
            end_weight = start_weight + weight_count

            bind_pose_normal = Vector3()

            for weight_index in range(start_weight, end_weight):
                weight = self.weights[weight_index]
                joint = joints[weight['joint']]

                rotation = Quaternion(joint['orientation'])

                bind_pose_normal += (rotation * normal) * float(weight['bias'])

            normal[:] = bind_pose_normal

        return vertices

    def prepare_weights(self):
        # in_bone is bone index + bias, where bias is 0.001 -> 1.0
        # therefore, to get the bias, take the fractional part of the number.
        # if fraction is 0.0, then it is 1.0
        # this bone index is ALWAYS the whole part of the number, 1.0 does NOT need to be subtracted
        weights = np.empty((self.num_weights,), dtype=[('in_weight_position', np.float32, 3),('in_bone', np.float32, 1)])
        weights[:]['in_weight_position'] = self.weights[:]['position']
        weights[:]['in_bone'] = self.weights[:]['joint']
        weights[:]['in_bone'] += np.fmod(self.weights[:]['bias'], 1.0)

        return weights

    @property
    def shader_filename(self):
        # paths may be in windows format
        shader = self.shader.replace('\\', '/')
        return os.path.split(shader)[1]


class MD5_Anim(MD5_Data):
    @classmethod
    def open(cls, filename):
        with open(filename) as f:
            lines = _lines(f)
            mesh = cls()
            mesh.from_buffer(lines)

            # TODO: convert to submeshes, etc
            return mesh

    def from_buffer(self, buffer):
        self.version = None
        self.commandline = None
        self.num_frames = None
        self.num_joints = None
        self.frame_rate = None
        self.num_components = None
        self.hierarchy = None
        self.bounds = None
        self.base_frame = None
        self.frames = []

        for line in buffer:
            tokens = line.split()
            command = tokens[0]
            if command == 'MD5Version':
                self.version = int(tokens[1])
                assert self.version == 10
            elif command == 'commandline':
                self.commandline = tokens[1]
            elif command == 'numFrames':
                self.num_frames = int(tokens[1])
                self.frames = [None] * self.num_frames
            elif command == 'numJoints':
                self.num_joints = int(tokens[1])
            elif command == 'frameRate':
                self.frame_rate = int(tokens[1])
            elif command == 'numAnimatedComponents':
                self.num_components = int(tokens[1])
            elif command == 'hierarchy':
                assert self.num_joints is not None
                self.hierarchy = MD5_Hierarchy()
                self.hierarchy.from_buffer(buffer, self.num_joints)
            elif command == 'bounds':
                assert self.num_frames is not None
                self.bounds = MD5_Bounds()
                self.bounds.from_buffer(buffer, self.num_frames)
            elif command == 'baseframe':
                assert self.num_joints is not None
                self.base_frame = MD5_BaseFrame()
                self.base_frame.from_buffer(buffer, self.num_joints)
            elif command == 'frame':
                assert self.num_frames is not None
                assert self.num_components is not None
                index = int(tokens[1])
                frame = MD5_Frame()
                frame.from_buffer(buffer, self.num_components)
                self.frames[index] = frame


class MD5_Hierarchy(MD5_Data):
    def from_buffer(self, buffer, num_joints):
        self.num_joints = num_joints
        self.joints = np.empty((self.num_joints,), dtype=[('name', np.object, 1),('parent', np.int16, 1),('flags', np.uint8, 1),('start', np.uint16, 1)])

        for index, line in enumerate(buffer):
            if line == '}':
                break

            # remove any comments
            line = line.split('//')[0]

            tokens = line.split()

            name, parent, flags, start = tokens

            # remove quotes
            name = name[1:-1]
            parent, flags, start = map(int, (parent, flags, start))

            self.joints[index]['name'] = name
            self.joints[index]['parent'] = parent
            self.joints[index]['flags'] = flags
            self.joints[index]['start'] = start

    def __getitem__(self, index):
        return self.joints[index]


class MD5_Bounds(MD5_Data):
    def from_buffer(self, buffer, num_frames):
        self.num_frames = num_frames
        self.bounds = np.empty((self.num_frames,2,3), dtype=np.float32)

        for index, line in enumerate(buffer):
            if line == '}':
                break

            # strip any parenthesis
            line = re.sub('[()]', '', line)

            x1,y1,z1, x2,y2,z2 = line.split()

            self.bounds[index][:] = ((x1,y1,z1), (x2,y2,z2))

    def __getitem__(self, index):
        return self.bounds[index]


class MD5_BaseFrame(MD5_Data):
    def from_buffer(self, buffer, num_joints):
        self.num_joints = num_joints
        self.joints = np.empty((self.num_joints,), dtype=[('position', np.float32, 3),('orientation', np.float32, 4)])

        for index, line in enumerate(buffer):
            if line == '}':
                break

            # remove any comments
            # strip any parenthesis
            line = re.sub('[()]', '', line)

            px,py,pz, qx,qy,qz = line.split()

            # remove quotes
            px,py,pz,qx,qy,qz = map(float, (px,py,pz,qx,qy,qz))
            # this will get recalculated per quaternion later
            #qw = self.compute_quaternion_w(qx, qy, qz)
            qw = 0.0

            self.joints[index]['position'][:] = (px,py,pz)
            self.joints[index]['orientation'][:] = (qx,qy,qz,qw)

    def __getitem__(self, index):
        return self.joints[index]


class MD5_Frame(MD5_Data):
    def from_buffer(self, buffer, num_components):
        self.num_components = num_components
        self.values = np.empty((self.num_components,), dtype=np.float32)

        offset = 0
        for index, line in enumerate(buffer):
            if line == '}':
                break

            values = line.split()
            values = map(float, values)
            self.values[offset:offset + len(values)] = values
            offset += len(values)

    def prepare_frame(self, hierarchy, base_frame):
        joints = np.empty((base_frame.num_joints,), dtype=[('position', np.float32, 4),('orientation', np.float32, 4)])

        # copy the base frame
        joints[:]['position'][:,:3] = base_frame[:]['position']
        joints[:]['position'][:,3] = 1.
        joints[:]['orientation'][:] = base_frame[:]['orientation']

        for index in range(base_frame.num_joints):
            joint = hierarchy[index]

            position = Vector3(joints[index]['position'][:3])
            orientation = Quaternion(joints[index]['orientation'])

            offset = joint['start']
            flags = joint['flags']
            if flags & 1:
                position.x = self.values[offset]
                offset += 1
            if flags & 2:
                position.y = self.values[offset]
                offset += 1
            if flags & 4:
                position.z = self.values[offset]
                offset += 1
            if flags & 8:
                orientation.x = self.values[offset]
                offset += 1
            if flags & 16:
                orientation.y = self.values[offset]
                offset += 1
            if flags & 32:
                orientation.z = self.values[offset]
                offset += 1

            orientation.w = self.compute_quaternion_w(*orientation.xyz)

            # offset by parent
            parent = joint['parent']
            if parent >= 0:
                parent = joints[parent]
                parent_position = Vector3(parent['position'][:3])
                parent_orientation = Quaternion(parent['orientation'])

                position[:] = (parent_orientation * position) + parent_position
                orientation[:] = parent_orientation * orientation
                orientation.normalise()

        return joints

    def __getitem__(self, index):
        return self.joints[index]
