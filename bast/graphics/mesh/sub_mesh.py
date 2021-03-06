from __future__ import absolute_import
from OpenGL import GL
from ...common.object import DescriptorMixin
from ..buffer.vertex_array import VertexArray
from ..buffer.buffer_pointer import BufferPointer


class SubMesh(DescriptorMixin):
    def __init__(self, material, indices=None, primitive=GL.GL_TRIANGLES, **pointers):
        self._pointers = pointers
        self._material = material
        self.primitive = primitive
        self.indices = indices

        for pointer in pointers.values():
            if not isinstance(pointer, BufferPointer):
                raise ValueError('Must be of type BufferPointer')

        self._vertex_array = VertexArray()
        self._bind_pointers()

    def _bind_pointers(self):
        # TODO: make this more efficient, don't just clear all pointers
        self._vertex_array.clear()

        # assign our pointers to the vertex array
        for name, pointer in self._pointers.items():
            if not isinstance(pointer, BufferPointer):
                raise ValueError('Must be a buffer pointer')

            attribute = self._material.program.attributes.get(name)
            if attribute:
                self._vertex_array[attribute.location] = pointer

    def render(self, **uniforms):
        # set our uniforms
        self._material.set_uniforms(**uniforms)

        # render
        with self._material:
            if self.indices is not None:
                self._vertex_array.render_indices(self.indices, self.primitive)
            else:
                self._vertex_array.render(self.primitive)

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, material):
        self._material = material
        self._bind_pointers()
