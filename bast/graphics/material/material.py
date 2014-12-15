from __future__ import absolute_import
from ..object import BindableObject
from ...common.object import DescriptorMixin

# TODO: iterate through properties
# if texture, create a sampler
# provide list of texture properties

class Material(DescriptorMixin, BindableObject):
    def __init__(self, program, **properties):
        self._program = program

        self._properties = properties.keys()
        for name, value in properties.items():
            setattr(self, name, value)

    def bind(self):
        # set our local material properties as uniforms
        self.set_uniforms(**self.properties)
        # bind the textures
        # TODO:
        # bind our shader
        self._program.bind()

    def unbind(self):
        # unbind the textures
        # TODO:
        # unbind the shader
        self._program.unbind()

    def set_uniforms(self, **uniforms):
        for name, value in uniforms.items():
            setattr(self._program, name, value)

    @property
    def program(self):
        return self._program

    @property
    def properties(self):
        return dict(
            (name, getattr(self, name))
            for name in self._properties
        )

"""
    @property
    def textures(self):
        # return properties that are textures
        pass
"""