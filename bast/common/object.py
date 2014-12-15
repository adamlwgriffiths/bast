
class DescriptorMixin(object):
    """Mixin to enable runtime-added descriptors."""
    def __getattribute__(self, name):
        attr = super(DescriptorMixin, self).__getattribute__(name)
        if hasattr(attr, "__get__") and not callable(attr):
            return attr.__get__(self, self.__class__)
        else:
            return attr

    def __setattr__(self, name, value):
        try:
            attr = super(DescriptorMixin, self).__getattribute__(name)
            return attr.__set__(self, value)
        except AttributeError:
            return super(DescriptorMixin, self).__setattr__(name, value)
