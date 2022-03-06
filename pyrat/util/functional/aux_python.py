def reg_classmethod(f):
    """
    register the function as class method
    :param f:
    :return:
    """
    return classmethod(f)


def reg_staticmethod(f):
    """
    register the function as static method
    :param f:
    :return:
    """
    return staticmethod(f)


def reg_property(f):
    """
    register the function as property
    :param f:
    :return:
    """
    return property(f)


def deco_import(d, name):
    print("DO SOME THING" + d + name)
    pass


def pre_import(d, name):

    pass
