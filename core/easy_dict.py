class EasyDict(dict):
    '''
    sourced from:
        https://github.com/ml4a/ml4a/blob/master/ml4a/utils/__init__.py
    '''
    def __init__(self, *args, **kwargs):
        super(EasyDict, self).__init__(*args, **kwargs)
        self.__dict__ = self