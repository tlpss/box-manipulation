from abc import ABC, abstractmethod


class KeypointedObject(ABC):
    """Base class for custom object with keypoints.
    Trying this for now but not sure if it's worth it.
    """

    def __init__(self):
        pass

    @property
    @abstractmethod
    def keypoints(self):
        pass
