
from .. import Beam


class PlanckBeam(Beam):

    def __init__(self, detector_name):
        self.detector_name = detector_name
