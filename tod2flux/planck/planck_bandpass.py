
from .. import Bandpass


class PlanckBandpass(Bandpass):
    
    def __init__(self, detector_name):
        self.detector_name = detector_name
