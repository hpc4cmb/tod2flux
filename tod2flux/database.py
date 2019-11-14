import os
import sqlite3
import sys


class Database:
    """ Database class to store detector fits and combined fluxes
    """

    def __init__(self, filename):
        self.filename = filename
        if os.path.isfile(filename):
            self.data = self.load(filename)
        return

    def enter(self, fit):
        pass

    def save(self):
        pass
