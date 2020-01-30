import os
import pickle
import sys


class Database:
    """ Database class to store detector fits and combined fluxes
    """

    def __init__(self, filename="fluxes.pck"):
        self.filename = filename
        self.targets = {}
        self.modes = {}
        self.load()
        return

    def __getitem__(self, dataset):
        return self.db[dataset]

    def __setitem__(self, dataset, fits):
        self.db[dataset] = fits

    def __delitem__(self, dataset):
        del self.db[dataset]

    def __contains__(self, dataset):
        return os.path.basename(dataset) in self.db

    def __len__(self):
        return len(self.db)

    def enter(self, fits):
        """ Write the fits into the database.

        For the moment, the database is a dictionary of Fit objects
        but we may decide on using an sqlite3 database or something
        more advanced in the future.
        """
        target = fits[0].target
        dataset = fits[0].dataset
        self.db[dataset] = fits
        if target not in self.targets:
            self.targets[target] = []
        self.targets[target].append(self.db[dataset])
        return

    def save(self):
        with open(self.filename, "wb") as fout:
            pickle.dump(self.db, fout)
        print("Wrote database to {}".format(self.filename), flush=True)
        return

    def load(self):
        if os.path.isfile(self.filename):
            with open(self.filename, "rb") as fin:
                self.db = pickle.load(fin)
            # Create a cross-reference table
            for dataset, fits in self.db.items():
                target = fits[0].target
                if target not in self.targets:
                    self.targets[target] = []
                self.targets[target].append(fits)
            print("Loaded database from {}".format(self.filename), flush=True)
        else:
            self.db = {}
            print("Initialized a new database", flush=True)
        return
