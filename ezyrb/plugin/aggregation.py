from .plugin import Plugin
import numpy as np
from ..approximation.rbf import RBF
from ..approximation.linear import Linear
from ..database import Database
import copy


class Aggregation(Plugin):

    def __init__(self, fit_function=None, predict_function=Linear()):
        super().__init__()
        self.fit_function = fit_function
        self.predict_function = predict_function

    def validation_weights(self, mrom, normalized=True):
        if self.fit_function is None:
            # proceed with standard space-dependent aggregation methods
            validation_predicted = dict()
            for name, rom in mrom.roms.items():
                validation_predicted[name] = rom.predict(rom.validation_full_database.parameters_matrix)

            g = {}
            sigma = 0.05
            for k, v in validation_predicted.items():
                snap = rom.validation_full_database.snapshots_matrix
                g[k] = np.exp(- ((v - snap)**2)/(2 * (sigma**2)))

            g_tensor = np.array([g[k] for k in g.keys()])

        elif self.fit_function is not None:
            g_tensor = self.fit_function(mrom) # can be a neural network for example

        else:
            raise NotImplementedError

        if normalized:
            g_tensor /= np.sum(g_tensor, axis=0)
        return g_tensor


    def fit_postprocessing(self, mrom):
        g_tensor = self.validation_weights(mrom, normalized=False)

        rom = list(mrom.roms.values())[0]

        # concatenate params and space
        space = rom.validation_full_database._pairs[0][1].space
        params = rom.validation_full_database.parameters_matrix

        input_ = np.hstack([
            np.tile(space, (params.shape[0], 1)),
            np.repeat(params, space.shape[0], axis=0)
        ])

        # Fit the regression/interpolation that will be used to predict the
        # weights in the test database
        self.predict_functions = []
        for i, rom in enumerate(mrom.roms.values()):
            g_ = g_tensor[i, ...].reshape(space.shape[0] * params.shape[0], -1)

            rom_func = copy.deepcopy(self.predict_function)
            try:
                rom_func.fit(input_, g_)
            except MemoryError:
                rom_func.fit(input_[::20, :], g_[::20, :])
            self.predict_functions.append(rom_func)

        mrom.weights_validation = {}
        for i, rom in enumerate(mrom.roms):
            mrom.weights_validation[rom] = g_tensor[i, ...]/np.sum(
                    g_tensor, axis=0)


    def predict_postprocessing(self, mrom):

        space = list(mrom.roms.values())[0].validation_full_database._pairs[0][1].space
        predict_weights = {}
        db = list(mrom.multi_predict_database.values())[0]
        input_ = np.hstack([
            np.tile(space, (db.parameters_matrix.shape[0], 1)),
            np.repeat(db.parameters_matrix, space.shape[0], axis=0)
        ])

        # predict weights with regression and normalize them
        gaussians_test = []
        for i, rom in enumerate(mrom.roms.values()):
            g_ = self.predict_functions[i].predict(input_)
            gaussians_test.append(g_)
        gaussians_test = np.array(gaussians_test)
        gaussians_test = gaussians_test.reshape(len(mrom.roms),
                db.parameters_matrix.shape[0], space.shape[0])

        predict_weights = gaussians_test/np.sum(gaussians_test, axis=0)
        predicted_solution = np.zeros((db.parameters_matrix.shape[0],
            db.snapshots_matrix.shape[1]))
        mrom.weights_predict = {}
        for i, rom in enumerate(mrom.roms):
            mrom.weights_predict[rom] = predict_weights[i, ...]
            db = mrom.multi_predict_database[rom]
            predicted_solution += db.snapshots_matrix * mrom.weights_predict[rom]

        mrom.predict_reduced_database = Database(
                mrom.predict_full_database.parameters_matrix,
                predicted_solution)

