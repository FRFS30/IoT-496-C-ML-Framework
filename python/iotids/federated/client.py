from ..nn.losses import BinaryCrossentropy
from ..nn.optimizers import Adam
from ..metrics.classification import accuracy


class FederatedClient:
    """
    Simulates one IoT device in the federated network.
    Maintains local model, trains on local data, exposes weights for FedAvg.
    """

    def __init__(self, client_id, model, local_data):
        """
        client_id : str or int
        model     : Sequential (or RandomForestClassifier)
        local_data: (X, y) tuple
        """
        self.client_id  = client_id
        self.model      = model
        self.X, self.y  = local_data
        self._round_log = []   # per-round local metric history

    # ------------------------------------------------------------------ #
    # Local training
    # ------------------------------------------------------------------ #
    def local_train(self, rounds=1, lr=1e-3, batch_size=64,
                    loss=None, verbose=False):
        """
        Train for `rounds` local epochs on private data.
        Returns final local training accuracy.
        """
        if loss is None:
            loss = BinaryCrossentropy(from_logits=False)

        opt = Adam(lr=lr)
        hist = self.model.fit(
            self.X, self.y,
            epochs=rounds,
            batch_size=batch_size,
            validation_split=0.1,
            optimizer=opt,
            loss=loss,
            verbose=verbose,
        )
        local_acc = hist["val_acc"][-1] if hist["val_acc"] else 0.0
        self._round_log.append({"local_acc": local_acc, "loss": hist["loss"][-1]})
        return local_acc

    # ------------------------------------------------------------------ #
    # Weight synchronisation — used by FedAvgServer
    # ------------------------------------------------------------------ #
    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def n_samples(self):
        return len(self.y)

    def get_log(self):
        return self._round_log
