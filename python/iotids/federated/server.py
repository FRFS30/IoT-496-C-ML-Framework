from ..metrics.classification import accuracy


class FedAvgServer:
    """
    Federated Averaging server.
    Orchestrates round-by-round global model aggregation.
    """

    def __init__(self, global_model, clients):
        self.global_model = global_model
        self.clients      = clients
        self._round_log   = []

    # ------------------------------------------------------------------ #
    # FedAvg aggregation — weighted by each client's dataset size
    # ------------------------------------------------------------------ #
    def aggregate(self, client_weights, client_sizes):
        """
        client_weights: list of weight structures (one per client)
        client_sizes  : list of ints (n_samples per client)
        Returns averaged weight structure.
        """
        total = sum(client_sizes) or 1
        n_clients = len(client_weights)

        # Deep-copy the first client's structure as a template
        avg = _deep_copy_weights(client_weights[0])

        # Zero the template
        _scale_weights(avg, 0.0)

        # Weighted sum
        for weights, size in zip(client_weights, client_sizes):
            w = size / total
            _add_scaled_weights(avg, weights, w)

        return avg

    # ------------------------------------------------------------------ #
    # Full FL training loop
    # ------------------------------------------------------------------ #
    def run(self, num_rounds, local_epochs=1, local_lr=1e-3,
            eval_X=None, eval_y=None, verbose=True):
        """
        num_rounds  : number of FL communication rounds
        eval_X/y    : optional held-out test set for global eval
        """
        for rnd in range(1, num_rounds + 1):
            # Broadcast current global weights to all clients
            global_weights = self.global_model.get_weights()
            for client in self.clients:
                client.set_weights(_deep_copy_weights(global_weights))

            # Local training
            client_weights = []
            client_sizes   = []
            for client in self.clients:
                client.local_train(rounds=local_epochs, lr=local_lr)
                client_weights.append(client.get_weights())
                client_sizes.append(client.n_samples())

            # Aggregate
            new_weights = self.aggregate(client_weights, client_sizes)
            self.global_model.set_weights(new_weights)

            # Round logging
            log = {"round": rnd}
            if eval_X is not None and eval_y is not None:
                metrics = self.global_model.evaluate(eval_X, eval_y)
                log.update(metrics)
                if verbose:
                    print(f"Round {rnd}/{num_rounds} | "
                          f"acc={metrics['accuracy']:.4f} "
                          f"f1={metrics['f1']:.4f} "
                          f"auc={metrics['auc']:.4f}")
            elif verbose:
                print(f"Round {rnd}/{num_rounds} complete")

            self._round_log.append(log)

        return self._round_log

    def get_log(self):
        return self._round_log


# ------------------------------------------------------------------ #
# Weight structure helpers (work on nested list structures)
# ------------------------------------------------------------------ #
def _deep_copy_weights(w):
    if isinstance(w, list):
        return [_deep_copy_weights(v) for v in w]
    if isinstance(w, float):
        return w
    return w


def _scale_weights(w, factor):
    """In-place multiply all leaf floats by factor."""
    if isinstance(w, list):
        for i in range(len(w)):
            if isinstance(w[i], list):
                _scale_weights(w[i], factor)
            else:
                w[i] = float(w[i]) * factor


def _add_scaled_weights(target, source, scale):
    """In-place: target += source * scale."""
    if isinstance(target, list):
        for i in range(len(target)):
            if isinstance(target[i], list):
                _add_scaled_weights(target[i], source[i], scale)
            elif isinstance(target[i], (int, float)):
                target[i] += float(source[i]) * scale
