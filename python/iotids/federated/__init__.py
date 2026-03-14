# federated/__init__.py
from .partition import iid_partition, non_iid_partition
from .client import FederatedClient
from .server import FedAvgServer