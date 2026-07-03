#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the netallocation package.

It package provides various functions to allocate flow in a pypsa power
system. Underlying packages are xarray and dask.

"""

from . import (
    breakdown,
    common,
    cost,
    evaluate,
    flow,
    grid,
    io,
    linalg,
    plot,
    plot_helpers,
    process,
    test,
    utils,
)
from .cost import allocate_cost, allocate_revenue
from .flow import flow_allocation as allocate_flow
from .grid import (
    Incidence,
    network_flow,
    network_injection,
    power_demand,
    power_production,
)
from .io import load_dataset, store_dataset
from .linalg import diag, inv, pinv
from .utils import as_dense, as_sparse

__version__ = "0.0.8"
__author__ = "Fabian Hofmann (FIAS)"
__copyright__ = "Copyright 2015-2020 Fabian Hofmann (FIAS), GNU GPL 3"
