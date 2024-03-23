#!/usr/bin/env python

# Human motion reconstruction
# 2022, Philipp Ruppel

import yaml

# Global variable to hold configuration data as a nested dictionary
config = { }

# Load configuration from YAML files
def load_config(filename):
    global config
    c = yaml.load(open(filename))
    config.update(c)
