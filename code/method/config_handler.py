# Class to handle configuration files for different experiments

import os
import json

class ConfigHandler:
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                value = ConfigHandler(value)
            elif isinstance(value, list) and all(isinstance(i, dict) for i in value):
                value = [ConfigHandler(i) for i in value]
            setattr(self, key, value)

    @staticmethod
    def load(name, search_paths=None):
        """
        Loads a config by name or full path.
        If a relative name is passed, looks in each search_path for a .json file.
        """
        # Use as-is if full path and file exists
        if os.path.isfile(name):
            path = name
        else:
            if not name.endswith(".json"):
                name += ".json"
            # Try to resolve using search_paths
            if search_paths is None:
                search_paths = ["configs"]  # fallback default
            for directory in search_paths:
                candidate = os.path.join(directory, name)
                if os.path.exists(candidate):
                    path = candidate
                    break
            else:
                raise FileNotFoundError(f"Could not find config: {name} in {search_paths}")

        with open(path, "r") as f:
            data = json.load(f)
        return ConfigHandler(data)

    def to_dict(self):
        result = {}
        for key in self.__dict__:
            value = getattr(self, key)
            if isinstance(value, ConfigHandler):
                result[key] = value.to_dict()
            elif isinstance(value, list) and all(isinstance(i, ConfigHandler) for i in value):
                result[key] = [i.to_dict() for i in value]
            else:
                result[key] = value
        return result
