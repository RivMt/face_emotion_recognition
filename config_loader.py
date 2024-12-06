import json

config = json.load(open("config.json"))


def get(key: str, default):
    if key in config:
        return config[key]
    return default


def get_host_address():
    return get("host_address", "localhost")


def get_target_address():
    return get("target_address", "localhost")


def get_port():
    return get("port", 8080)


def get_buffer_size():
    return get("buffer_size", 1024)


def get_timeout():
    return get("timeout", 1)

