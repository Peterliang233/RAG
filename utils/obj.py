import json

def format_json(data, indent=4):
    try:
        return json.dumps(data, indent=indent)
    except Exception as e:
        return str(data)