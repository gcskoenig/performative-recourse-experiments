import json
from pathlib import Path

def convert_params_file(file_path: Path):
    params = {}
    with file_path.open() as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('//'):
                continue
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                # Convert numeric values if possible
                if value.lower() in ['true', 'false']:
                    params[key] = value.lower() == 'true'
                else:
                    try:
                        # Handle floats and ints
                        if '.' in value:
                            params[key] = float(value)
                        else:
                            params[key] = int(value)
                    except ValueError:
                        params[key] = value
    # Write JSON file in the same folder
    json_file = file_path.parent / "params.json"
    with json_file.open("w") as f:
        json.dump(params, f, indent=4)
    print(f"Converted {file_path} to {json_file}")

if __name__ == "__main__":
    # Adjust the root directory if needed
    root_folder = Path("results")
    for params_file in root_folder.rglob("params.txt"):
        convert_params_file(params_file)