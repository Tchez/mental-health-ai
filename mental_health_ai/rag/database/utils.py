import json
import os
from typing import List


def read_json_in_nested_path(root_path: str) -> List[List[dict]]:
    """Read all JSON files in a nested path.

    Args:
        root_path (str): Path to the root directory where the documents are located.

    Returns:
        List[List[dict]]: List of documents in JSON format.
    """  # noqa: E501

    all_json_docs: List[List[dict]] = []

    try:  # noqa: PLR1702
        for root, _, files in os.walk(root_path):
            print(f'Files founded in {root}: {files}')
            for file in files:
                if file.lower().endswith('.json'):
                    file_path = os.path.join(root, file)

                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = json.load(f)
                        if isinstance(file_content, list):
                            all_json_docs.append(file_content)
                        elif isinstance(file_content, dict):
                            all_json_docs.append([file_content])
                        else:
                            print(f'Invalid JSON file: {file_path}')

        return all_json_docs

    except Exception as e:
        print(f'Failed to load documents: {e}')
        return []
