import yaml
import pickle
from pathlib import Path
from typing import Any
from src.file_collector import get_articles

NODE_SIZE_MAP = {
    0: 5,
    1: 10,
    2: 15,
    3: 20,
    4: 25,
    5: 30,
    6: 35,
}

NODE_TYPE_COLORS = {
    "article": "#C17817",
    "author": "#3F84E5",
    "keyword": "#B20D30",
}

class FileParser:
    def __init__(self):
        pass

    @staticmethod
    def parse_config(file_name: str) -> dict[str, Any]:
        """ parse configuration file """
        with open(file_name, "r") as f:
            contents = yaml.safe_load(f)

        # convert to dictionary
        contents = dict(contents)

        # edit variables
        contents["output_directory"]  = Path(contents["output_directory"]).expanduser()

        # ensure that output directory exists
        contents["output_directory"].mkdir(exist_ok=True, parents=False)

        return contents

    @staticmethod
    def get_data(
            base_url: str,
            tool: str,
            email: str,
            query: str,
            batch_size: int,
            update_flag: bool,
            out_dir: Path,
    ) -> list[dict[str, object]]:
        # check if data/ exists under output directory
        data_dir = out_dir / "data/"
        data_dir.mkdir(exist_ok=True, parents=False)

        # collect articles from PubMed
        data_file = data_dir / "data.pkl"
        if update_flag or (not data_file.exists()):
            articles = get_articles(
                base_url=base_url,
                tool_name=tool,
                email=email,
                query=query,
                batch_size=batch_size,
            )
            # save data dictionary
            with open(data_file, "wb") as f:
                pickle.dump(articles, f)
        else:
            with open(data_file, "rb") as f:
                articles = pickle.load(f)

        return articles
