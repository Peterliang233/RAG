import yaml
import os
from pydantic import BaseModel
from typing import Optional



class LLM(BaseModel):
    model: str
    api_key: Optional[str]
    embedding_model: str

class Chroma(BaseModel):
    chroma_db_path: str
    
class Database(BaseModel):
    chroma: Chroma
    
class AppConfig(BaseModel):
    database: Database
    llm: LLM
    resource_folder_path: str
    

def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    try:
        with open(config_path, "r") as f:
            dict_conf = yaml.safe_load(f)
            app_config = AppConfig(**dict_conf)
            app_config.llm.api_key = os.environ.get("ALI_API_KEY")
            return app_config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")


app_config = load_config()

# if __name__ == "__main__":
#     app_config = load_config()
#     print(app_config.llm)