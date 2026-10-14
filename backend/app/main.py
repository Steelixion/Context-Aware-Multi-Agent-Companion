import multiprocessing as mp

import uvicorn

from .api.server import create_app
from .config import Config


def main() -> None:
    cfg = Config()
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    mp.freeze_support()
    main()
