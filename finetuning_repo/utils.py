import logging
from typing_extensions import Literal
from rich.logging import RichHandler
from transformers import StoppingCriteria
import torch

class StopOnTokens(StoppingCriteria):
    def __init__(self,stop_token_ids):
        self.stop_token_ids = stop_token_ids
        super().__init__()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = self.stop_token_ids
        for ids in stop_ids:
            stop_id_len = len(ids)
            input_id_len = len(input_ids[0])
            test = list(input_ids[0][input_id_len-stop_id_len:input_id_len].cpu().numpy())
            if test == ids:
                return True
        return False


def get_logger(name: str, level: Literal["info", "warning", "debug"]) -> logging.Logger:
    rich_handler = RichHandler(level=logging.INFO, rich_tracebacks=True, markup=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging._nameToLevel[level.upper()])

    if not logger.handlers:
        logger.addHandler(rich_handler)

    logger.propagate = False

    return logger