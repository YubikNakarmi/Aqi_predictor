"""scripts pacakages for running the ingestion and training scripts"""

__all__ = ["ingestion_run","train_hourly"]



''' exceptoin for script missing'''
def _missing(name, pkg="scripts"):
    def _raiser(*args, **kwargs):
        raise RuntimeError(
            f"'{pkg}.{name}.run' is not available. "
            "Ensure the script exists and exposes `def run(...):` (and that top-level "
            "execution is guarded by `if __name__ == '__main__':`)."
        )
    return _raiser

#for ingestion
try:
    from .ingestion_hourly import run as ingestion_run
except Exception:
    ingestion_run = _missing("ingestion_hourly")

#for hourly_train

# try:
#     from .train_hourly import
