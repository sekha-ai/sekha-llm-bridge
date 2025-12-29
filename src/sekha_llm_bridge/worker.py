from .celery_app import celery_app

# Simple module so you can run:
# celery -A sekha_llm_bridge.worker.celery_app worker --loglevel=info

if __name__ == "__main__":
    celery_app.start()
