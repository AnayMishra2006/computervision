import threading
from app.main import main as cv_main
import uvicorn

def start_cv():
    cv_main()

if __name__ == "__main__":
    threading.Thread(target=start_cv, daemon=True).start()
    uvicorn.run("app.api.server:app", host="0.0.0.0", port=8000, reload=False)