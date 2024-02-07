import os
from cog import BasePredictor, Input, Path
from pyngrok import ngrok, conf

class Predictor(BasePredictor):
    def predict(
        self,
        token: str = Input()
    ) -> str:
        conf.get_default().auth_token = token
        public_url = ngrok.connect(7860).public_url
        print(public_url)
        os.system(f"jupyter notebook --allow-root --port 7860 --ip 0.0.0.0 --NotebookApp.token '' --no-browse --notebook-dir /content")
        return public_url