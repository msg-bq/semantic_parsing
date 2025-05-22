import spacy
from spacy.cli import download


def load_spacy_model(model_name: str = "en_core_web_sm"):
    """加载spaCy模型，如果模型不存在则自动下载"""
    try:
        # 尝试加载模型
        model = spacy.load(model_name)
        print(f"成功加载模型: {model_name}")
        return model
    except OSError:
        print(f"未找到模型: {model_name}，开始下载...")
        try:
            # 自动下载模型
            download(model_name)
            # 下载后再次尝试加载
            model = spacy.load(model_name)
            print(f"模型下载并加载成功: {model_name}")
            return model
        except Exception as e:
            raise RuntimeError(f"模型下载失败: {model_name}") from e
