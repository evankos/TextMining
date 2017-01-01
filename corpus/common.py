
_GLOVE_PATH = ''
_PROJECT_ROOT = ''
_MEMORY = None
_EMBEDDING_DIM = 50

def set_glove(glove_path):
    """
    Set path to glove files
    :param glove_path:
    """
    global _GLOVE_PATH
    _GLOVE_PATH = glove_path

def glove():
    """
    Get glove path
    :return:
    """
    return _GLOVE_PATH

def set_project_root(project_root):
    global _PROJECT_ROOT
    _PROJECT_ROOT = project_root

def project_root():
    return _PROJECT_ROOT

def set_memory(mem):
    global _MEMORY
    _MEMORY = mem

def memory():
    return _MEMORY

def set_embedding_dim(embedding_dim):
    global _EMBEDDING_DIM
    _EMBEDDING_DIM = embedding_dim

def embedding_dim():
    return _EMBEDDING_DIM