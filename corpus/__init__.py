import os
import json
from .common import set_glove
from .common import set_project_root
from .common import set_memory
from sklearn.externals.joblib import Memory

embedding_dim = 100
_project_dir = os.getcwd()
_config = json.load(open(os.path.join(_project_dir,'config.json')))
_glove_path=os.path.join(os.path.dirname(__file__),
                        "glove/w2v.6B.%dd.txt" %(_config.get('embedding_dim',
                                                             embedding_dim)))
mem = Memory("%s/memcache" % _project_dir)
set_memory(mem)
set_glove(_glove_path)
set_project_root(_project_dir)

