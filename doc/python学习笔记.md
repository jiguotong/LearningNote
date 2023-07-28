### yield关键字

https://blog.csdn.net/weixin_44726976/article/details/109058763

### 列表推导式

https://blog.csdn.net/weixin_43790276/article/details/90247423

### 进度条使用相关

在列表的外层套一个tqdm（python封装好的进度条类）

```python
from tqdm import tqdm
list = range(0,5)
for i in tqdm(list):
    pass
```

### pycharm使用技巧，包含远程开发

https://blog.csdn.net/weixin_38037405/article/details/120550201

### 使用numpy的where功能可以进行条件处理数组

```python
target = np.array([1,2,3,4,5,6,7,8,9,10])
print(target)

# 对小于4的数组元素赋值0，对大于7的元素赋值255
target[target<4]=0
target[target>7]=255
print(target)
# 对介于[4,7]的元素进行减1再进行平方，非此区间内元素保持不变
target = np.where((target>=4) & (target<=7),(target-1)*2,target)
print(target)
```

### Python中的isinstance()函数

https://blog.csdn.net/qq_36998053/article/details/122682397

### Python函数装饰器的使用

https://blog.csdn.net/qq_45476428/article/details/126962919

### def main() -> None: 的作用是声明该函数返回类型为None，即没有返回值，如果是 -> def main() -> int:则说明返回值是int类型 



# 3.常用工具学习
Hydra官网：https://hydra.cc/docs/1.3/intro/

## 3.0 Python的数据类————@dataclass 基于装饰器 仅限Python3.7及以上
https://blog.csdn.net/be5yond/article/details/119545119
https://zhuanlan.zhihu.com/p/555359585
基本使用
```python
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import List

@dataclass
class Player():
    name: str
    age: int
    description: str='player'
    skills: List[str]= field(default_factory = list)

# 使用
player1 = Player('Justin',24,skills=['eat','sleep','deeplearning','C++'])
print(player1.name)
print(player1)
```
## 3.1 Hydra学习————python中用来配置变量的工具
### 3.1.1 安装
pip install hydra-core
### 3.1.2 了解YAML文件和python函数装饰器的使用
YAML文件：https://blog.csdn.net/xikaifeng/article/details/121612180
函数装饰器：https://blog.csdn.net/qq_45476428/article/details/126962919

### 3.1.3 使用Hydra
main.py
```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="config")
def test(config: DictConfig):
    # To access elements of the config
    print(f"The batch size is {config.deeplearning['batch_size']}")
    print(f"The learning rate is {config.User[0]['name']}")

    # 用OmegaConf.to_yaml组织成yaml的可视化格式
    print(OmegaConf.to_yaml(config))

if __name__ == "__main__":
    test()
```
configs/config.yaml
```yaml
### config/config.yaml
deeplearning:
  batch_size: 10
  lr: 1e-4

User:
-  name: jiguotong
-  age: 24
```
### 3.1.4 进阶使用Hydra-group
可以在yaml文件中进行嵌套其他yaml文件
文件结构
```shell
├── configs
│   ├── config_db.yaml
│   └── db
│       ├── mysql.yaml
│       └── postgresql.yaml
└── main.py
```
main.py
```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="config")
def test_for_db(config: DictConfig):
    print(config.db.driver)
    print(config.db.user)
    print(OmegaConf.to_yaml(config))

if __name__ == "__main__":
    test_for_db()
```

configs/config_db.yaml
```yaml
### config/config_db.yaml
defaults:
  - db: mysql
```

configs/db/mysql.yaml
```yaml
driver: mysql
user: mysql_user
password: 654321
timeout: 20
```

configs/db/postgresql.yaml
```yaml
driver: postgresql
user: postgres_user
password: 123456
timeout: 20
```

### 3.1.5 Hydra搭配数据类@dataclass的使用
test.py
```python
#!/usr/bin/env python
#  -*- encoding: utf-8 -*-

from typing import List
from dataclasses import dataclass
from dataclasses import field

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

# 申明数据类
@dataclass
class Player():
    name: str
    age: int
    description: str='player'
    skills: List[str]= field(default_factory = list)

# 存储示例
cs = ConfigStore.instance()

# 使用类 即config1指向Player类
cs.store(name="config1", node=Player)

# 使用hydra装饰函数
@hydra.main(version_base=None, config_name="config1")
def main_config(cfg: Player) -> None:
    print('main_config1--->')
    print(OmegaConf.to_yaml(cfg))
    pass

# ----------------------------------------------------------------------
# 小结
if __name__ == '__main__':
    main_config()
```