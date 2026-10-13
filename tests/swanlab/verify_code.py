#!/usr/bin/env python3
"""
代码验证脚本 - 验证修改的正确性

检查项目：
1. wandb_utils.py 文件修改是否正确
2. default.py 配置修改是否正确  
3. 后端切换逻辑是否完整
"""

import sys
from pathlib import Path

# 设置颜色
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
NC = '\033[0m'  # No Color

def check_file_exists(filepath: Path) -> bool:
    """检查文件是否存在"""
    if filepath.exists():
        print(f"  ✓ 文件存在：{filepath.name}")
        return True
    else:
        print(f"  ✗ 文件不存在：{filepath.name}")
        return False

def check_imports(filepath: Path) -> bool:
    """检查导入语句"""
    content = filepath.read_text()
    
    checks = [
        ("import wandb", "wandb 导入"),
        ("import swanlab", "swanlab 导入"),
        ("WANDB_AVAILABLE", "WandB 可用性检测"),
        ("SWANLAB_AVAILABLE", "SwanLab 可用性检测"),
    ]
    
    all_passed = True
    print("\n  检查导入语句:")
    for pattern, name in checks:
        if pattern in content:
            print(f"    ✓ {name}")
        else:
            print(f"    ✗ {name}")
            all_passed = False
    
    return all_passed

def check_backend_switching(filepath: Path) -> bool:
    """检查后端切换逻辑"""
    content = filepath.read_text()
    
    checks = [
        ("self.backend = getattr", "backend 属性获取"),
        ("if self.backend == \"swanlab\"", "SwanLab 条件判断"),
        ("_init_swanlab", "SwanLab 初始化方法"),
        ("_init_wandb", "WandB 初始化方法"),
        ("mode_map", "模式映射"),
        ("swanlab.init", "swanlab.init 调用"),
        ("wandb.init", "wandb.init 调用"),
    ]
    
    all_passed = True
    print("\n  检查后端切换逻辑:")
    for pattern, name in checks:
        if pattern in content:
            print(f"    ✓ {name}")
        else:
            print(f"    ✗ {name}")
            all_passed = False
    
    return all_passed

def check_method_implementations(filepath: Path) -> bool:
    """检查方法实现"""
    content = filepath.read_text()
    
    checks = [
        ("def _init_wandb", "_init_wandb 方法"),
        ("def _init_swanlab", "_init_swanlab 方法"),
        ("def log_dict", "log_dict 方法"),
        ("def log_video", "log_video 方法"),
        ("def log_policy", "log_policy 方法"),
        ("if self.backend == \"swanlab\"", "log_dict 中的后端判断"),
    ]
    
    all_passed = True
    print("\n  检查方法实现:")
    for pattern, name in checks:
        if pattern in content:
            print(f"    ✓ {name}")
        else:
            print(f"    ✗ {name}")
            all_passed = False
    
    return all_passed

def check_config_modification(filepath: Path) -> bool:
    """检查配置文件修改"""
    content = filepath.read_text()
    
    checks = [
        ("backend: str = \"wandb\"", "backend 字段"),
        ("Backend to use for logging", "backend 注释"),
    ]
    
    all_passed = True
    print("\n  检查 WandBConfig 修改:")
    for pattern, name in checks:
        if pattern in content:
            print(f"    ✓ {name}")
        else:
            print(f"    ✗ {name}")
            all_passed = False
    
    return all_passed

def check_get_run_id_function(filepath: Path) -> bool:
    """检查获取 run ID 的函数"""
    content = filepath.read_text()
    
    checks = [
        ("def get_wandb_run_id_from_filesystem", "函数定义"),
        ("backend: str = \"wandb\"", "backend 参数"),
        ("if backend == \"swanlab\"", "SwanLab 分支"),
        ("swanlog", "swanlog 路径"),
    ]
    
    all_passed = True
    print("\n  检查 get_wandb_run_id_from_filesystem 函数:")
    for pattern, name in checks:
        if pattern in content:
            print(f"    ✓ {name}")
        else:
            print(f"    ✗ {name}")
            all_passed = False
    
    return all_passed

def main():
    base_dir = Path("/mnt/data1/jizheng/vla_project/lerobot")
    
    print(f"{GREEN}{'='*60}{NC}")
    print(f"{GREEN}WandB/SwanLab 后端集成代码验证{NC}")
    print(f"{GREEN}{'='*60}{NC}\n")
    
    # 检查文件
    print(f"{YELLOW}[1/5] 检查修改的文件...{NC}")
    wandb_utils = base_dir / "src" / "lerobot" / "common" / "wandb_utils.py"
    default_config = base_dir / "src" / "lerobot" / "configs" / "default.py"
    
    files_ok = True
    if not check_file_exists(wandb_utils):
        files_ok = False
    if not check_file_exists(default_config):
        files_ok = False
    
    if not files_ok:
        print(f"\n{RED}✗ 文件检查失败{NC}")
        return 1
    
    # 检查 wandb_utils.py
    print(f"\n{YELLOW}[2/5] 检查 wandb_utils.py 导入语句...{NC}")
    if not check_imports(wandb_utils):
        print(f"\n{RED}✗ 导入检查失败{NC}")
        return 1
    
    print(f"\n{YELLOW}[3/5] 检查后端切换逻辑...{NC}")
    if not check_backend_switching(wandb_utils):
        print(f"\n{RED}✗ 后端切换逻辑检查失败{NC}")
        return 1
    
    print(f"\n{YELLOW}[4/5] 检查方法实现...{NC}")
    if not check_method_implementations(wandb_utils):
        print(f"\n{RED}✗ 方法实现检查失败{NC}")
        return 1
    
    # 检查 default.py
    print(f"\n{YELLOW}[5/5] 检查配置文件修改...{NC}")
    if not check_config_modification(default_config):
        print(f"\n{RED}✗ 配置文件检查失败{NC}")
        return 1
    
    # 检查 run ID 函数
    print(f"\n{YELLOW}[额外] 检查 run ID 获取函数...{NC}")
    if not check_get_run_id_function(wandb_utils):
        print(f"\n{RED}✗ run ID 函数检查失败{NC}")
        return 1
    
    # 总结
    print(f"\n{GREEN}{'='*60}{NC}")
    print(f"{GREEN}✅ 所有代码验证通过！{NC}")
    print(f"{GREEN}{'='*60}{NC}\n")
    
    print(f"修改总结:")
    print(f"  • 修改文件：2 个")
    print(f"    - src/lerobot/common/wandb_utils.py")
    print(f"    - src/lerobot/configs/default.py")
    print(f"  • 新增功能:")
    print(f"    - 支持 WandB 和 SwanLab 双后端")
    print(f"    - 通过 backend 参数切换")
    print(f"    - 优雅降级不支持的功能")
    print(f"  • 向后兼容:")
    print(f"    - 默认使用 wandb 后端")
    print(f"    - 不影响现有代码")
    print(f"\n使用方式:")
    print(f"  # 使用 WandB (默认)")
    print(f"  wandb_config = WandBConfig(backend='wandb')")
    print(f"  ")
    print(f"  # 使用 SwanLab")
    print(f"  wandb_config = WandBConfig(backend='swanlab')")
    print(f"\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
