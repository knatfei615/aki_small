#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
内存优化验证脚本
检查所有优化措施是否正确实施
"""

import os
import sys

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"[OK] {description}: {filepath}")
        return True
    else:
        print(f"[FAIL] {description}: {filepath} not found")
        return False

def check_file_content(filepath, patterns, description):
    """检查文件是否包含特定内容"""
    if not os.path.exists(filepath):
        print(f"[FAIL] {description}: file not found")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    all_found = True
    for pattern in patterns:
        if pattern in content:
            print(f"  [OK] Found: {pattern[:50]}...")
        else:
            print(f"  [FAIL] Missing: {pattern[:50]}...")
            all_found = False
    
    if all_found:
        print(f"[OK] {description}")
    else:
        print(f"[WARN] {description}: some checks failed")
    
    return all_found

def main():
    print("=" * 60)
    print("Streamlit Memory Optimization Verification")
    print("=" * 60)
    print()
    
    checks_passed = 0
    checks_total = 0
    
    # 1. 检查文件存在性
    print("1. Check file existence")
    print("-" * 60)
    
    files = [
        ("app.py", "Main application"),
        (".streamlit/config.toml", "Streamlit config"),
        ("requirements.txt", "Dependencies"),
        ("README_CN.md", "Chinese README"),
        ("MEMORY_OPTIMIZATION.md", "Optimization docs"),
        ("DEPLOYMENT_GUIDE.md", "Deployment guide"),
        ("CHECKLIST.md", "Checklist"),
    ]
    
    for filepath, desc in files:
        checks_total += 1
        if check_file_exists(filepath, desc):
            checks_passed += 1
    
    print()
    
    # 2. 检查 app.py 优化
    print("2. Check app.py optimizations")
    print("-" * 60)
    
    app_patterns = [
        "@st.cache_data(ttl=3600, max_entries=1)",
        "import gc",
        "def cleanup_memory():",
        "plt.close(fig)",
        "del fig",
        "n_jobs=2",
        "max_samples=0.8",
        "y_val.tolist()",
    ]
    
    checks_total += 1
    if check_file_content("app.py", app_patterns, "app.py optimization"):
        checks_passed += 1
    
    print()
    
    # 3. 检查配置文件
    print("3. Check Streamlit config")
    print("-" * 60)
    
    config_patterns = [
        "[server]",
        "maxUploadSize",
        "[runner]",
        "fastReruns",
        "[logger]",
    ]
    
    checks_total += 1
    if check_file_content(".streamlit/config.toml", config_patterns, "config.toml"):
        checks_passed += 1
    
    print()
    
    # 4. 检查 requirements.txt
    print("4. Check dependencies")
    print("-" * 60)
    
    req_patterns = [
        "streamlit==",
        "pandas==",
        "numpy==",
        "scikit-learn==",
        "matplotlib==",
        "seaborn==",
    ]
    
    checks_total += 1
    if check_file_content("requirements.txt", req_patterns, "requirements.txt"):
        checks_passed += 1
    
    print()
    
    # 5. 统计图表清理
    print("5. Check plot cleanup")
    print("-" * 60)
    
    if os.path.exists("app.py"):
        with open("app.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        st_pyplot_count = content.count("st.pyplot(fig)")
        plt_close_count = content.count("plt.close(fig)")
        del_fig_count = content.count("del fig")
        
        print(f"  st.pyplot(fig) calls: {st_pyplot_count}")
        print(f"  plt.close(fig) calls: {plt_close_count}")
        print(f"  del fig calls: {del_fig_count}")
        
        checks_total += 1
        if st_pyplot_count == plt_close_count and del_fig_count >= st_pyplot_count * 0.9:
            print(f"[OK] Plot cleanup complete")
            checks_passed += 1
        else:
            print(f"[WARN] Plot cleanup may be incomplete")
    
    print()
    
    # 最终统计
    print("=" * 60)
    print(f"Verification complete: {checks_passed}/{checks_total} checks passed")
    print("=" * 60)
    
    if checks_passed == checks_total:
        print("[SUCCESS] All checks passed! Ready to deploy!")
        return 0
    elif checks_passed >= checks_total * 0.8:
        print("[OK] Most checks passed. Application is optimized.")
        return 0
    else:
        print("[WARN] Some checks failed. Please review.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
