#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Git版本控制教程脚本
"""

def print_git_tutorial():
    """打印Git教程"""
    print("Git版本控制教程")
    print("="*30)
    
    print("\n1. 初始化Git仓库:")
    print("   git init")
    
    print("\n2. 添加文件到暂存区:")
    print("   git add <文件名>")
    print("   git add .  # 添加所有文件")
    
    print("\n3. 提交更改:")
    print("   git commit -m \"提交信息\"")
    
    print("\n4. 查看状态:")
    print("   git status")
    
    print("\n5. 查看提交历史:")
    print("   git log")
    print("   git log --oneline  # 简洁格式")
    
    print("\n6. 查看差异:")
    print("   git diff  # 工作区与暂存区的差异")
    print("   git diff --cached  # 暂存区与最后一次提交的差异")
    
    print("\n7. 撤销操作:")
    print("   git checkout -- <文件名>  # 撤销工作区的更改")
    print("   git reset HEAD <文件名>  # 撤销暂存区的更改")
    print("   git reset --hard HEAD^  # 撤销最后一次提交")
    
    print("\n8. 分支操作:")
    print("   git branch  # 查看分支")
    print("   git branch <分支名>  # 创建分支")
    print("   git checkout <分支名>  # 切换分支")
    print("   git merge <分支名>  # 合并分支")
    
    print("\n9. 远程仓库:")
    print("   git remote add origin <仓库地址>  # 添加远程仓库")
    print("   git push -u origin master  # 推送到远程仓库")
    print("   git pull origin master  # 从远程仓库拉取")
    
    print("\n10. 标签:")
    print("    git tag <标签名>  # 创建标签")
    print("    git push --tags  # 推送标签到远程仓库")

def print_best_practices():
    """打印Git最佳实践"""
    print("\n\nGit最佳实践")
    print("="*30)
    
    print("\n1. 提交信息规范:")
    print("   - 使用祈使句，如'Add feature'而不是'Added feature'")
    print("   - 首字母小写")
    print("   - 不以句号结尾")
    print("   - 第一行简短描述，空一行后详细说明")
    
    print("\n2. 分支管理:")
    print("   - 主分支保持稳定")
    print("   - 功能开发在单独分支进行")
    print("   - 及时合并和删除已完成的分支")
    
    print("\n3. 提交频率:")
    print("   - 频繁提交，每次提交只包含一个逻辑更改")
    print("   - 在完成一个功能点后立即提交")
    
    print("\n4. .gitignore文件:")
    print("   - 忽略不必要的文件（如编译产物、日志文件等）")
    print("   - 保护敏感信息不被提交")

def print_useful_aliases():
    """打印有用的Git别名"""
    print("\n\n有用的Git别名")
    print("="*30)
    
    print("\n设置常用别名:")
    print("   git config --global alias.st status")
    print("   git config --global alias.co checkout")
    print("   git config --global alias.br branch")
    print("   git config --global alias.ci commit")
    print("   git config --global alias.lg \"log --oneline --graph --all\"")
    
    print("\n使用别名:")
    print("   git st  # 等同于 git status")
    print("   git co <分支名>  # 等同于 git checkout <分支名>")
    print("   git lg  # 图形化查看提交历史")

def main():
    print_git_tutorial()
    print_best_practices()
    print_useful_aliases()
    
    print("\n\n💡 提示: 实际操作时请在项目目录中执行这些命令")

if __name__ == "__main__":
    main()