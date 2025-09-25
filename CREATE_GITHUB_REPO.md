# 创建GitHub仓库并推送代码指南

## 步骤1：在GitHub上创建仓库

1. 访问 [GitHub](https://github.com/) 并登录您的账户
2. 点击右上角的 "+" 号，选择 "New repository"
3. 设置仓库信息：
   - Repository name: `unitraj_moe`
   - Description: `MOE轨迹预测模型`
   - 选择 "Public"（公开）或 "Private"（私有）
   - **不要**勾选 "Initialize this repository with a README"
   - **不要**添加 `.gitignore` 或许可证
4. 点击 "Create repository"

## 步骤2：推送本地代码到GitHub

创建仓库后，您会看到一个页面，上面有推送现有仓库的说明。按照以下步骤操作：

```bash
# 设置Git用户信息（如果尚未设置）
git config --global user.name "您的GitHub用户名"
git config --global user.email "您的邮箱"

# 添加远程仓库地址（将URL替换为您刚创建的仓库地址）
git remote add origin https://github.com/您的用户名/unitraj_moe.git

# 确保您在main分支上
git checkout main

# 推送main分支到GitHub
git push -u origin main
```

## 步骤3：验证推送

推送完成后，刷新GitHub仓库页面，您应该能看到所有文件都已上传。

## 故障排除

### 如果遇到权限问题

1. 使用GitHub个人访问令牌：
   - 在GitHub上生成个人访问令牌（Settings > Developer settings > Personal access tokens）
   - 使用令牌而不是密码进行身份验证

2. 或者使用SSH：
   - 设置SSH密钥
   - 使用SSH URL而不是HTTPS URL

### 如果遇到推送拒绝

如果遇到 "Updates were rejected" 错误，可以尝试：

```bash
# 获取远程更改
git fetch origin

# 合并远程更改
git merge origin/main

# 再次推送
git push -u origin main
```

## 后续开发流程

1. 创建功能分支进行开发：
   ```bash
   git checkout -b feature/新功能名称
   ```

2. 提交更改：
   ```bash
   git add .
   git commit -m "描述您的更改"
   ```

3. 推送到GitHub：
   ```bash
   git push origin feature/新功能名称
   ```

4. 在GitHub上创建Pull Request进行代码审查

## 有用的Git命令

- 查看状态：`git status`
- 查看提交历史：`git log --oneline`
- 查看差异：`git diff`
- 切换分支：`git checkout 分支名`
- 查看远程仓库：`git remote -v`