#!/bin/bash
# 自动部署 GitBook 到 GitHub Pages

# 1. 构建静态网站
echo "Building GitBook..."
gitbook build

# 2. 初始化 git（如果还没初始化）
cd _book
if [ ! -d .git ]; then
    git init
    git checkout -b gh-pages
fi

# 3. 提交所有文件
git add -A
git commit -m "Deploy documentation $(date +'%Y-%m-%d %H:%M:%S')"

# 4. 推送到 GitHub（首次需要设置远程仓库）
# git remote add origin https://github.com/你的用户名/仓库名.git
# git push -f origin gh-pages

echo "✅ 构建完成！"
echo "📝 下一步："
echo "   1. 在 GitHub 创建新仓库（或使用现有仓库）"
echo "   2. 运行: git remote add origin https://github.com/你的用户名/仓库名.git"
echo "   3. 运行: git push -f origin gh-pages"
echo "   4. 在仓库 Settings > Pages 里选择 gh-pages 分支"
echo "   5. 几分钟后访问: https://你的用户名.github.io/仓库名/"
