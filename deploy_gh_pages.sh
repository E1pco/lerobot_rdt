#!/bin/bash
set -e

echo "🔨 构建 GitBook..."
gitbook build

echo "📦 准备部署到 GitHub Pages..."
cd _book

# 初始化独立的 gh-pages 分支
if [ ! -d .git ]; then
    git init
    git checkout -b gh-pages
else
    git checkout gh-pages 2>/dev/null || git checkout -b gh-pages
fi

# 配置 git 用户（如果未配置）
git config user.name "$(git config --global user.name || echo 'GitBook Deploy')"
git config user.email "$(git config --global user.email || echo 'deploy@gitbook.local')"

# 添加 .nojekyll 文件（防止 GitHub Pages 忽略 _ 开头的文件夹）
touch .nojekyll

# 提交所有文件
git add -A
git commit -m "Deploy GitBook: $(date +'%Y-%m-%d %H:%M:%S')" || echo "No changes to commit"

# 推送到 GitHub Pages
git remote add origin https://github.com/E1pco/lerobot_rdt.git 2>/dev/null || true
git push -f origin gh-pages

echo ""
echo "✅ 部署成功！"
echo ""
echo "📝 下一步操作："
echo "   1. 访问: https://github.com/E1pco/lerobot_rdt/settings/pages"
echo "   2. 在 'Source' 下拉菜单选择 'gh-pages' 分支"
echo "   3. 点击 'Save'"
echo "   4. 等待 1-2 分钟后访问: https://e1pco.github.io/lerobot_rdt/"
echo ""
